// ggml_backend_sched submit threads (ggml_backend_sched_set_submit_threads) on the CPU: two emulated asynchronous devices
// ("streams" = one task thread per device: graphs, copies, event records/waits run later, in order, with random delays),
// plus the CPU backend, with pipeline parallelism (4 copies). A chain of layers spread over the three
// backends is computed for many graphs in flight, each graph rebuilt in the copy it allocates in, without synchronizing
// between graphs; the outputs must equal a host reference and the run without submit threads. A missing stream wait or a
// slot reused too early shows up as wrong outputs (or a hang).

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-backend-impl.h"

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <mutex>
#include <random>
#include <thread>
#include <unordered_map>
#include <vector>

// ---- emulated device -----------------------------------------------------------------------------------------------

struct tdev_stream {
    std::mutex mtx;
    std::condition_variable cv;
    std::deque<std::function<void()>> q;
    int busy = 0;
    bool stop = false;
    std::thread th;
    std::mt19937 rng;
    int compute_us = 0; // extra time of every graph (a slow pipeline stage)

    explicit tdev_stream(unsigned seed) : rng(seed) {
        th = std::thread([this] {
            while (true) {
                std::function<void()> f;
                {
                    std::unique_lock<std::mutex> lock(mtx);
                    cv.wait(lock, [&] { return stop || !q.empty(); });
                    if (q.empty()) {
                        return;
                    }
                    f = std::move(q.front());
                    q.pop_front();
                    busy = 1;
                }
                f();
                {
                    std::lock_guard<std::mutex> lock(mtx);
                    busy = 0;
                }
                cv.notify_all();
            }
        });
    }
    ~tdev_stream() {
        {
            std::lock_guard<std::mutex> lock(mtx);
            stop = true;
        }
        cv.notify_all();
        th.join();
    }
    void push(std::function<void()> f) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            q.push_back(std::move(f));
        }
        cv.notify_all();
    }
    void drain() {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&] { return q.empty() && busy == 0; });
    }
    void delay() {
        int us;
        {
            std::lock_guard<std::mutex> lock(mtx);
            us = (int) (rng() % 1500);
        }
        std::this_thread::sleep_for(std::chrono::microseconds(us));
    }
};

struct tdev_event {
    std::mutex mtx;
    std::condition_variable cv;
    int64_t recorded  = 0; // records issued
    int64_t completed = 0; // records that the stream reached
};

struct tdev_device_ctx {
    char name[16];
    ggml_backend_buffer_type buft;
};

struct tdev_backend_ctx {
    tdev_stream * stream;
    ggml_backend_t cpu;
};

static ggml_backend_device g_dev[2];
static tdev_device_ctx      g_dev_ctx[2];
static ggml_backend_reg     g_reg;

// buffers: plain host memory that the CPU backend does not accept (not is_host), so the scheduler copies between them
static void   tbuf_free(ggml_backend_buffer_t b)  { free(b->context); }
static void * tbuf_base(ggml_backend_buffer_t b)  { return b->context; }
static void   tbuf_memset(ggml_backend_buffer_t, ggml_tensor * t, uint8_t v, size_t off, size_t size) { memset((char *) t->data + off, v, size); }
static void   tbuf_set(ggml_backend_buffer_t, ggml_tensor * t, const void * d, size_t off, size_t size) { memcpy((char *) t->data + off, d, size); }
static void   tbuf_get(ggml_backend_buffer_t, const ggml_tensor * t, void * d, size_t off, size_t size) { memcpy(d, (const char *) t->data + off, size); }
static void   tbuf_clear(ggml_backend_buffer_t b, uint8_t v) { memset(b->context, v, b->size); }

static const ggml_backend_buffer_i tbuf_i = {
    /* .free_buffer   = */ tbuf_free,
    /* .get_base      = */ tbuf_base,
    /* .init_tensor   = */ NULL,
    /* .memset_tensor = */ tbuf_memset,
    /* .set_tensor    = */ tbuf_set,
    /* .get_tensor    = */ tbuf_get,
    /* .set_tensor_2d = */ NULL,
    /* .get_tensor_2d = */ NULL,
    /* .cpy_tensor    = */ NULL,
    /* .clear         = */ tbuf_clear,
    /* .reset         = */ NULL,
};

static const char * tbuft_name(ggml_backend_buffer_type_t buft) { return ((tdev_device_ctx *) buft->context)->name; }
static ggml_backend_buffer_t tbuft_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    void * p = aligned_alloc(64, GGML_PAD(size, 64) + 64);
    return ggml_backend_buffer_init(buft, tbuf_i, p, size);
}
static size_t tbuft_align(ggml_backend_buffer_type_t) { return 64; }
static bool   tbuft_is_host(ggml_backend_buffer_type_t) { return false; }

static const ggml_backend_buffer_type_i tbuft_i = {
    /* .get_name         = */ tbuft_name,
    /* .alloc_buffer     = */ tbuft_alloc,
    /* .alloc_buffer_n   = */ NULL,
    /* .get_alignment    = */ tbuft_align,
    /* .get_max_size     = */ NULL,
    /* .get_alloc_size   = */ NULL,
    /* .get_alloc_size_n = */ NULL,
    /* .is_host          = */ tbuft_is_host,
};

static bool is_tbuf(const ggml_tensor * t) {
    ggml_backend_buffer_t b = t->view_src ? t->view_src->buffer : t->buffer;
    return b && (b->buft == &g_dev_ctx[0].buft || b->buft == &g_dev_ctx[1].buft);
}

// backend: work is queued on the stream
static const char * tb_name(ggml_backend_t b) { return ((tdev_device_ctx *) b->device->context)->name; }
static void tb_free(ggml_backend_t b) {
    auto * ctx = (tdev_backend_ctx *) b->context;
    delete ctx->stream;
    ggml_backend_free(ctx->cpu);
    delete ctx;
    delete b;
}
static void tb_set_async(ggml_backend_t b, ggml_tensor * t, const void * d, size_t off, size_t size) {
    auto * s = ((tdev_backend_ctx *) b->context)->stream;
    s->push([s, t_data = (char *) t->data, d, off, size] { s->delay(); memcpy(t_data + off, d, size); });
}
static void tb_get_async(ggml_backend_t b, const ggml_tensor * t, void * d, size_t off, size_t size) {
    auto * s = ((tdev_backend_ctx *) b->context)->stream;
    s->push([s, t_data = (const char *) t->data, d, off, size] { s->delay(); memcpy(d, t_data + off, size); });
}
static void tb_sync(ggml_backend_t b) { ((tdev_backend_ctx *) b->context)->stream->drain(); }
// like a kernel launch: the graph is read now (the scheduler may rebuild its tensors once the graph is queued), so a copy of
// its tensor structs (same data pointers) runs on the stream, after the earlier work of the stream
static ggml_status tb_compute(ggml_backend_t b, ggml_cgraph * g) {
    auto * ctx = (tdev_backend_ctx *) b->context;
    const int n_nodes = ggml_graph_n_nodes(g);
    const size_t n_max = 8*(size_t) n_nodes + 64;
    ggml_init_params params = { ggml_tensor_overhead()*n_max + ggml_graph_overhead_custom(n_nodes + 1, false), nullptr, true };
    ggml_context * gctx = ggml_init(params);
    std::unordered_map<const ggml_tensor *, ggml_tensor *> map;
    std::function<ggml_tensor * (const ggml_tensor *)> clone = [&](const ggml_tensor * t) -> ggml_tensor * {
        if (t == nullptr) {
            return nullptr;
        }
        auto it = map.find(t);
        if (it != map.end()) {
            return it->second;
        }
        ggml_tensor * c = ggml_new_tensor_1d(gctx, GGML_TYPE_I8, 0);
        memcpy(c, t, sizeof(ggml_tensor));
        map[t] = c;
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            c->src[i] = clone(t->src[i]);
        }
        c->view_src = clone(t->view_src);
        return c;
    };
    ggml_cgraph * g2 = ggml_new_graph_custom(gctx, n_nodes + 1, false);
    for (int i = 0; i < n_nodes; i++) {
        ggml_graph_add_node(g2, clone(ggml_graph_node(g, i)));
    }
    ctx->stream->push([s = ctx->stream, cpu = ctx->cpu, gctx, g2] {
        s->delay();
        std::this_thread::sleep_for(std::chrono::microseconds(s->compute_us));
        ggml_backend_graph_compute(cpu, g2);
        ggml_free(gctx);
    });
    return GGML_STATUS_SUCCESS;
}
static void tb_event_record(ggml_backend_t b, ggml_backend_event_t ev) {
    auto * e = (tdev_event *) ev->context;
    int64_t n;
    {
        std::lock_guard<std::mutex> lock(e->mtx);
        n = ++e->recorded;
    }
    auto * s = ((tdev_backend_ctx *) b->context)->stream;
    s->push([e, n] {
        {
            std::lock_guard<std::mutex> lock(e->mtx);
            e->completed = std::max(e->completed, n);
        }
        e->cv.notify_all();
    });
}
static void tb_event_wait(ggml_backend_t b, ggml_backend_event_t ev) {
    auto * e = (tdev_event *) ev->context;
    int64_t n;
    {
        std::lock_guard<std::mutex> lock(e->mtx);
        n = e->recorded;
    }
    auto * s = ((tdev_backend_ctx *) b->context)->stream;
    s->push([e, n] {
        std::unique_lock<std::mutex> lock(e->mtx);
        e->cv.wait(lock, [&] { return e->completed >= n; });
    });
}

static const ggml_backend_i tb_i = {
    /* .get_name            = */ tb_name,
    /* .free                = */ tb_free,
    /* .set_tensor_async    = */ tb_set_async,
    /* .get_tensor_async    = */ tb_get_async,
    /* .set_tensor_2d_async = */ NULL,
    /* .get_tensor_2d_async = */ NULL,
    /* .cpy_tensor_async    = */ NULL,
    /* .synchronize         = */ tb_sync,
    /* .graph_plan_create   = */ NULL,
    /* .graph_plan_free     = */ NULL,
    /* .graph_plan_update   = */ NULL,
    /* .graph_plan_compute  = */ NULL,
    /* .graph_compute       = */ tb_compute,
    /* .event_record        = */ tb_event_record,
    /* .event_wait          = */ tb_event_wait,
    /* .graph_optimize      = */ NULL,
};

// the submit threads' push: queued on the source stream only
static bool tb_cpy_src_async(ggml_backend_t src_b, ggml_backend_t dst_b, const ggml_tensor * src, ggml_tensor * dst) {
    if (src_b->iface.graph_compute != tb_compute || dst_b->iface.graph_compute != tb_compute || !is_tbuf(src) || !is_tbuf(dst)) {
        return false;
    }
    auto * s = ((tdev_backend_ctx *) src_b->context)->stream;
    s->push([s, d = (char *) dst->data, p = (const char *) src->data, n = ggml_nbytes(dst)] { s->delay(); memcpy(d, p, n); });
    return true;
}

static const char * td_name(ggml_backend_dev_t d) { return ((tdev_device_ctx *) d->context)->name; }
static void td_memory(ggml_backend_dev_t, size_t * f, size_t * t) { *f = *t = 1u << 30; }
static enum ggml_backend_dev_type td_type(ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_GPU; }
static void td_props(ggml_backend_dev_t d, ggml_backend_dev_props * p) {
    memset(p, 0, sizeof(*p));
    p->name = td_name(d);
    p->type = GGML_BACKEND_DEVICE_TYPE_GPU;
    p->caps.async = true;
    p->caps.events = true;
}
static ggml_backend_buffer_type_t td_buft(ggml_backend_dev_t d) { return &((tdev_device_ctx *) d->context)->buft; }
static bool td_supports_op(ggml_backend_dev_t, const ggml_tensor *) { return true; }
static bool td_supports_buft(ggml_backend_dev_t d, ggml_backend_buffer_type_t b) { return b == td_buft(d); }
static ggml_backend_event_t td_event_new(ggml_backend_dev_t d) { return new ggml_backend_event { d, new tdev_event() }; }
static void td_event_free(ggml_backend_dev_t, ggml_backend_event_t ev) { delete (tdev_event *) ev->context; delete ev; }
static void td_event_sync(ggml_backend_dev_t, ggml_backend_event_t ev) {
    auto * e = (tdev_event *) ev->context;
    std::unique_lock<std::mutex> lock(e->mtx);
    const int64_t n = e->recorded;
    e->cv.wait(lock, [&] { return e->completed >= n; });
}

static const ggml_backend_device_i td_i = {
    /* .get_name             = */ td_name,
    /* .get_description      = */ td_name,
    /* .get_memory           = */ td_memory,
    /* .get_type             = */ td_type,
    /* .get_props            = */ td_props,
    /* .init_backend         = */ NULL,
    /* .get_buffer_type      = */ td_buft,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ td_supports_op,
    /* .supports_buft        = */ td_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ td_event_new,
    /* .event_free           = */ td_event_free,
    /* .event_synchronize    = */ td_event_sync,
};

static const char * tr_name(ggml_backend_reg_t) { return "TDEV"; }
static size_t tr_count(ggml_backend_reg_t) { return 2; }
static ggml_backend_dev_t tr_dev(ggml_backend_reg_t, size_t i) { return &g_dev[i]; }
static void * tr_proc(ggml_backend_reg_t, const char * name) {
    return strcmp(name, "ggml_backend_cpy_tensor_src_async") == 0 ? (void *) tb_cpy_src_async : nullptr;
}
static const ggml_backend_reg_i tr_i = { tr_name, tr_count, tr_dev, tr_proc };

static ggml_backend_t tdev_init(int i, unsigned seed) {
    static ggml_guid guid = { 0x5e, 0x3d };
    auto * ctx = new tdev_backend_ctx { new tdev_stream(seed), ggml_backend_cpu_init() };
    ggml_backend_cpu_set_n_threads(ctx->cpu, 1);
    return new ggml_backend { &guid, tb_i, &g_dev[i], ctx };
}

// ---- test ----------------------------------------------------------------------------------------------------------

static const int     N_LAYER = 12;
static const int64_t N_EMBD  = 8192;
// mixed: T0, T0, T1, T1, CPU, T0, T1, CPU, T1, T0, T0, T1 (CPU = backend 2)
// pipe:  T0 x 6, T1 x 6 with a slow T1: T0 runs graphs ahead, its copies into T1 must wait until T1 has read the slot's
//        previous input (the copy belongs to the pipeline copy, not to the graph)
static const int     LAYER_BACKEND[2][N_LAYER] = {
    { 0, 0, 1, 1, 2, 0, 1, 2, 1, 0, 0, 1 },
    { 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1 },
};

static float layer_scale(int il) { return 0.5f + 0.01f*il; }

static std::vector<float> make_input(int c) {
    std::vector<float> x(N_EMBD);
    for (int64_t i = 0; i < N_EMBD; i++) {
        x[i] = (float) ((c*131 + i*7) % 97) / 97.0f - 0.5f;
    }
    return x;
}

static std::vector<float> reference(const std::vector<float> & x) {
    std::vector<float> h = x;
    for (int il = 0; il < N_LAYER; il++) {
        for (int64_t i = 0; i < N_EMBD; i++) {
            h[i] = h[i]*layer_scale(il) + x[i];
        }
    }
    return h;
}

// the submit callback as llama's MoE ring uses it (ring_plan_copy / submit_graph): the scheduler's thread leaves a per-copy plan
// (here the graph number) before the graph is queued, each backend's thread takes it at begin; checks that every backend sees
// begin/end alternating, the graphs in order, always on one thread (not the caller's with threads)
#define PROBE_MAX_COPIES 16

struct submit_probe {
    std::mutex mtx;
    int plan[PROBE_MAX_COPIES] = {};
    int next[3] = {};
    bool open[3] = {};
    std::thread::id tid[3];
    ggml_backend_t backends[3] = {};
    std::thread::id main_tid;
    bool threads = false;
    int n_err = 0;
};

static void submit_probe_cb(ggml_backend_sched_t, ggml_backend_t backend, int copy, bool begin, void * user_data) {
    auto * p = (submit_probe *) user_data;
    std::lock_guard<std::mutex> lock(p->mtx);
    int b = 0;
    while (b < 3 && p->backends[b] != backend) {
        b++;
    }
    auto err = [&](const char * what) {
        if (p->n_err++ < 5) {
            fprintf(stderr, "  submit callback: backend %d copy %d %s: %s\n", b, copy, begin ? "begin" : "end", what);
        }
    };
    if (b == 3 || copy < 0 || copy >= PROBE_MAX_COPIES) {
        err("unknown backend or copy");
        return;
    }
    if (p->tid[b] == std::thread::id()) {
        p->tid[b] = std::this_thread::get_id();
    } else if (p->tid[b] != std::this_thread::get_id()) {
        err("called on another thread");
    }
    if (p->threads && std::this_thread::get_id() == p->main_tid) {
        err("called on the scheduler's thread");
    }
    if (begin) {
        if (p->open[b]) {
            err("begin twice");
        }
        if (p->plan[copy] != p->next[b]) {
            err("graph out of order");
        }
        p->next[b]++;
    } else if (!p->open[b]) {
        err("end without begin");
    }
    p->open[b] = begin;
}

// run n_graph graphs: with submit threads none of them is synchronized before the end
static bool run(int n_graph, bool threads, int mode, unsigned seed, std::vector<std::vector<float>> & out) {
    ggml_backend_t backends[3] = { tdev_init(0, seed), tdev_init(1, seed + 1), ggml_backend_cpu_init() };
    if (mode == 1) {
        ((tdev_backend_ctx *) backends[1]->context)->stream->compute_us = 4000;
    }
    ggml_backend_cpu_set_n_threads(backends[2], 2);
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, NULL, 3, 4096, true, false);
    if (ggml_backend_sched_get_n_copies(sched) < 2) {
        fprintf(stderr, "expected pipeline copies\n");
        return false;
    }
    if (threads && !ggml_backend_sched_set_submit_threads(sched, true)) {
        fprintf(stderr, "submit threads refused\n");
        return false;
    }
    submit_probe probe;
    probe.threads  = threads;
    probe.main_tid = std::this_thread::get_id();
    for (int b = 0; b < 3; b++) {
        probe.backends[b] = backends[b];
    }
    if (threads) {
        ggml_backend_sched_set_submit_callback(sched, submit_probe_cb, &probe);
    }

    const size_t ctx_size = ggml_tensor_overhead()*256 + ggml_graph_overhead();
    ggml_context * ctxs[4] = { nullptr, nullptr, nullptr, nullptr };
    out.assign(n_graph, std::vector<float>(N_EMBD, 0.0f));
    std::vector<std::vector<float>> inputs(n_graph);

    bool ok = true;
    for (int c = 0; c < n_graph && ok; c++) {
        // the graph of a copy is rebuilt once the threads are done with that copy's previous graph
        const int k = threads ? ggml_backend_sched_wait_next_copy(sched) : c % 4;
        ggml_free(ctxs[k]);
        ggml_init_params params = { ctx_size, nullptr, true };
        ctxs[k] = ggml_init(params);
        ggml_context * ctx = ctxs[k];

        ggml_tensor * inp = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N_EMBD);
        ggml_set_name(inp, "inp");
        ggml_set_input(inp);
        ggml_tensor * h = inp;
        ggml_cgraph * gf = ggml_new_graph(ctx);
        std::vector<std::pair<ggml_tensor *, int>> assign;
        for (int il = 0; il < N_LAYER; il++) {
            ggml_tensor * s = ggml_scale(ctx, h, layer_scale(il));
            ggml_format_name(s, "scale-%d", il);
            h = ggml_add(ctx, s, inp);
            ggml_format_name(h, "add-%d", il);
            assign.push_back({ s, LAYER_BACKEND[mode][il] });
            assign.push_back({ h, LAYER_BACKEND[mode][il] });
        }
        ggml_set_output(h);
        ggml_build_forward_expand(gf, h);

        ggml_backend_sched_reset(sched);
        for (auto & a : assign) {
            ggml_backend_sched_set_tensor_backend(sched, a.first, backends[a.second]);
        }
        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            fprintf(stderr, "alloc failed\n");
            ok = false;
            break;
        }
        if (threads && ggml_backend_sched_get_copy(sched) != k) {
            fprintf(stderr, "graph %d: allocated in copy %d, expected %d\n", c, ggml_backend_sched_get_copy(sched), k);
            ok = false;
            break;
        }
        inputs[c] = make_input(c);
        ggml_backend_tensor_set(inp, inputs[c].data(), 0, ggml_nbytes(inp));
        {
            std::lock_guard<std::mutex> lock(probe.mtx);
            probe.plan[ggml_backend_sched_get_copy(sched)] = c;
        }

        if (ggml_backend_sched_graph_compute_async(sched, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "compute failed\n");
            ok = false;
            break;
        }
        ggml_backend_t ob = ggml_backend_sched_get_tensor_backend(sched, h);
        ggml_backend_sched_tensor_get_async(sched, ob, h, out[c].data(), 0, ggml_nbytes(h));
        ggml_backend_sched_flush(sched);
        if (!threads) {
            ggml_backend_sched_synchronize(sched); // (the reads of the outputs into out[c] must be done before ctxs[k] is freed)
        }
    }
    if (threads) {
        ggml_backend_sched_set_submit_threads(sched, false);
        for (int b = 0; b < 3; b++) {
            if (probe.open[b] || probe.next[b] != n_graph) {
                fprintf(stderr, "  submit callback: backend %d saw %d of %d graphs%s\n", b, probe.next[b], n_graph, probe.open[b] ? ", one not ended" : "");
                probe.n_err++;
            }
        }
        if (probe.n_err > 0) {
            ok = false;
        }
    }
    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_free(sched);
    for (ggml_context * ctx : ctxs) {
        ggml_free(ctx);
    }
    for (ggml_backend_t b : backends) {
        ggml_backend_free(b);
    }
    return ok;
}

int main(int argc, char ** argv) {
    const int n_graph = argc > 1 ? atoi(argv[1]) : 24;
    const int n_round = argc > 2 ? atoi(argv[2]) : 3;

    g_reg = { GGML_BACKEND_API_VERSION, tr_i, nullptr };
    for (int i = 0; i < 2; i++) {
        snprintf(g_dev_ctx[i].name, sizeof(g_dev_ctx[i].name), "TDEV%d", i);
        g_dev_ctx[i].buft = { tbuft_i, &g_dev[i], &g_dev_ctx[i] };
        g_dev[i] = { td_i, &g_reg, &g_dev_ctx[i] };
    }

    int n_fail = 0;
    for (int r = 0; r < n_round; r++) {
        for (int mt = 0; mt < 4; mt++) {
            const int mode    = mt / 2;
            const int threads = mt % 2;
            std::vector<std::vector<float>> out;
            const auto t0 = std::chrono::steady_clock::now();
            if (!run(n_graph, threads != 0, mode, 1234 + r*17, out)) {
                n_fail++;
                continue;
            }
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            int n_bad = 0;
            for (int c = 0; c < n_graph; c++) {
                const std::vector<float> ref = reference(make_input(c));
                for (int64_t i = 0; i < N_EMBD; i++) {
                    if (std::fabs(out[c][i] - ref[i]) > 1e-4f*(1.0f + std::fabs(ref[i]))) {
                        if (n_bad < 5) {
                            fprintf(stderr, "  graph %d [%" PRId64 "]: %f, expected %f\n", c, i, out[c][i], ref[i]);
                        }
                        n_bad++;
                        break;
                    }
                }
            }
            printf("round %d, %s, submit threads %d: %d graphs, %d wrong, %.0f ms\n", r, mode ? "pipe " : "mixed", threads, n_graph, n_bad, ms);
            n_fail += n_bad > 0;
        }
    }
    printf("%s\n", n_fail == 0 ? "OK" : "FAILED");
    return n_fail == 0 ? 0 : 1;
}
