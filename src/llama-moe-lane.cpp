#include "llama-moe-lane.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <immintrin.h>
#define LLAMA_MOE_LANE_PAUSE() _mm_pause()
#else
#define LLAMA_MOE_LANE_PAUSE() std::this_thread::yield()
#endif

// the channel words are written by a device over PCIe (or by another thread)
static uint32_t lane_load(const uint8_t * chan, int word) {
    return reinterpret_cast<const std::atomic<uint32_t> *>(chan)[word].load(std::memory_order_acquire);
}

static void lane_store(uint8_t * chan, int word, uint32_t v) {
    reinterpret_cast<std::atomic<uint32_t> *>(chan)[word].store(v, std::memory_order_release);
}

struct llama_moe_lane::impl {
    struct graph {
        ggml_context * ctx  = nullptr;
        ggml_cgraph  * gf   = nullptr;
        ggml_gallocr_t galloc = nullptr;
    };

    struct slot {
        std::atomic<bool> ready { false };
        llama_moe_lane_layer l;
        size_t off_ids  = 0;
        size_t off_x    = 0;
        size_t off_rows = 0;
        uint32_t last   = 0;          // sequence word of the last request seen
        std::vector<int32_t> ids;     // [n_used, max_tokens] e for the lane experts, -1 for the others
        std::vector<graph> graphs;    // [max_tokens + 1] built on first use
    };

    std::vector<slot> slots;

    struct pool {
        ggml_backend_t backend = nullptr;
        ggml_threadpool_t threadpool = nullptr;
    };
    std::vector<pool> pools;
    void (*threadpool_free)(ggml_threadpool_t) = nullptr;
    int n_threads = 1;

    std::thread thread;
    std::atomic<bool> stop { false };

    std::atomic<uint64_t> n_requests { 0 };
    std::atomic<uint64_t> n_experts  { 0 };
    std::atomic<uint64_t> n_rows     { 0 };
    std::atomic<uint64_t> busy_us    { 0 };

    int64_t spin_us = 50000; // polling without a pause between requests, then sleeps of 100 us

    std::vector<int> cpus; // the polling thread's CPUs (one pool only); empty: the default affinity

    // pins the calling thread to cpus; with save, its old affinity is kept in old_set first
#if defined(__linux__)
    bool pin(bool save, cpu_set_t * old_set) const {
        if (cpus.empty()) {
            return false;
        }
        if (save && pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), old_set) != 0) {
            return false;
        }
        cpu_set_t set;
        CPU_ZERO(&set);
        for (int c : cpus) {
            if (c >= 0 && c < CPU_SETSIZE) {
                CPU_SET(c, &set);
            }
        }
        return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
    }
#endif

    impl(size_t n_slots, int n_threads, int poll, const std::vector<std::vector<int>> & pool_cpus, bool numa_affinity) :
            slots(n_slots), n_threads(std::max(1, n_threads)) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (dev == nullptr) {
            throw std::runtime_error("MoE CPU lane: no CPU backend");
        }
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        auto tp_new   = (ggml_threadpool_t (*)(ggml_threadpool_params *)) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
        auto tp_set   = (void (*)(ggml_backend_t, ggml_threadpool_t))   ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
        auto tp_numa  = (void (*)(ggml_threadpool_t, bool))             ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_set_numa_affinity");
        auto set_nth  = (ggml_backend_set_n_threads_t)                   ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        threadpool_free = (void (*)(ggml_threadpool_t))                  ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free");
        if (pool_cpus.size() == 1) {
            cpus = pool_cpus[0];
        }
        for (const std::vector<int> & pc : pool_cpus.empty() ? std::vector<std::vector<int>>(1) : pool_cpus) {
            pool p;
            p.backend = ggml_backend_dev_init(dev, nullptr);
            if (p.backend == nullptr) {
                throw std::runtime_error("MoE CPU lane: failed to initialize the CPU backend");
            }
            if (set_nth) {
                set_nth(p.backend, this->n_threads);
            }
            if (tp_new && tp_set && threadpool_free) {
                ggml_threadpool_params tpp = ggml_threadpool_params_default(this->n_threads);
                tpp.poll = (uint32_t) std::clamp(poll, 0, 100);
                for (int c : pc) {
                    if (c >= 0 && c < GGML_MAX_N_THREADS) {
                        tpp.cpumask[c] = true;
                    }
                }
                tpp.strict_cpu = false;
#if defined(__linux__)
                // ggml_threadpool_new also pins the calling thread (the pool's thread 0): keep the caller's affinity
                cpu_set_t old_set;
                const bool saved = !pc.empty() && pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &old_set) == 0;
#endif
                p.threadpool = tp_new(&tpp);
#if defined(__linux__)
                if (saved) {
                    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &old_set);
                }
#endif
                if (p.threadpool) {
                    if (!numa_affinity && tp_numa) {
                        tp_numa(p.threadpool, false);
                    }
                    tp_set(p.backend, p.threadpool);
                }
            }
            pools.push_back(p);
        }
        if (const char * env = getenv("LLAMA_MOE_CPU_LANE_SPIN_US")) {
            spin_us = std::max<int64_t>(0, atoll(env));
        }

        thread = std::thread([this] { run(); });
    }

    ~impl() {
        stop.store(true);
        if (thread.joinable()) {
            thread.join();
        }
        for (slot & s : slots) {
            for (graph & g : s.graphs) {
                if (g.galloc) {
                    ggml_gallocr_free(g.galloc);
                }
                if (g.ctx) {
                    ggml_free(g.ctx);
                }
            }
        }
        for (pool & p : pools) {
            ggml_backend_free(p.backend);
            if (p.threadpool) {
                threadpool_free(p.threadpool);
            }
        }
    }

    void set(size_t i, const llama_moe_lane_layer & l) {
        slot & s = slots.at(i);
        if (s.ready.load(std::memory_order_acquire)) {
            return;
        }
        GGML_ASSERT(l.chan && l.down && (l.gate_up || (l.up && l.gate)));
        GGML_ASSERT(l.pool >= 0 && (size_t) l.pool < pools.size());
        s.l = l;
        ggml_moe_lane_layout(l.n_embd, l.n_used, l.max_tokens, &s.off_ids, &s.off_x, &s.off_rows);
        s.last = lane_load(l.chan, GGML_MOE_LANE_SEQ);
        s.ids.assign((size_t) l.n_used*l.max_tokens, -1);
        s.graphs.resize(l.max_tokens + 1);
        s.ready.store(true, std::memory_order_release);
    }

    // the experts of the lane for n tokens: x and ids in place, the result written to the rows of the channel
    graph & get_graph(slot & s, int64_t n) {
        graph & g = s.graphs[n];
        if (g.gf) {
            return g;
        }
        const llama_moe_lane_layer & l = s.l;
        const size_t n_nodes = 16;
        ggml_init_params ip = {
            /*.mem_size   =*/ ggml_tensor_overhead()*n_nodes*2 + ggml_graph_overhead_custom(n_nodes, false),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        g.ctx = ggml_init(ip);
        ggml_context * ctx = g.ctx;

        auto weights = [&](const ggml_tensor * w) {
            ggml_tensor * t = ggml_new_tensor(ctx, w->type, 3, w->ne);
            GGML_ASSERT(t->nb[1] == w->nb[1] && t->nb[2] == w->nb[2]);
            t->data = w->data;
            return t;
        };

        ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, l.n_embd, 1, n);
        x->data = l.chan + s.off_x;
        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, l.n_used, n);
        ids->data = s.ids.data();

        ggml_tensor * up;
        ggml_tensor * gate;
        if (l.gate_up) {
            ggml_tensor * gu = ggml_mul_mat_id(ctx, weights(l.gate_up), x, ids);
            const int64_t n_ff = gu->ne[0]/2;
            gate = ggml_view_3d(ctx, gu, n_ff, gu->ne[1], gu->ne[2], gu->nb[1], gu->nb[2], 0);
            up   = ggml_view_3d(ctx, gu, n_ff, gu->ne[1], gu->ne[2], gu->nb[1], gu->nb[2], n_ff*gu->nb[0]);
        } else {
            up   = ggml_mul_mat_id(ctx, weights(l.up),   x, ids);
            gate = ggml_mul_mat_id(ctx, weights(l.gate), x, ids);
        }
        ggml_tensor * act = l.clamp > 0.0f ? ggml_swiglu_clamp(ctx, gate, up, l.clamp) : ggml_swiglu_split(ctx, gate, up);
        ggml_tensor * out = ggml_mul_mat_id(ctx, weights(l.down), act, ids); // [n_embd, n_used, n]
        GGML_ASSERT(out->ne[0] == l.n_embd && ggml_is_contiguous(out));
        out->data = l.chan + s.off_rows;

        g.gf = ggml_new_graph_custom(ctx, n_nodes, false);
        ggml_build_forward_expand(g.gf, out);
        g.galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
        if (!ggml_gallocr_alloc_graph(g.galloc, g.gf)) {
            throw std::runtime_error("MoE CPU lane: failed to allocate a graph");
        }
        return g;
    }

    void serve(slot & s, uint32_t seq) {
        const auto t0 = std::chrono::steady_clock::now();
        const llama_moe_lane_layer & l = s.l;
        const int64_t n = std::clamp<int64_t>(lane_load(l.chan, GGML_MOE_LANE_NTOK), 1, l.max_tokens);
        const int32_t * code = (const int32_t *) (l.chan + s.off_ids);

        uint64_t rows = 0;
        uint64_t experts = 0;
        for (int64_t i = 0; i < n*l.n_used; ++i) {
            const int32_t c = code[i];
            if (c < -l.n_expert) {
                const int32_t e = -1 - l.n_expert - c;
                s.ids[i] = e;
                rows++;
                bool first = true;
                for (int64_t j = 0; j < i && first; ++j) {
                    first = s.ids[j] != e;
                }
                experts += first;
            } else {
                s.ids[i] = -1;
            }
        }
        if (rows > 0) {
            graph & g = get_graph(s, n);
            ggml_backend_graph_compute(pools[l.pool].backend, g.gf);
            n_requests.fetch_add(1, std::memory_order_relaxed);
            n_experts.fetch_add(experts, std::memory_order_relaxed);
            n_rows.fetch_add(rows, std::memory_order_relaxed);
        }
        std::atomic_thread_fence(std::memory_order_seq_cst);
        lane_store(l.chan, GGML_MOE_LANE_DONE, seq);
        busy_us.fetch_add(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count(),
            std::memory_order_relaxed);
    }

    void run() {
#if defined(__linux__)
        pin(false, nullptr); // the polling thread computes as the pool's thread 0
#endif
        auto last_active = std::chrono::steady_clock::now();
        while (!stop.load(std::memory_order_relaxed)) {
            bool any = false;
            for (slot & s : slots) {
                if (!s.ready.load(std::memory_order_acquire)) {
                    continue;
                }
                const uint32_t seq = lane_load(s.l.chan, GGML_MOE_LANE_SEQ);
                if (seq == s.last) {
                    continue;
                }
                s.last = seq;
                any = true;
                if (lane_load(s.l.chan, GGML_MOE_LANE_DONE) == seq) {
                    continue; // no lane experts, the device completed it itself
                }
                serve(s, seq);
            }
            if (any) {
                last_active = std::chrono::steady_clock::now();
            } else if (std::chrono::steady_clock::now() - last_active > std::chrono::microseconds(spin_us)) {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            } else {
                LLAMA_MOE_LANE_PAUSE();
            }
        }
    }
};

llama_moe_lane::llama_moe_lane(size_t n_slots, int n_threads, int poll, const std::vector<int> & cpus) :
    pimpl(new impl(n_slots, n_threads, poll, { cpus }, true)) {}

llama_moe_lane::llama_moe_lane(size_t n_slots, int n_threads, int poll, const std::vector<std::vector<int>> & pool_cpus, bool numa_affinity) :
    pimpl(new impl(n_slots, n_threads, poll, pool_cpus, numa_affinity)) {}

llama_moe_lane::~llama_moe_lane() = default;

void llama_moe_lane::set(size_t i, const llama_moe_lane_layer & l) {
    pimpl->set(i, l);
}

bool llama_moe_lane::is_set(size_t i) const {
    return pimpl->slots.at(i).ready.load(std::memory_order_acquire);
}

llama_moe_lane::stats llama_moe_lane::get_stats() const {
    stats st;
    st.requests = pimpl->n_requests.load();
    st.experts  = pimpl->n_experts.load();
    st.rows     = pimpl->n_rows.load();
    st.busy_ms  = pimpl->busy_us.load()/1000.0;
    for (const auto & s : pimpl->slots) {
        if (s.ready.load(std::memory_order_acquire)) {
            st.timeouts += lane_load(s.l.chan, GGML_MOE_LANE_ERR);
        }
    }
    return st;
}

int llama_moe_lane::n_threads() const {
    return pimpl->n_threads;
}
