#include "llama-kv-cache-elastic.h"

#include "llama-impl.h"
#include "llama-kv-cache.h"

#include <algorithm>
#include <cstdlib>
#include <map>

static thread_local bool g_kv_elastic = false;

llama_kv_elastic_scope::llama_kv_elastic_scope(bool enable) : prev(g_kv_elastic) {
    g_kv_elastic = enable;
}

llama_kv_elastic_scope::~llama_kv_elastic_scope() {
    g_kv_elastic = prev;
}

bool llama_kv_elastic_requested() {
    return g_kv_elastic;
}

//
// llama_kv_elastic
//

llama_kv_elastic::llama_kv_elastic(uint32_t n_stream, uint32_t kv_size) : n_stream(n_stream), kv_size(kv_size) {
    const char * env_step = getenv("LLAMA_KV_ELASTIC_STEP");
    const char * env_room = getenv("LLAMA_KV_ELASTIC_HEADROOM_MIB");

    step     = env_step ? std::max(1, atoi(env_step)) : 4096;
    headroom = size_t(env_room ? std::max(0, atoi(env_room)) : 512)*1024*1024;

    n_backed.assign(n_stream, 0);
}

llama_kv_elastic::vmm_fns llama_kv_elastic::get_fns(ggml_backend_buffer_type_t buft) {
    vmm_fns res;

    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
    if (dev == nullptr || ggml_backend_buft_is_host(buft)) {
        return res;
    }

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg == nullptr) {
        return res;
    }

    res.alloc       = (alloc_fn_t)       ggml_backend_reg_get_proc_address(reg, "ggml_backend_vmm_buffer_alloc");
    res.commit      = (commit_fn_t)      ggml_backend_reg_get_proc_address(reg, "ggml_backend_vmm_buffer_commit");
    res.release     = (release_fn_t)     ggml_backend_reg_get_proc_address(reg, "ggml_backend_vmm_buffer_release");
    res.backed      = (backed_fn_t)      ggml_backend_reg_get_proc_address(reg, "ggml_backend_vmm_buffer_get_backed_size");
    res.granularity = (granularity_fn_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_vmm_get_granularity");

    if (!res.alloc || !res.commit || !res.release || !res.backed || !res.granularity || res.granularity(buft) == 0) {
        return {};
    }

    return res;
}

ggml_backend_buffer_t llama_kv_elastic::alloc(ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    const vmm_fns fns = get_fns(buft);
    if (!fns.alloc) {
        return nullptr;
    }

    // each tensor starts on a granule, so tensors never share one
    const size_t gran = fns.granularity(buft);

    size_t size = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        if (t->view_src == nullptr) {
            size = GGML_PAD(size, gran) + ggml_backend_buft_get_alloc_size(buft, t);
        }
    }

    ggml_backend_buffer_t buf = fns.alloc(buft, size);
    if (buf == nullptr) {
        return nullptr;
    }

    char * base = (char *) ggml_backend_buffer_get_base(buf);

    size_t offs = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        if (t->view_src == nullptr) {
            offs = GGML_PAD(offs, gran);
            if (ggml_backend_tensor_alloc(buf, t, base + offs) != GGML_STATUS_SUCCESS) {
                ggml_backend_buffer_free(buf);
                return nullptr;
            }
            offs += ggml_backend_buft_get_alloc_size(buft, t);
        }
    }

    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        if (t->view_src != nullptr && t->buffer == nullptr) {
            if (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                ggml_backend_buffer_free(buf);
                return nullptr;
            }
        }
    }

    bufs.push_back({ buf, ggml_backend_buft_get_device(buft), fns });

    return buf;
}

bool llama_kv_elastic::owns(ggml_backend_buffer_t buf) const {
    return std::any_of(bufs.begin(), bufs.end(), [&](const buffer_info & b) { return b.buf == buf; });
}

void llama_kv_elastic::add(const ggml_tensor * t, bool full) {
    GGML_ASSERT(t->ne[1] == kv_size && t->ne[2] == n_stream);

    for (size_t i = 0; i < bufs.size(); ++i) {
        if (bufs[i].buf == t->buffer) {
            const size_t offs = (const char *) t->data - (const char *) ggml_backend_buffer_get_base(t->buffer);
            regions.push_back({ i, offs, t->nb[1], t->nb[2], full });
            return;
        }
    }

    GGML_ABORT("tensor %s is not in an elastic buffer", t->name);
}

void llama_kv_elastic::range(const region & r, uint32_t s, uint32_t c0, uint32_t c1, size_t & lo, size_t & hi) const {
    if (r.full) {
        c0 = c0 == 0 ? 0 : kv_size;
        c1 = c1 == 0 ? 0 : kv_size;
    }

    lo = r.offs + s*r.nb2 + c0*r.nb1;
    hi = r.offs + s*r.nb2 + c1*r.nb1;
}

bool llama_kv_elastic::reserve(uint32_t s, uint32_t n_cells) {
    GGML_ASSERT(s < n_stream);

    const uint32_t n_old = n_backed[s];
    const uint32_t n_new = std::min(kv_size, round_up(n_cells));

    if (n_cells <= n_old) {
        return true;
    }

    // keep the headroom free on every device, the compute pool grows on demand too
    std::map<ggml_backend_dev_t, size_t> need;
    for (const auto & r : regions) {
        size_t lo, hi;
        range(r, s, n_old, n_new, lo, hi);
        need[bufs[r.ibuf].dev] += hi - lo;
    }

    for (const auto & [dev, n] : need) {
        size_t free, total;
        ggml_backend_dev_memory(dev, &free, &total);

        if (free < n + headroom) {
            LLAMA_LOG_WARN("%s: stream %u: %u -> %u cells needs %.1f MiB on %s, free %.1f MiB (headroom %.1f MiB)\n", __func__,
                    s, n_old, n_new, n/1024.0/1024.0, ggml_backend_dev_name(dev), free/1024.0/1024.0, headroom/1024.0/1024.0);
            return false;
        }
    }

    for (const auto & r : regions) {
        size_t lo, hi;
        range(r, s, n_old, n_new, lo, hi);

        const auto & b = bufs[r.ibuf];
        if (!b.fns.commit(b.buf, lo, hi - lo)) {
            LLAMA_LOG_WARN("%s: stream %u: out of device memory on %s at %u -> %u cells\n", __func__, s, ggml_backend_dev_name(b.dev), n_old, n_new);
            release_above(s, n_old);
            return false;
        }
    }

    LLAMA_LOG_DEBUG("%s: stream %u: %u -> %u cells, backed %.1f MiB\n", __func__, s, n_old, n_new, backed_bytes()/1024.0/1024.0);

    n_backed[s] = n_new;

    return true;
}

void llama_kv_elastic::release_above(uint32_t s, uint32_t n_cells) {
    for (const auto & r : regions) {
        size_t lo, hi;
        range(r, s, n_cells, kv_size, lo, hi);

        const auto & b = bufs[r.ibuf];
        b.fns.release(b.buf, lo, hi - lo);
    }
}

void llama_kv_elastic::trim(uint32_t s, uint32_t n_cells) {
    GGML_ASSERT(s < n_stream);

    // keep one step above the rounded size, so that a few removed tokens do not unmap and map again
    const uint32_t n_keep = n_cells == 0 ? 0 : std::min(kv_size, round_up(n_cells) + step);

    if (n_keep >= n_backed[s]) {
        return;
    }

    LLAMA_LOG_DEBUG("%s: stream %u: %u -> %u cells\n", __func__, s, n_backed[s], n_keep);

    release_above(s, n_keep);

    n_backed[s] = n_keep;
}

size_t llama_kv_elastic::memory_size(ggml_backend_buffer_t buf, ggml_backend_buffer_type_t buft, size_t size, bool no_alloc) const {
    if (no_alloc) {
        // plan for one stream at full size, the others grow into what is left
        return get_fns(buft).alloc ? size/n_stream : size;
    }

    for (const auto & b : bufs) {
        if (b.buf == buf) {
            return b.fns.backed(buf);
        }
    }

    return size;
}

size_t llama_kv_elastic::backed_bytes() const {
    size_t res = 0;
    for (const auto & b : bufs) {
        res += b.fns.backed(b.buf);
    }
    return res;
}

//
// llama_kv_cache: elastic hooks
//

void llama_kv_cache::elastic_init() {
    for (const auto & layer : layers) {
        if (layer.k && elastic->owns(layer.k->buffer)) {
            elastic->add(layer.k, false);
        }
        if (layer.v && elastic->owns(layer.v->buffer)) {
            elastic->add(layer.v, v_trans);
        }
    }

    if (elastic->empty()) {
        elastic.reset();
        LLAMA_LOG_WARN("%s: kv_elastic: no device buffer supports it, the KV cache is allocated in full\n", __func__);
        return;
    }

    if (v_trans) {
        LLAMA_LOG_WARN("%s: kv_elastic: the V cache is transposed (no flash attention), a stream backs all its V rows when used\n", __func__);
    }

    LLAMA_LOG_INFO("%s: kv_elastic: %u streams x %u cells reserved, memory backed on use\n", __func__, n_stream, get_size());
}

bool llama_kv_cache::elastic_reserve(const slot_info & sinfo) {
    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        if (sinfo.idxs[s].empty()) {
            continue;
        }

        const uint32_t n = *std::max_element(sinfo.idxs[s].begin(), sinfo.idxs[s].end()) + 1;

        if (!elastic->reserve(sinfo.strm[s], n)) {
            return false;
        }
    }

    return true;
}

void llama_kv_cache::elastic_trim() {
    for (uint32_t s = 0; s < n_stream; ++s) {
        elastic->trim(s, v_cells[s].used_max_p1());
    }
}

bool llama_kv_cache::elastic_copy_prepare(uint32_t ssrc, uint32_t sdst) {
    // the whole stream is copied: dst must be backed wherever src is, the rest copies zeros onto zeros
    return elastic->reserve(sdst, elastic->get_backed(ssrc));
}
