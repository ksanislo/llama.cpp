#pragma once

#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <vector>

// the KV caches made while this is alive (by one thread) are elastic, see llama_context_params.kv_elastic
struct llama_kv_elastic_scope {
    explicit llama_kv_elastic_scope(bool enable);
    ~llama_kv_elastic_scope();

    bool prev;
};

bool llama_kv_elastic_requested();

// K/V memory of each stream is backed as the stream grows and given back when it shrinks
// the tensors keep their full [n_embd, kv_size, n_stream] layout in reserved address space (CUDA VMM)
// unbacked parts read as zeros, so the attention over free (masked) cells stays finite
class llama_kv_elastic {
public:
    llama_kv_elastic(uint32_t n_stream, uint32_t kv_size);

    // allocate the tensors of ctx with no memory behind them yet, nullptr if buft cannot do this
    ggml_backend_buffer_t alloc(ggml_context * ctx, ggml_backend_buffer_type_t buft);

    bool owns(ggml_backend_buffer_t buf) const;

    // t: [n_embd, kv_size, n_stream] in a buffer from alloc()
    // full: the stream needs all its rows as soon as it holds one cell (transposed V)
    void add(const ggml_tensor * t, bool full);

    bool empty() const { return regions.empty(); }

    // back cells [0, n_cells) of stream s
    bool reserve(uint32_t s, uint32_t n_cells);

    // give back the memory of stream s above n_cells
    void trim(uint32_t s, uint32_t n_cells);

    uint32_t get_backed(uint32_t s) const { return n_backed[s]; }

    // bytes to report for buf in the memory breakdown
    size_t memory_size(ggml_backend_buffer_t buf, ggml_backend_buffer_type_t buft, size_t size, bool no_alloc) const;

    size_t backed_bytes() const;

private:
    typedef ggml_backend_buffer_t (*alloc_fn_t)(ggml_backend_buffer_type_t buft, size_t size);
    typedef bool   (*commit_fn_t)    (ggml_backend_buffer_t buf, size_t offset, size_t size);
    typedef size_t (*release_fn_t)   (ggml_backend_buffer_t buf, size_t offset, size_t size);
    typedef size_t (*backed_fn_t)    (ggml_backend_buffer_t buf);
    typedef size_t (*granularity_fn_t)(ggml_backend_buffer_type_t buft);

    struct vmm_fns {
        alloc_fn_t       alloc       = nullptr;
        commit_fn_t      commit      = nullptr;
        release_fn_t     release     = nullptr;
        backed_fn_t      backed      = nullptr;
        granularity_fn_t granularity = nullptr;
    };

    struct buffer_info {
        ggml_backend_buffer_t buf;
        ggml_backend_dev_t    dev;
        vmm_fns               fns;
    };

    struct region {
        size_t ibuf;
        size_t offs; // tensor offset in the buffer
        size_t nb1;  // bytes per cell
        size_t nb2;  // bytes per stream
        bool   full;
    };

    static vmm_fns get_fns(ggml_backend_buffer_type_t buft);

    // [lo, hi) of stream s in region r for cells [c0, c1)
    void range(const region & r, uint32_t s, uint32_t c0, uint32_t c1, size_t & lo, size_t & hi) const;

    void release_above(uint32_t s, uint32_t n_cells);

    uint32_t round_up(uint32_t n_cells) const { return (n_cells + step - 1)/step*step; }

    const uint32_t n_stream;
    const uint32_t kv_size;

    uint32_t step;     // env LLAMA_KV_ELASTIC_STEP: cells backed at a time
    size_t   headroom; // env LLAMA_KV_ELASTIC_HEADROOM_MIB: device memory left free for compute

    std::vector<buffer_info> bufs;
    std::vector<region>      regions;
    std::vector<uint32_t>    n_backed; // [n_stream] cells
};
