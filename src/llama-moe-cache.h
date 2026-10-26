#pragma once

#include "ggml-backend.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

struct llama_model;

// keeps recently and repeatedly used experts of host-resident MoE layers in a device buffer
// each layer has a slot map in host memory: when the scheduler copies it to the device, the copy callback uploads the missing experts
// with LLAMA_MOE_ASYNC=n, the batches the cache does not serve read the experts of a layer from one of n staging slots per device instead,
// filled on a separate stream of the device; large batches stage all the experts of the next layer while the current one computes
// with LLAMA_MOE_ASYNC_BORROW=1 the staging slots lie in cache slots of the device, which large batches borrow and the next cached batch gives back
// with LLAMA_MOE_RING=K the K staging slots are a ring: every uncached expert of a device's staged layers streams on the copy stream in
// layer order, independent of the routing, as far ahead as the slots allow, with no host step per layer (borrows the cache by default)
// with LLAMA_MOE_RING_TENSORS=1 the ring's entries are the expert tensors (gate, up, down) with slots of their own (default one each,
// LLAMA_MOE_RING_TENSOR_SLOTS), each released as soon as the next tensor's matmul is queued: ~1 layer of staging instead of K
// with LLAMA_MOE_RING_INPLACE=1 in addition the cached experts are read in place from the cache (ggml_mul_mat_id_slots over a per-layer
// map) and the staging slots hold only the uncached ones (LLAMA_MOE_RING_INPLACE_CAP places, the rest read from host memory directly)
// with LLAMA_MOE_PREFETCH=k the router of the next layer is applied to the input of the current one, and the missing experts among its top k
// are uploaded on a second stream while the current layer and the attention of the next one compute (decode, cached batches only)
// with LLAMA_MOE_DEVICE_HITS=1|2 small batches look the slots up on the device and read the misses straight from the pinned host experts,
// so their graph has no host step; the experts they select are logged on the device and the next graph fills the cache from the log
// with LLAMA_MOE_CPU_LANE=f in addition, a share f of their misses is computed by a CPU thread pool while the device computes the rest
// with LLAMA_MOE_DECODE_MEM=1 the cache of a device also holds the part of its compute buffer that only large batches use: the
// tails of its banks are released while such a batch runs and that compute memory is backed instead, then the reverse
// with LLAMA_MOE_CACHE_SPLIT=miss the budget is split among the devices by the profiled routings of their experts

// the compute buffers of the context, measured with a cache built without memory (measure)
struct llama_moe_cache_compute {
    bool measure = false;
    std::vector<size_t> large; // [backends] compute buffer of the worst-case batch
    std::vector<size_t> small; // [backends] compute buffer of a small batch (LLAMA_MOE_DECODE_MEM_TOKENS)
    std::vector<size_t> reserve; // [backends] memory to leave free (llama_context_params::moe_cache_reserve)
};

class llama_moe_cache {
public:
    // backends are all the backends of the context, each GPU gets its own cache of the given size for the layers assigned to it
    // cpu_share is the share of the misses computed on the CPU instead of uploaded
    // profile: expert usage profile (see --moe-cache-profile), empty = LLAMA_MOE_CACHE_PROFILE or none
    // draft: the cache of a draft / MTP context (--moe-cache-draft): device hits from a small cache, no ring / staging / decode memory,
    // settings read from LLAMA_MOE_DRAFT_X first; LLAMA_MOE_DRAFT_CACHE_DEVICE=<name> caches the layers on that GPU instead of theirs
    llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, float cpu_share,
            const std::string & profile = "", const llama_moe_cache_compute & compute = {}, bool draft = false);
    ~llama_moe_cache();

    // whether the context should measure its compute buffers first (LLAMA_MOE_DECODE_MEM, LLAMA_MOE_CACHE_SPLIT=miss)
    static bool wants_compute_sizes();
    // tokens of the small batch to measure
    static int32_t small_batch_tokens();

    // LLAMA_MOE_DECODE_MEM: the compute buffer type for the scheduler on this backend, nullptr to keep the default
    ggml_backend_buffer_type_t compute_buft(ggml_backend_t backend) const;
    // how much larger than a graph's own tensors its compute buffer plan may be before it is re-planned (0: any)
    float compact_ratio() const;
    // ggml_backend_sched alloc callback: back the compute memory the graph uses, lending the cache's tails for large batches
    void on_alloc(ggml_backend_sched_t sched);
    // a graph was built only to reserve memory (no alloc callback follows): forget what it read
    void reserved();
    // decode(): the batch holds more tokens than a small (decode) batch, e.g. other slots' tokens beside a prompt chunk
    void set_batch_large(bool large);

    // the device that caches layer il
    ggml_backend_t backend(int32_t il) const;

    // the slot map of layer il, if its experts can be read from the cache or the staging slots for n_tokens tokens, nullptr otherwise
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const;

    // the experts of w in the cache, nullptr if w is not cached
    ggml_tensor * get_experts(const ggml_tensor * w) const;

    // the experts of w to read with the slots of get_slot_map: in the cache or in a staging slot
    ggml_tensor * get_experts(const ggml_tensor * w, int64_t n_tokens, int64_t n_expert_used) const;

    // whether the slot map of layer il for n_tokens tokens is the cache's (else the staging map of LLAMA_MOE_ASYNC)
    bool uses_cache(int32_t il, int64_t n_tokens, int64_t n_expert_used) const;

    // the experts of w in its staging slot, nullptr if w is not staged
    ggml_tensor * get_staged_experts(const ggml_tensor * w) const;

    // LLAMA_MOE_RING_TENSORS: whether the staged batches of n_tokens tokens of layer il look their experts up per expert tensor,
    // and the staging map of w for that lookup (nullptr if not); each lookup starts a split right before its matmul, so the ring
    // is called back once per expert tensor
    bool stage_per_tensor(int32_t il, int64_t n_tokens, int64_t n_expert_used) const;
    ggml_tensor * get_stage_map(const ggml_tensor * w, int64_t n_tokens, int64_t n_expert_used, ggml_backend_t * backend) const;

    // LLAMA_MOE_RING_INPLACE: whether the staged batches of n_tokens tokens of layer il (of the layer of w) read the experts with
    // ggml_mul_mat_id_slots: the cached ones in place from the cache (get_experts), the others from the staging slot
    // (get_staged_experts) or, beyond its places, from the host experts mapped to the device (get_inplace_host), by the staging map
    bool stage_in_place(int32_t il, int64_t n_tokens, int64_t n_expert_used) const;
    bool stage_in_place(const ggml_tensor * w, int64_t n_tokens, int64_t n_expert_used) const;
    ggml_tensor * get_inplace_host(const ggml_tensor * w) const;

    // the share of the misses that the CPU computes while the device uploads the others, 0 if all misses are uploaded
    float cpu_share() const;

    // LLAMA_MOE_DEVICE_HITS: whether layer il looks up its slots on the device for n_tokens tokens
    bool device_hits(int32_t il, int64_t n_tokens, int64_t n_expert_used) const;

    // the device slot map of layer il: I32 [1, n_expert], the slot of a cached expert, -1 - e for the others
    ggml_tensor * get_device_map(int32_t il) const;

    // the device log of layer il: I32 [cap], the graph copies the selected experts to its start
    ggml_tensor * get_device_log(int32_t il) const;

    // the host experts of w as a device tensor (mapped pinned memory), for ggml_mul_mat_id_mapped; nullptr without device hits
    ggml_tensor * get_mapped_experts(const ggml_tensor * w, int64_t n_tokens, int64_t n_expert_used) const;

    // LLAMA_MOE_CPU_LANE: the channel (device address) for ggml_moe_doorbell/ggml_moe_join if a share of the misses of layer il
    // is computed by the CPU lane beside the device for n_tokens tokens, nullptr otherwise; registers the layer's host experts
    // (up, gate and down, or gate_up and down) and activation (swiglu, or swiglu_clamp if clamp > 0) with the lane
    void * cpu_lane(int32_t il, int64_t n_tokens, int64_t n_expert_used, const ggml_tensor * up, const ggml_tensor * gate,
            const ggml_tensor * down, const ggml_tensor * gate_up, float clamp);
    // the share of the misses the CPU lane computes; LLAMA_MOE_CPU_LANE_MULTI sets it for graphs of 2+ tokens
    float   cpu_lane_frac(int64_t n_tokens = 1) const;
    int32_t cpu_lane_max_tokens() const;
    int32_t cpu_lane_timeout_ms() const;

    // the experts of layer il computed on the CPU: I32 [1, n_expert] in host memory, e for these experts, -1 for the others
    ggml_tensor * get_cpu_map(int32_t il) const;

    // the CPU part of the layer of the slot lookup, with n_used experts per token, no inputs if the graph has no CPU part
    // its inputs on the device are read together with the selected experts, so the CPU part does not wait for the device
    // its output is copied to the device without waiting on the host
    void set_cpu_part(const ggml_tensor * lookup, int32_t n_used, std::vector<const ggml_tensor *> inputs, const ggml_tensor * output);

    // the router of the layer after il, if its experts are to be predicted from the input of layer il (LLAMA_MOE_PREFETCH), n_pred is the top k
    struct next_router {
        ggml_tensor * gate_inp    = nullptr;
        ggml_tensor * gate_inp_b  = nullptr;
        ggml_tensor * exp_probs_b = nullptr;
        int32_t       n_pred      = 0;
    };
    bool get_next_router(int32_t il, int64_t n_tokens, int64_t n_expert_used, next_router & r) const;

    // the predicted experts of the next layer, read back with the selected experts of the lookup, nullptr if none
    void set_prediction(const ggml_tensor * lookup, const ggml_tensor * pred);

    // called before each graph compute, prev_in_flight if the previous graph may still run (no synchronization since)
    void begin(bool prev_in_flight, ggml_cgraph * graph);

    // whether the copy callback may run on the scheduler's submit threads (one per device), else why not
    bool submit_threads_ok(std::string & why) const;

    // submit threads: once before a batch they queue (the threads skip begin() and launched())
    void submit_settle();

    // submit threads: the LLAMA_MOE_RING plan of the graph in pipeline copy `copy` (scheduler's thread, before it is queued)
    void ring_plan_copy(ggml_cgraph * graph, int copy);

    // submit threads: on the thread of backend, before (begin) and after it queues its part of the graph in pipeline copy `copy`
    void submit_graph(ggml_backend_t backend, int copy, bool begin);

    // called after the graph is queued
    void launched();

    // ggml_backend_sched copy callback, returns false if src is not a slot map, the staging map of a layer or an input of the CPU part
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

    // for large batches: copy the experts of w that are in the cache, starting at expert e and up to expert last, to the copy dst of w
    // returns the number of experts copied, 0 if expert e is not in the cache
    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last);

    // for large batches: the device address of expert e of w in the cache of backend, nullptr if it is not cached there
    // (counted as copied from the cache: the caller copies size bytes from it)
    const void * cached_expert(ggml_backend_t backend, const ggml_tensor * w, int64_t e, size_t size);

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_cache_ptr = std::unique_ptr<llama_moe_cache>;
