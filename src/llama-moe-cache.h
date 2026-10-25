#pragma once

#include "ggml-backend.h"

#include <map>
#include <memory>
#include <vector>

struct llama_model;

// keeps recently and repeatedly used experts of host-resident MoE layers in a device buffer
// each layer has a slot map in host memory: when the scheduler copies it to the device, the copy callback uploads the missing experts
// with LLAMA_MOE_ASYNC=n, the batches the cache does not serve read the experts of a layer from one of n staging slots per device instead,
// filled on a separate stream of the device; large batches stage all the experts of the next layer while the current one computes
// with LLAMA_MOE_ASYNC_BORROW=1 the staging slots lie in cache slots of the device, which large batches borrow and the next cached batch gives back
// with LLAMA_MOE_DEVICE_HITS=1|2 small batches look the slots up on the device and read the misses straight from the pinned host experts,
// so their graph has no host step; the experts they select are logged on the device and the next graph fills the cache from the log
// with LLAMA_MOE_CPU_LANE=f in addition, a share f of their misses is computed by a CPU thread pool while the device computes the rest
class llama_moe_cache {
public:
    // backends are all the backends of the context, each GPU gets its own cache of the given size for the layers assigned to it
    // cpu_share is the share of the misses computed on the CPU instead of uploaded
    llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, float cpu_share);
    ~llama_moe_cache();

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
    float   cpu_lane_frac() const;
    int32_t cpu_lane_max_tokens() const;
    int32_t cpu_lane_timeout_ms() const;

    // the experts of layer il computed on the CPU: I32 [1, n_expert] in host memory, e for these experts, -1 for the others
    ggml_tensor * get_cpu_map(int32_t il) const;

    // the CPU part of the layer of the slot lookup, with n_used experts per token, no inputs if the graph has no CPU part
    // its inputs on the device are read together with the selected experts, so the CPU part does not wait for the device
    // its output is copied to the device without waiting on the host
    void set_cpu_part(const ggml_tensor * lookup, int32_t n_used, std::vector<const ggml_tensor *> inputs, const ggml_tensor * output);

    // called before each graph compute, prev_in_flight if the previous graph may still run (no synchronization since)
    void begin(bool prev_in_flight, ggml_cgraph * graph);

    // called after the graph is queued
    void launched();

    // ggml_backend_sched copy callback, returns false if src is not a slot map, the staging map of a layer or an input of the CPU part
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

    // for large batches: copy the experts of w that are in the cache, starting at expert e and up to expert last, to the copy dst of w
    // returns the number of experts copied, 0 if expert e is not in the cache
    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last);

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_cache_ptr = std::unique_ptr<llama_moe_cache>;
