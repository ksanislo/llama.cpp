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
class llama_moe_cache {
public:
    // backends are all the backends of the context, each GPU gets its own cache of the given size for the layers assigned to it
    llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size);
    ~llama_moe_cache();

    // the device that caches layer il
    ggml_backend_t backend(int32_t il) const;

    // the slot map of layer il, if its experts can be read from the cache or the staging slots for n_tokens tokens, nullptr otherwise
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const;

    // the experts of w in the cache, nullptr if w is not cached
    ggml_tensor * get_experts(const ggml_tensor * w) const;

    // the experts of w to read with the slots of get_slot_map: in the cache or in a staging slot
    ggml_tensor * get_experts(const ggml_tensor * w, int64_t n_tokens, int64_t n_expert_used) const;

    // the experts of w in its staging slot, nullptr if w is not staged
    ggml_tensor * get_staged_experts(const ggml_tensor * w) const;

    // ggml_backend_sched copy callback, returns false if src is not a slot map (or the staging map of a layer)
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
