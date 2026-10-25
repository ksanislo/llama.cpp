#pragma once

#include "ggml.h"

#include <cstdint>
#include <memory>
#include <vector>

// the host side of the MoE CPU lane (ggml_moe_doorbell / ggml_moe_join): one thread polls the channels of the layers and
// computes the experts a device handed to the CPU with ggml-cpu's MUL_MAT_ID on a thread pool of its own, then writes their
// rows to the channel and sets its done word; the device meanwhile computes the other experts
// depends on ggml only (test-backend-ops builds it too)
struct llama_moe_lane_layer {
    uint8_t * chan       = nullptr; // host address of the channel
    int32_t   n_expert   = 0;
    int32_t   n_used     = 0;
    int32_t   max_tokens = 0;
    int64_t   n_embd     = 0;
    // host experts, readable by the CPU: up, gate and down, or gate_up (gate and up merged) and down
    const ggml_tensor * up      = nullptr;
    const ggml_tensor * gate    = nullptr;
    const ggml_tensor * down    = nullptr;
    const ggml_tensor * gate_up = nullptr;
    float clamp = 0.0f; // > 0: swiglu_clamp with this limit, else swiglu
};

class llama_moe_lane {
public:
    // n_slots channels, served by a pool of n_threads (the polling thread included), ggml threadpool polling level poll;
    // cpus not empty: every thread of the lane runs on these CPUs (one NUMA node's, for the devices of that node)
    llama_moe_lane(size_t n_slots, int n_threads, int poll, const std::vector<int> & cpus = {});
    ~llama_moe_lane();

    // sets slot i, once (later calls are ignored); the thread polls its channel from then on
    void set(size_t i, const llama_moe_lane_layer & l);
    bool is_set(size_t i) const;

    struct stats {
        uint64_t requests = 0; // doorbells with lane experts
        uint64_t experts  = 0; // distinct experts computed
        uint64_t rows     = 0; // (token, expert) rows computed
        double   busy_ms  = 0.0;
        uint64_t timeouts = 0; // joins that gave up waiting (error words of the channels)
    };
    stats get_stats() const;

    int n_threads() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
