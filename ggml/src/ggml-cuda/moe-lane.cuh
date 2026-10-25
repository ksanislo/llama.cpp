#include "common.cuh"

// MoE CPU lane (ggml_moe_doorbell, ggml_moe_join): the channel is pinned host memory mapped to the device

#define MOE_LANE_MAX_IDS 4096

void ggml_cuda_op_moe_doorbell(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_moe_join(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
