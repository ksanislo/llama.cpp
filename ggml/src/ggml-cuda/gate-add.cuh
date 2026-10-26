#pragma once

#include "common.cuh"

// GGML_CUDA_DECODE_FUSION shgate: dst = a + b*sigmoid(g), the UNARY(SIGMOID) -> MUL -> ADD of a gated shared expert
// (g is one gate per row of b, or one per element)
bool ggml_cuda_sigmoid_mul_add_supported(const ggml_tensor * sigmoid, const ggml_tensor * mul, const ggml_tensor * add);
void ggml_cuda_op_sigmoid_mul_add(ggml_backend_cuda_context & ctx, const ggml_tensor * sigmoid, const ggml_tensor * mul, ggml_tensor * add);
