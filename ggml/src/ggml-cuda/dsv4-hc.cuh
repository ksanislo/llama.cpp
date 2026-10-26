#include "common.cuh"
#include "ggml.h"

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
// gate_scale_in/gate_scale_out: the SCALE nodes of a fused SCALE -> SIGMOID -> SCALE chain feeding src[2]
void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_tensor * gate_scale_in = nullptr, const ggml_tensor * gate_scale_out = nullptr);

// GGML_CUDA_DECODE_FUSION hcnorm: hc_post (optionally behind its SCALE -> SIGMOID -> SCALE gate) followed by the
// RMS_NORM -> MUL of the next hc_mix; writes both the hc_post output and the MUL output
bool ggml_cuda_dsv4_hc_post_rms_norm_supported(const ggml_tensor * hc_post, const ggml_tensor * rms_norm, const ggml_tensor * mul);
void ggml_cuda_op_dsv4_hc_post_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * hc_post,
        const ggml_tensor * gate_scale_in, const ggml_tensor * gate_scale_out, const ggml_tensor * rms_norm, ggml_tensor * mul,
        bool write_q8);
