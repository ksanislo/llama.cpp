#include "common.cuh"
#include "dsv4-hc.cuh"
#include "mmvq.cuh"


static constexpr int DSV4_HC = 4;


static __device__ void dsv4_hc_comb_norm_cols(float * comb, float eps) {
    for (int idst = 0; idst < DSV4_HC; ++idst) {
        float sum = eps;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static __device__ void dsv4_hc_comb_norm_rows(float * comb, float eps) {
    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float sum = eps;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static __global__ void dsv4_hc_comb_f32(
        const float * mixes,
        const float * scale,
        const float * base,
        float * dst,
        int64_t n_tokens,
        int64_t sm0,
        int64_t sm1,
        int64_t ss0,
        int64_t sb0,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        float eps,
        int32_t n_iter) {
    constexpr int comb_offset = 2*DSV4_HC;

    ggml_cuda_pdl_lc();
    const int64_t it = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;

    if (it >= n_tokens) {
        return;
    }

    ggml_cuda_pdl_sync();

    const float scale_comb = scale[2*ss0];
    float comb[DSV4_HC*DSV4_HC];

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float max = -INFINITY;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = mixes[(comb_offset + idx)*sm0 + it*sm1] * scale_comb + base[(comb_offset + idx)*sb0];
            comb[idx] = v;
            max = fmaxf(max, v);
        }

        float sum = 0.0f;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = expf(comb[idx] - max);
            comb[idx] = v;
            sum += v;
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            comb[idx] = comb[idx] * inv_sum + eps;
        }
    }

    dsv4_hc_comb_norm_cols(comb, eps);
    for (int32_t i = 1; i < n_iter; ++i) {
        dsv4_hc_comb_norm_rows(comb, eps);
        dsv4_hc_comb_norm_cols(comb, eps);
    }

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            dst[idst*sd0 + isrc*sd1 + it*sd2] = comb[idx];
        }
    }
}

template <bool gated>
static __global__ void dsv4_hc_pre_f32(
        const float * x,
        const float * weights,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sx2,
        int64_t sw0,
        int64_t sw1,
        int64_t sw2,
        int64_t sd0,
        int64_t sd1,
        float   scale) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0 = ir % n_embd;
    const int64_t it = ir / n_embd;

    float sum = 0.0f;
    for (int64_t ih = 0; ih < hc; ++ih) {
        const float xv = x[i0*sx0 + ih*sx1 + it*sx2];
        float wv;
        if constexpr (gated) {
            wv = 1.0f / (1.0f + expf(-weights[i0*sw0 + ih*sw1 + it*sw2]));
        } else {
            wv = weights[ih*sw0 + it*sw1];
        }
        sum += xv * wv;
    }

    dst[i0*sd0 + it*sd1] = scale * sum;
}

// one element of hc_post, shared with the norm fusion below so both compile to the same arithmetic
static __device__ __forceinline__ float dsv4_hc_post_gate(float p, float gate_scale_in, float gate_scale_out) {
    return gate_scale_out * (1.0f / (1.0f + expf(-(gate_scale_in * p))));
}

static __device__ __forceinline__ float dsv4_hc_post_value(float x, float p, float residual) {
    float sum = x * p;
    sum += residual;
    return sum;
}

// gated: post = gate_scale_out*sigmoid(gate_scale_in*post), the fused SCALE -> SIGMOID -> SCALE chain of qwen4exp
template <bool has_comb, bool gated>
static __global__ void dsv4_hc_post_f32(
        const float * x,
        const float * residual,
        const float * post,
        const float * comb,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sr0,
        int64_t sr1,
        int64_t sr2,
        int64_t sp0,
        int64_t sp1,
        int64_t sc0,
        int64_t sc1,
        int64_t sc2,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        float gate_scale_in,
        float gate_scale_out) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * hc * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0   = ir % n_embd;
    const int64_t idst = (ir / n_embd) % hc;
    const int64_t it   = ir / (n_embd * hc);

    float p = post[idst*sp0 + it*sp1];
    if constexpr (gated) {
        p = dsv4_hc_post_gate(p, gate_scale_in, gate_scale_out);
    }

    float sum;
    if constexpr (has_comb) {
        sum = x[i0*sx0 + it*sx1] * p;
        for (int64_t isrc = 0; isrc < hc; ++isrc) {
            sum += residual[i0*sr0 + isrc*sr1 + it*sr2] * comb[idst*sc0 + isrc*sc1 + it*sc2];
        }
    } else {
        sum = dsv4_hc_post_value(x[i0*sx0 + it*sx1], p, residual[i0*sr0 + idst*sr1 + it*sr2]);
    }

    dst[i0*sd0 + idst*sd1 + it*sd2] = sum;
}

// hc_post (identity comb, F32, rows contiguous) and the RMS_NORM -> MUL of the next hc_mix on its output, one block
// per (stream, token) row. The norm half is the vectorized path of rms_norm_f32 (same block size, loads and
// reduction), so the norm equals the unfused one; the hc_post values are written as well (the next residual).
// write_q8: also the q8_1 rows of the output (rows of hc*ncols per token, as quantize_q8_1 makes them for an MMVQ that
// reads the output; the block sums follow quantize_q8_1's warp_reduce_sum order, see below)
template <int block_size, int vpt, bool gated, bool write_q8>
static __global__ void dsv4_hc_post_rms_norm_f32(
        const float * x, const float * residual, const float * post, float * hc_dst, float * dst, const float * mul,
        const int ncols, const int hc,
        const int64_t sx1, const int64_t sr1, const int64_t sr2, const int64_t sp0, const int64_t sp1,
        const int64_t shd1, const int64_t shd2,
        const int64_t sm1, const int64_t sm2, const int mul_nrows, const int mul_nchannels,
        const float eps, const float gate_scale_in, const float gate_scale_out,
        block_q8_1 * q8, const int64_t q8_row_blocks) {
    ggml_cuda_pdl_lc();
    const int row     = blockIdx.x; // stream
    const int channel = blockIdx.y; // token
    const int tid     = threadIdx.x;

    extern __shared__ float s_sum[];

    ggml_cuda_pdl_sync();

    float p = post[row*sp0 + channel*sp1];
    if constexpr (gated) {
        p = dsv4_hc_post_gate(p, gate_scale_in, gate_scale_out);
    }

    const float4 * x4  = (const float4 *) (x        + channel*sx1);
    const float4 * r4  = (const float4 *) (residual + row*sr1 + channel*sr2);
    float4       * h4  = (float4       *) (hc_dst   + row*shd1 + channel*shd2);
    const float4 * m4  = (const float4 *) (mul      + (row % mul_nrows)*sm1 + (channel % mul_nchannels)*sm2);
    float4       * d4  = (float4       *) (dst      + ((int64_t) channel*hc + row)*ncols);

    const int ncols4 = ncols / 4;

    float4 xs[vpt];
    float  tmp = 0.0f;

#pragma unroll
    for (int k = 0; k < vpt; k++) {
        const int i = tid + k*block_size;
        if (i < ncols4) {
            const float4 xv = x4[i];
            const float4 rv = r4[i];
            xs[k] = make_float4(dsv4_hc_post_value(xv.x, p, rv.x), dsv4_hc_post_value(xv.y, p, rv.y),
                                dsv4_hc_post_value(xv.z, p, rv.z), dsv4_hc_post_value(xv.w, p, rv.w));
            h4[i] = xs[k];
            tmp += xs[k].x*xs[k].x + xs[k].y*xs[k].y + xs[k].z*xs[k].z + xs[k].w*xs[k].w;
        }
    }
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
    const float scale = rsqrtf(tmp / ncols + eps);

#pragma unroll
    for (int k = 0; k < vpt; k++) {
        const int i = tid + k*block_size;
        float4 r = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (i < ncols4) {
            r = make_float4(scale*xs[k].x, scale*xs[k].y, scale*xs[k].z, scale*xs[k].w);
            const float4 m = m4[i];
            r.x *= m.x;
            r.y *= m.y;
            r.z *= m.z;
            r.w *= m.w;
            d4[i] = r;
        }
        if constexpr (write_q8) {
            // a q8_1 block of 32 values = 8 lanes of 4. quantize_q8_1 has one value per lane and reduces with xor
            // offsets 16, 8, 4, 2, 1: value j = 4*lane + c, so 16, 8, 4 are lane offsets 4, 2, 1 and 2, 1 are components
            float amax = fmaxf(fmaxf(fabsf(r.x), fabsf(r.y)), fmaxf(fabsf(r.z), fabsf(r.w)));
            float4 sv = r;
#pragma unroll
            for (int offset = 4; offset > 0; offset >>= 1) {
                amax  = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset, WARP_SIZE));
                sv.x += __shfl_xor_sync(0xffffffff, sv.x, offset, WARP_SIZE);
                sv.y += __shfl_xor_sync(0xffffffff, sv.y, offset, WARP_SIZE);
                sv.z += __shfl_xor_sync(0xffffffff, sv.z, offset, WARP_SIZE);
                sv.w += __shfl_xor_sync(0xffffffff, sv.w, offset, WARP_SIZE);
            }
            sv.x += sv.z;
            sv.y += sv.w;
            const float sum = sv.x + sv.y;
            if (i < ncols4) {
                const float d = amax / 127.0f;
                const int64_t e  = (int64_t) row*ncols + 4*i; // element in the token's q8 row
                block_q8_1  * yb = q8 + channel*q8_row_blocks + e/QK8_1;
                char4 q;
                q.x = amax == 0.0f ? 0 : roundf(r.x / d);
                q.y = amax == 0.0f ? 0 : roundf(r.y / d);
                q.z = amax == 0.0f ? 0 : roundf(r.z / d);
                q.w = amax == 0.0f ? 0 : roundf(r.w / d);
                *(char4 *) (yb->qs + e % QK8_1) = q;
                if (e % QK8_1 == 0) {
                    yb->ds = make_half2(d, sum);
                }
            }
        }
    }
}

bool ggml_cuda_dsv4_hc_post_rms_norm_supported(const ggml_tensor * hc_post, const ggml_tensor * rms_norm, const ggml_tensor * mul) {
    const ggml_tensor * x        = hc_post->src[0];
    const ggml_tensor * residual = hc_post->src[1];
    const ggml_tensor * w        = mul->src[0] == rms_norm ? mul->src[1] : mul->src[0];

    if (hc_post->src[3] != nullptr || rms_norm->src[0] != hc_post || (mul->src[0] != rms_norm && mul->src[1] != rms_norm)) {
        return false;
    }
    for (const ggml_tensor * t : { x, residual, hc_post, rms_norm, mul, w }) {
        if (t->type != GGML_TYPE_F32 || t->nb[0] != sizeof(float) || t->ne[3] != 1 || ((uintptr_t) t->data & 15) != 0) {
            return false;
        }
    }
    const int64_t n_embd = x->ne[0];
    const int64_t hc     = residual->ne[1];
    const int64_t nt     = x->ne[1];
    if (n_embd % 4 != 0 || n_embd/4 > 4096 || x->ne[2] != 1 || nt > 65535 || hc > 65535 ||
            !ggml_are_same_shape(hc_post, residual) || !ggml_are_same_shape(mul, hc_post) ||
            hc_post->ne[0] != n_embd || hc_post->ne[1] != hc || hc_post->ne[2] != nt) {
        return false;
    }
    // rows stay 16-byte aligned, the norm output is written contiguous like rms_norm_f32's
    for (const ggml_tensor * t : { x, residual, hc_post, w }) {
        for (int d = 1; d < 4; ++d) {
            if (t->nb[d] % 16 != 0) {
                return false;
            }
        }
    }
    if (!ggml_is_contiguous(mul) || !ggml_is_contiguous(rms_norm)) {
        return false;
    }
    // w broadcasts like the MUL: [n_embd, hc or 1, nt or 1]
    if (w->ne[0] != n_embd || (w->ne[1] != hc && w->ne[1] != 1) || (w->ne[2] != nt && w->ne[2] != 1)) {
        return false;
    }
    return true;
}

void ggml_cuda_op_dsv4_hc_post_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * hc_post,
        const ggml_tensor * gate_scale_in, const ggml_tensor * gate_scale_out, const ggml_tensor * rms_norm, ggml_tensor * mul,
        bool write_q8) {
    const ggml_tensor * x        = hc_post->src[0];
    const ggml_tensor * residual = hc_post->src[1];
    const ggml_tensor * w        = mul->src[0] == rms_norm ? mul->src[1] : mul->src[0];

    const bool          gated = gate_scale_in != nullptr;
    const ggml_tensor * post  = gated ? gate_scale_in->src[0] : hc_post->src[2];
    const float gate_in  = gated ? ggml_get_op_params_f32(gate_scale_in,  0) : 1.0f;
    const float gate_out = gated ? ggml_get_op_params_f32(gate_scale_out, 0) : 1.0f;
    GGML_ASSERT(post->type == GGML_TYPE_F32);

    const int ncols = (int) x->ne[0];
    const int hc    = (int) residual->ne[1];
    const int nt    = (int) x->ne[1];

    float eps;
    memcpy(&eps, rms_norm->op_params, sizeof(float));

    const int64_t f = sizeof(float);
    const dim3 blocks_num(hc, nt, 1);
    cudaStream_t stream = ctx.stream();

    // q8: the output as [ncols*hc, nt] rows goes into the q8_1 cache for the MMVQ that reads it (hc_down)
    block_q8_1 * q8 = nullptr;
    const int64_t row_len = (int64_t) ncols*hc;
    if (write_q8 && ctx.q8_active && ncols % QK8_1 == 0 && row_len % MATRIX_ROW_PADDING == 0 && ggml_is_contiguous(mul)) {
        ggml_tensor rows = *mul;
        rows.ne[0] = row_len;
        rows.ne[1] = nt;
        rows.ne[2] = rows.ne[3] = 1;
        rows.nb[1] = row_len*f;
        rows.nb[2] = rows.nb[3] = rows.nb[1]*nt;
        bool hit = false;
        q8 = (block_q8_1 *) ggml_cuda_q8_cache_get(ctx, &rows, row_len, hit);
    }
    const int64_t q8_row_blocks = row_len / QK8_1;

#define HC_POST_RMS_LAUNCH(BLOCK_SIZE, VPT) \
    do { \
        const dim3 block_dims(BLOCK_SIZE, 1, 1); \
        const ggml_cuda_kernel_launch_params launch_params{ blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float) : 0, stream }; \
        auto kernel = q8 ? (gated ? dsv4_hc_post_rms_norm_f32<BLOCK_SIZE, VPT, true,  true> : dsv4_hc_post_rms_norm_f32<BLOCK_SIZE, VPT, false, true>) \
                         : (gated ? dsv4_hc_post_rms_norm_f32<BLOCK_SIZE, VPT, true, false> : dsv4_hc_post_rms_norm_f32<BLOCK_SIZE, VPT, false, false>); \
        ggml_cuda_kernel_launch(kernel, launch_params, \
            (const float *) x->data, (const float *) residual->data, (const float *) post->data, (float *) hc_post->data, \
            (float *) mul->data, (const float *) w->data, ncols, hc, \
            (int64_t) (x->nb[1]/f), (int64_t) (residual->nb[1]/f), (int64_t) (residual->nb[2]/f), \
            (int64_t) (post->nb[0]/f), (int64_t) (post->nb[1]/f), \
            (int64_t) (hc_post->nb[1]/f), (int64_t) (hc_post->nb[2]/f), \
            (int64_t) (w->nb[1]/f), (int64_t) (w->nb[2]/f), (int) w->ne[1], (int) w->ne[2], \
            eps, gate_in, gate_out, q8, q8_row_blocks); \
    } while (0)

    // the block size of rms_norm_f32_dispatch's vectorized path for this ncols
    const int n4 = ncols / 4;
    if (n4 <= 32) {
        HC_POST_RMS_LAUNCH(32, 1);
    } else if (n4 <= 128) {
        HC_POST_RMS_LAUNCH(128, 1);
    } else if (n4 <= 256) {
        HC_POST_RMS_LAUNCH(256, 1);
    } else if (n4 <= 512) {
        HC_POST_RMS_LAUNCH(512, 1);
    } else if (n4 <= 1024) {
        HC_POST_RMS_LAUNCH(1024, 1);
    } else {
        HC_POST_RMS_LAUNCH(1024, 4);
    }
#undef HC_POST_RMS_LAUNCH
}

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int64_t hc_mix_dim = (2 + DSV4_HC)*DSV4_HC;

    GGML_ASSERT(mixes->ne[0] == hc_mix_dim);
    GGML_ASSERT(dst->ne[0] == DSV4_HC);
    GGML_ASSERT(dst->ne[1] == DSV4_HC);
    GGML_ASSERT(dst->ne[2] == mixes->ne[1]);
    GGML_ASSERT(scale->ne[0] >= 3);
    GGML_ASSERT(base->ne[0] == hc_mix_dim);

    GGML_TENSOR_LOCALS(size_t, nbm, mixes, nb);
    GGML_TENSOR_LOCALS(size_t, nbs, scale, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, base,  nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,   nb);

    const int64_t n_tokens = mixes->ne[1];
    const float eps = ggml_get_op_params_f32(dst, 0);
    const int32_t n_iter = ggml_get_op_params_i32(dst, 1);

    const int block_size = 256;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((n_tokens + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(dsv4_hc_comb_f32, launch_params,
            (const float *) mixes->data, (const float *) scale->data, (const float *) base->data, (float *) dst->data,
            n_tokens,
            nbm0 / sizeof(float), nbm1 / sizeof(float),
            nbs0 / sizeof(float),
            nbb0 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float),
            eps, n_iter);
}

void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x       = dst->src[0];
    const ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    const int64_t n_embd   = x->ne[0];
    const int64_t hc       = x->ne[1];
    const int64_t n_tokens = x->ne[2];

    const float scale = ggml_get_op_params_f32(dst, 0);
    const bool  gated = ggml_get_op_params_i32(dst, 1) != 0;

    const int block_size = 256;
    const int64_t nr = n_embd * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    auto kernel = gated ? dsv4_hc_pre_f32<true> : dsv4_hc_pre_f32<false>;
    ggml_cuda_kernel_launch(kernel, launch_params,
            (const float *) x->data, (const float *) weights->data, (float *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float), nbx2 / sizeof(float),
            nbw0 / sizeof(float), nbw1 / sizeof(float), nbw2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float),
            scale);
}

void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_tensor * gate_scale_in, const ggml_tensor * gate_scale_out) {
    const ggml_tensor * x        = dst->src[0];
    const ggml_tensor * residual = dst->src[1];
    const ggml_tensor * comb     = dst->src[3];

    // when fused, read the raw post weights from before the SCALE -> SIGMOID -> SCALE chain
    const bool          gated = gate_scale_in != nullptr;
    const ggml_tensor * post  = gated ? gate_scale_in->src[0] : dst->src[2];

    const float gate_in  = gated ? ggml_get_op_params_f32(gate_scale_in,  0) : 1.0f;
    const float gate_out = gated ? ggml_get_op_params_f32(gate_scale_out, 0) : 1.0f;

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);
    GGML_ASSERT(comb == nullptr || comb->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,        nb);
    GGML_TENSOR_LOCALS(size_t, nbr, residual, nb);
    GGML_TENSOR_LOCALS(size_t, nbp, post,     nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,      nb);

    const size_t nbc0 = comb ? comb->nb[0] : 0;
    const size_t nbc1 = comb ? comb->nb[1] : 0;
    const size_t nbc2 = comb ? comb->nb[2] : 0;

    const int64_t n_embd   = x->ne[0];
    const int64_t n_tokens = x->ne[1];
    const int64_t hc       = residual->ne[1];

    const int block_size = 256;
    const int64_t nr = n_embd * hc * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    auto kernel = comb ? (gated ? dsv4_hc_post_f32<true,  true> : dsv4_hc_post_f32<true,  false>)
                       : (gated ? dsv4_hc_post_f32<false, true> : dsv4_hc_post_f32<false, false>);
    ggml_cuda_kernel_launch(kernel, launch_params,
            (const float *) x->data, (const float *) residual->data,
            (const float *) post->data, comb ? (const float *) comb->data : nullptr, (float *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float),
            nbr0 / sizeof(float), nbr1 / sizeof(float), nbr2 / sizeof(float),
            nbp0 / sizeof(float), nbp1 / sizeof(float),
            nbc0 / sizeof(float), nbc1 / sizeof(float), nbc2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float),
            gate_in, gate_out);
}
