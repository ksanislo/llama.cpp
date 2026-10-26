#include "gate-add.cuh"

// separate roundings for the product and the sum (no FMA), as the unfused MUL and ADD kernels compute them
template <bool per_row>
static __global__ void sigmoid_mul_add_f32(const float * a, const float * b, const float * g, float * dst, const int64_t ncols,
        const int64_t n) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const float gv = per_row ? g[i / ncols] : g[i];
    const float s  = 1.0f / (1.0f + expf(-gv)); // op_sigmoid
    dst[i] = __fadd_rn(a[i], __fmul_rn(b[i], s));
}

bool ggml_cuda_sigmoid_mul_add_supported(const ggml_tensor * sigmoid, const ggml_tensor * mul, const ggml_tensor * add) {
    if (sigmoid->op != GGML_OP_UNARY || ggml_get_unary_op(sigmoid) != GGML_UNARY_OP_SIGMOID ||
            mul->op != GGML_OP_MUL || add->op != GGML_OP_ADD) {
        return false;
    }
    if (mul->src[0] != sigmoid && mul->src[1] != sigmoid) {
        return false;
    }
    if (add->src[0] != mul && add->src[1] != mul) {
        return false;
    }
    const ggml_tensor * g = sigmoid->src[0];
    const ggml_tensor * b = mul->src[0] == sigmoid ? mul->src[1] : mul->src[0];
    const ggml_tensor * a = add->src[0] == mul ? add->src[1] : add->src[0];
    for (const ggml_tensor * t : { g, sigmoid, b, mul, a, add }) {
        if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) {
            return false;
        }
    }
    // b and a are the full shape, the gate broadcasts along the rows only
    if (!ggml_are_same_shape(b, mul) || !ggml_are_same_shape(a, add) || !ggml_are_same_shape(mul, add)) {
        return false;
    }
    if (g->ne[0] != 1 && g->ne[0] != b->ne[0]) {
        return false;
    }
    for (int d = 1; d < 4; ++d) {
        if (g->ne[d] != b->ne[d]) {
            return false;
        }
    }
    return true;
}

void ggml_cuda_op_sigmoid_mul_add(ggml_backend_cuda_context & ctx, const ggml_tensor * sigmoid, const ggml_tensor * mul, ggml_tensor * add) {
    const ggml_tensor * g = sigmoid->src[0];
    const ggml_tensor * b = mul->src[0] == sigmoid ? mul->src[1] : mul->src[0];
    const ggml_tensor * a = add->src[0] == mul ? add->src[1] : add->src[0];

    const int64_t n     = ggml_nelements(add);
    const int64_t ncols = add->ne[0];
    const int     block = 256;
    const int64_t nblk  = (n + block - 1) / block;

    cudaStream_t stream = ctx.stream();
    if (g->ne[0] == 1) {
        sigmoid_mul_add_f32<true><<<nblk, block, 0, stream>>>((const float *) a->data, (const float *) b->data,
            (const float *) g->data, (float *) add->data, ncols, n);
    } else {
        sigmoid_mul_add_f32<false><<<nblk, block, 0, stream>>>((const float *) a->data, (const float *) b->data,
            (const float *) g->data, (float *) add->data, ncols, n);
    }
}
