#include "moe-lane.cuh"

// the channel words are written by the other side over PCIe: volatile accesses, system-scope fences

static __device__ __forceinline__ uint32_t moe_lane_get(const uint8_t * chan, int word) {
    return ((const volatile uint32_t *) chan)[word];
}

static __device__ __forceinline__ void moe_lane_set(uint8_t * chan, int word, uint32_t v) {
    ((volatile uint32_t *) chan)[word] = v;
}

static __device__ __forceinline__ uint64_t moe_lane_time_ns() {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    return clock64(); // ~1 GHz, the timeout is a safety net only
#else
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
#endif
}

static __device__ __forceinline__ void moe_lane_backoff() {
#ifdef GGML_USE_HIP
    __builtin_amdgcn_s_sleep(4);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    __nanosleep(64);
#endif
}

// one block: plan the lane (the first n_lane distinct misses in the order of ids), publish ids and x, ring
static __global__ void moe_doorbell_kernel(
        const int32_t * __restrict__ ids, const float * __restrict__ x, int32_t * __restrict__ out, uint8_t * chan,
        const int n, const int64_t n_x, const int n_tokens, const int n_expert, const float frac,
        const size_t off_ids, const size_t off_x) {
    extern __shared__ int32_t s[];
    int32_t * s_ids  = s;         // [n]
    int32_t * s_rank = s + n;     // [n] rank of a first-seen miss, -1 otherwise
    __shared__ int32_t s_lane;

    const int tid = threadIdx.x;
    for (int i = tid; i < n; i += blockDim.x) {
        s_ids[i] = ids[i];
    }
    __syncthreads();

    for (int i = tid; i < n; i += blockDim.x) {
        const int32_t v = s_ids[i];
        bool first = v < 0 && v >= -n_expert;
        for (int j = 0; j < i && first; ++j) {
            first = s_ids[j] != v;
        }
        s_rank[i] = first ? 1 : -1;
    }
    __syncthreads();

    if (tid == 0) {
        const uint32_t ov = moe_lane_get(chan, GGML_MOE_LANE_FRAC);
        const float f = ov != 0 ? (ov - 1)*0.001f : frac;
        int32_t r = 0;
        for (int i = 0; i < n; ++i) {
            if (s_rank[i] >= 0) {
                s_rank[i] = r++;
            }
        }
        s_lane = min(r, (int32_t) floorf(r*f + 0.5f));
    }
    __syncthreads();

    const int32_t n_lane = s_lane;
    int32_t * chan_ids = (int32_t *) (chan + off_ids);
    for (int i = tid; i < n; i += blockDim.x) {
        const int32_t v = s_ids[i];
        int32_t o = v;
        if (v < 0 && v >= -n_expert) {
            int j = 0;
            while (s_ids[j] != v) {
                ++j;
            }
            if (s_rank[j] < n_lane) {
                o = -1 - n_expert - (-1 - v);
            }
        }
        out[i]      = o;
        chan_ids[i] = o;
    }
    if (n_lane > 0) {
        float * chan_x = (float *) (chan + off_x);
        for (int64_t i = tid; i < n_x; i += blockDim.x) {
            chan_x[i] = x[i];
        }
    }
    __threadfence_system();
    __syncthreads();

    if (tid == 0) {
        const uint32_t seq = moe_lane_get(chan, GGML_MOE_LANE_SEQ) + 1;
        moe_lane_set(chan, GGML_MOE_LANE_NTOK,  (uint32_t) n_tokens);
        moe_lane_set(chan, GGML_MOE_LANE_NLANE, (uint32_t) n_lane);
        if (n_lane == 0) {
            moe_lane_set(chan, GGML_MOE_LANE_DONE, seq);
        }
        __threadfence_system();
        moe_lane_set(chan, GGML_MOE_LANE_SEQ, seq);
        __threadfence_system();
    }
}

// one thread: if the ids have lane experts, wait until the host completed the request (or the timeout)
static __global__ void moe_join_wait_kernel(const int32_t * __restrict__ ids, uint8_t * chan, const int n, const int n_expert, const uint64_t timeout_ns) {
    bool lane = false;
    for (int i = 0; i < n && !lane; ++i) {
        lane = ids[i] < -n_expert;
    }
    if (!lane) {
        return;
    }
    const uint32_t seq = moe_lane_get(chan, GGML_MOE_LANE_SEQ);
    const uint64_t t0  = moe_lane_time_ns();
    while (moe_lane_get(chan, GGML_MOE_LANE_DONE) != seq) {
        if (moe_lane_time_ns() - t0 > timeout_ns) {
            moe_lane_set(chan, GGML_MOE_LANE_ERR, moe_lane_get(chan, GGML_MOE_LANE_ERR) + 1);
            break;
        }
        moe_lane_backoff();
    }
    __threadfence_system();
}

// one block per row: the row of a, or of the channel for a lane expert (zero if its request timed out)
static __global__ void moe_join_copy_kernel(
        const float * __restrict__ a, const int32_t * __restrict__ ids, uint8_t * chan, float * __restrict__ dst,
        const int64_t n_embd, const int n_expert, const size_t off_rows) {
    const int i = blockIdx.x;
    const bool lane = ids[i] < -n_expert;
    __shared__ bool s_ok;
    if (lane && threadIdx.x == 0) {
        s_ok = moe_lane_get(chan, GGML_MOE_LANE_DONE) == moe_lane_get(chan, GGML_MOE_LANE_SEQ);
    }
    __syncthreads();

    float * d = dst + i*n_embd;
    if (!lane) {
        const float * src = a + i*n_embd;
        for (int64_t k = threadIdx.x; k < n_embd; k += blockDim.x) {
            d[k] = src[k];
        }
    } else if (s_ok) {
        const volatile float * src = (const volatile float *) (chan + off_rows) + i*n_embd;
        for (int64_t k = threadIdx.x; k < n_embd; k += blockDim.x) {
            d[k] = src[k];
        }
    } else {
        for (int64_t k = threadIdx.x; k < n_embd; k += blockDim.x) {
            d[k] = 0.0f;
        }
    }
}

void ggml_cuda_op_moe_doorbell(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * ids = dst->src[0];
    const ggml_tensor * x   = dst->src[1];

    uint8_t * chan;
    memcpy(&chan, dst->op_params, sizeof(chan));
    const int32_t n_expert   = ggml_get_op_params_i32(dst, 2);
    const int32_t max_tokens = ggml_get_op_params_i32(dst, 3);
    const float   frac       = ggml_get_op_params_f32(dst, 4);

    size_t off_ids, off_x;
    ggml_moe_lane_layout(x->ne[0], ids->ne[0], max_tokens, &off_ids, &off_x, nullptr);

    const int n = (int) ggml_nelements(ids);
    GGML_ASSERT(n <= MOE_LANE_MAX_IDS);

    const int nth = std::min(256, std::max(32, GGML_PAD(n, 32)));
    moe_doorbell_kernel<<<1, nth, 2*n*sizeof(int32_t), ctx.stream()>>>(
        (const int32_t *) ids->data, (const float *) x->data, (int32_t *) dst->data, chan,
        n, ggml_nelements(x), (int) ids->ne[1], n_expert, frac, off_ids, off_x);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_op_moe_join(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * a   = dst->src[0];
    const ggml_tensor * ids = dst->src[1];

    uint8_t * chan;
    memcpy(&chan, dst->op_params, sizeof(chan));
    const int32_t n_expert   = ggml_get_op_params_i32(dst, 2);
    const int32_t max_tokens = ggml_get_op_params_i32(dst, 3);
    const int32_t timeout_ms = ggml_get_op_params_i32(dst, 4);

    size_t off_rows;
    ggml_moe_lane_layout(a->ne[0], ids->ne[0], max_tokens, nullptr, nullptr, &off_rows);

    const int n = (int) ggml_nelements(ids);
    cudaStream_t stream = ctx.stream();
    moe_join_wait_kernel<<<1, 1, 0, stream>>>((const int32_t *) ids->data, chan, n, n_expert, 1000000ull*(uint64_t) timeout_ms);
    CUDA_CHECK(cudaGetLastError());
    moe_join_copy_kernel<<<n, 256, 0, stream>>>((const float *) a->data, (const int32_t *) ids->data, chan, (float *) dst->data,
        a->ne[0], n_expert, off_rows);
    CUDA_CHECK(cudaGetLastError());
}
