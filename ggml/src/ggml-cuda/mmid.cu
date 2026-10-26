#include "common.cuh"
#include "mmid.cuh"

// To reduce shared memory use, store "it" and "iex_used" with 22/10 bits each.
struct mm_ids_helper_store {
    uint32_t data;

    __device__ mm_ids_helper_store(const uint32_t it, const uint32_t iex_used) {
        data = (it & 0x003FFFFF) | (iex_used << 22);
    }

    __device__ uint32_t it() const {
        return data & 0x003FFFFF;
    }

    __device__ uint32_t iex_used() const {
        return data >> 22;
    }
};
static_assert(sizeof(mm_ids_helper_store) == 4, "unexpected size for mm_ids_helper_store");

// the generic path passes 0, which needs no padding since it never groups lanes by token
template <int n> struct mm_ids_pow2 { static constexpr int value = 2*mm_ids_pow2<(n + 1)/2>::value; };
template <>      struct mm_ids_pow2<1> { static constexpr int value = 1; };
template <>      struct mm_ids_pow2<0> { static constexpr int value = 1; };

// Helper function for mul_mat_id, converts ids to a more convenient format.
// ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert.
// ids_dst describes the same mapping but for the dst tensor.
// The upper and lower bounds for the ith expert in the compact src1 tensor are stored in expert_bounds[i:i+1].
template <int n_expert_used_template>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    const int expert = blockIdx.x;

    extern __shared__ char data_mm_ids_helper[];
    mm_ids_helper_store * store = (mm_ids_helper_store *) data_mm_ids_helper;

    int nex_prev   = 0; // Number of columns for experts with a lower index.
    int it_compact = 0; // Running index for the compact slice of this expert.

    if constexpr (n_expert_used_template == 0) {
        // Generic implementation:
        for (int it = 0; it < n_tokens; ++it) {
            int iex_used = -1; // The index at which the expert is used, if any.
            for (int iex = threadIdx.x; iex < n_expert_used; iex += warp_size) {
                const int expert_used = ids[it*si1 + iex];
                nex_prev += expert_used < expert;
                if (expert_used == expert) {
                    iex_used = iex;
                }
            }

            if (iex_used != -1) {
                store[it_compact] = mm_ids_helper_store(it, iex_used);
            }

            if (warp_reduce_any<warp_size>(iex_used != -1)) {
                it_compact++;
            }
        }
    } else {
        // Implementation optimized for specific numbers of experts used:
        // a warp holds a whole number of token slots, so the slot count is padded to a power of 2
        constexpr int neu_padded = mm_ids_pow2<n_expert_used_template>::value;
        static_assert(neu_padded <= warp_size && warp_size % neu_padded == 0, "bad n_expert_used");
        for (int it0 = 0; it0 < n_tokens; it0 += warp_size/neu_padded) {
            const int it = it0 + threadIdx.x / neu_padded;

            const int iex = threadIdx.x % neu_padded; // The index at which the expert is used, if any.
            const int expert_used = (neu_padded == n_expert_used || iex < n_expert_used) && it < n_tokens ?
                ids[it*si1 + iex] : INT_MAX;
            const int iex_used = expert_used == expert ? iex : -1;
            nex_prev += expert_used < expert;

            // Whether the threads at this token position have used the expert:
            const int it_compact_add_self = warp_reduce_any<neu_padded>(iex_used != -1);

            // Do a scan over threads at lower token positions in warp to get the correct index for writing data:
            int it_compact_add_lower = 0;
#pragma unroll
            for (int offset = neu_padded; offset < warp_size; offset += neu_padded) {
                const int tmp = __shfl_up_sync(0xFFFFFFFF, it_compact_add_self, offset, warp_size);
                if (threadIdx.x >= static_cast<unsigned int>(offset)) {
                    it_compact_add_lower += tmp;
                }
            }

            if (iex_used != -1) {
                store[it_compact + it_compact_add_lower] = mm_ids_helper_store(it, iex_used);
            }

            // The thread with the highest index in the warp always has the sum over the whole warp, use it to increment all threads:
            it_compact += __shfl_sync(0xFFFFFFFF, it_compact_add_lower + it_compact_add_self, warp_size - 1, warp_size);
        }
    }
    nex_prev = warp_reduce_sum<warp_size>(nex_prev);
    ggml_cuda_syncwarp();

    for (int itc = threadIdx.x; itc < it_compact; itc += warp_size) {
        const mm_ids_helper_store store_it = store[itc];
        const int it       = store_it.it();
        const int iex_used = store_it.iex_used();
        ids_dst[nex_prev + itc] = it*n_expert_used + iex_used;
        // ids_src1 holds the forward map, or the inverse map (token slot -> compact row) for quant dedup
        if (write_inverse) {
            ids_src1[it*n_expert_used + iex_used] = nex_prev + itc;
        } else {
            ids_src1[nex_prev + itc] = it*sis1 + iex_used % nchannels_y;
        }
    }

    if (threadIdx.x != 0) {
        return;
    }

    expert_bounds[expert] = nex_prev;

    if (expert < static_cast<int>(gridDim.x) - 1) {
        return;
    }

    expert_bounds[gridDim.x] = nex_prev + it_compact;
}

// GGML_CUDA_MMQ_MOE=ids: the same maps with nwarps warps per expert, each scanning a contiguous token range into its own part
// of the store (a token uses an expert at most once, so a range never holds more entries than tokens), then written in token
// order at the prefix of the warps before it - identical output, the token scan nwarps times shorter
template <int n_expert_used_template, int nwarps>
__launch_bounds__(nwarps*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mm_ids_helper_par(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_tokens, const int nchannels_y, const int si1, const int sis1, const bool write_inverse) {
    constexpr int warp_size     = ggml_cuda_get_physical_warp_size();
    constexpr int n_expert_used = n_expert_used_template;
    constexpr int neu_padded    = mm_ids_pow2<n_expert_used_template>::value;
    constexpr int tpi           = warp_size/neu_padded; // tokens per warp iteration
    static_assert(neu_padded <= warp_size && warp_size % neu_padded == 0, "bad n_expert_used");
    const int expert = blockIdx.x;
    const int warp   = threadIdx.y;
    const int lane   = threadIdx.x;

    // all dynamic: static shared memory would lower the dynamic limit below smpbo (cudaFuncSetAttribute: invalid argument)
    extern __shared__ char data_mm_ids_helper[];
    int * warp_count = (int *) data_mm_ids_helper;
    int * warp_prev  = warp_count + nwarps;
    mm_ids_helper_store * store = (mm_ids_helper_store *) (warp_prev + nwarps);

    const int chunk = ((n_tokens + nwarps - 1)/nwarps + tpi - 1)/tpi * tpi;
    const int it_lo = min(warp*chunk, n_tokens);
    const int it_hi = min(it_lo + chunk, n_tokens);

    int nex_prev   = 0;
    int it_compact = 0;
    for (int it0 = it_lo; it0 < it_hi; it0 += tpi) {
        const int it  = it0 + lane / neu_padded;
        const int iex = lane % neu_padded;
        const int expert_used = (neu_padded == n_expert_used || iex < n_expert_used) && it < it_hi ? ids[it*si1 + iex] : INT_MAX;
        const int iex_used = expert_used == expert ? iex : -1;
        nex_prev += expert_used < expert;

        const int it_compact_add_self = warp_reduce_any<neu_padded>(iex_used != -1);
        int it_compact_add_lower = 0;
#pragma unroll
        for (int offset = neu_padded; offset < warp_size; offset += neu_padded) {
            const int tmp = __shfl_up_sync(0xFFFFFFFF, it_compact_add_self, offset, warp_size);
            if (lane >= offset) {
                it_compact_add_lower += tmp;
            }
        }
        if (iex_used != -1) {
            store[it_lo + it_compact + it_compact_add_lower] = mm_ids_helper_store(it, iex_used);
        }
        it_compact += __shfl_sync(0xFFFFFFFF, it_compact_add_lower + it_compact_add_self, warp_size - 1, warp_size);
    }
    nex_prev = warp_reduce_sum<warp_size>(nex_prev);
    if (lane == 0) {
        warp_count[warp] = it_compact;
        warp_prev[warp]  = nex_prev;
    }
    __syncthreads();

    int before = 0; // entries of the warps before this one
    int total  = 0;
    int prev   = 0; // columns of the experts with a lower index
#pragma unroll
    for (int w = 0; w < nwarps; ++w) {
        before += w < warp ? warp_count[w] : 0;
        total  += warp_count[w];
        prev   += warp_prev[w];
    }

    for (int itc = lane; itc < it_compact; itc += warp_size) {
        const mm_ids_helper_store store_it = store[it_lo + itc];
        const int it       = store_it.it();
        const int iex_used = store_it.iex_used();
        const int col      = prev + before + itc;
        ids_dst[col] = it*n_expert_used + iex_used;
        if (write_inverse) {
            ids_src1[it*n_expert_used + iex_used] = col;
        } else {
            ids_src1[col] = it*sis1 + iex_used % nchannels_y;
        }
    }

    if (warp != 0 || lane != 0) {
        return;
    }
    expert_bounds[expert] = prev;
    if (expert == static_cast<int>(gridDim.x) - 1) {
        expert_bounds[gridDim.x] = prev + total;
    }
}

template <int n_expert_used_template>
static void launch_mm_ids_helper_par(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    constexpr int nwarps = 8;
    GGML_ASSERT(n_tokens < (1 << 22) && "too few bits in mm_ids_helper_store");

    const int id = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[id].warp_size;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
    CUDA_SET_SHARED_MEMORY_LIMIT((mm_ids_helper_par<n_expert_used_template, nwarps>), smpbo);

    const size_t nbytes_shared = 2*nwarps*sizeof(int) + n_tokens*sizeof(mm_ids_helper_store);
    GGML_ASSERT(nbytes_shared <= smpbo);
    mm_ids_helper_par<n_expert_used_template, nwarps><<<dim3(n_experts, 1, 1), dim3(warp_size, nwarps, 1), nbytes_shared, stream>>>
        (ids, ids_src1, ids_dst, expert_bounds, n_tokens, nchannels_y, si1, sis1, write_inverse);
}

template <int n_expert_used_template>
static void launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    GGML_ASSERT(n_tokens          < (1 << 22) && "too few bits in mm_ids_helper_store");
    GGML_ASSERT(n_expert_used_var < (1 << 10) && "too few bits in mm_ids_helper_store");

    const int id = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[id].warp_size;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
    CUDA_SET_SHARED_MEMORY_LIMIT(mm_ids_helper<n_expert_used_template>, smpbo);

    const dim3 num_blocks(n_experts, 1, 1);
    const dim3 block_size(warp_size, 1, 1);
    const size_t nbytes_shared = n_tokens*sizeof(mm_ids_helper_store);
    GGML_ASSERT(nbytes_shared <= smpbo);
    mm_ids_helper<n_expert_used_template><<<num_blocks, block_size, nbytes_shared, stream>>>
        (ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used_var, nchannels_y, si1, sis1, write_inverse);
}

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    switch (n_expert_used) {
        case  2:
            launch_mm_ids_helper< 2>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  4:
            launch_mm_ids_helper< 4>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  6:
            launch_mm_ids_helper< 6>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  8:
            launch_mm_ids_helper< 8>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 10:
            launch_mm_ids_helper<10>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 16:
            launch_mm_ids_helper<16>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 32:
            launch_mm_ids_helper<32>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        default:
            launch_mm_ids_helper< 0>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
    }
}

bool ggml_cuda_launch_mm_ids_helper_par(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    if (ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size != 32) {
        return false;
    }
    switch (n_expert_used) {
        case  4: launch_mm_ids_helper_par< 4>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, nchannels_y, si1, sis1, write_inverse, stream); return true;
        case  6: launch_mm_ids_helper_par< 6>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, nchannels_y, si1, sis1, write_inverse, stream); return true;
        case  8: launch_mm_ids_helper_par< 8>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, nchannels_y, si1, sis1, write_inverse, stream); return true;
        case 10: launch_mm_ids_helper_par<10>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, nchannels_y, si1, sis1, write_inverse, stream); return true;
        case 16: launch_mm_ids_helper_par<16>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, nchannels_y, si1, sis1, write_inverse, stream); return true;
        default: return false;
    }
}
