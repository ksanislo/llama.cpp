#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// device buffer with address space reserved for the whole size, memory backed on demand (VMM)
// unbacked ranges read as zeros and must not be written; alloc returns NULL when VMM is not available
GGML_BACKEND_API ggml_backend_buffer_t ggml_backend_cuda_vmm_buffer_alloc(ggml_backend_buffer_type_t buft, size_t size);
GGML_BACKEND_API bool   ggml_backend_cuda_vmm_buffer_commit(ggml_backend_buffer_t buffer, size_t offset, size_t size);
GGML_BACKEND_API size_t ggml_backend_cuda_vmm_buffer_release(ggml_backend_buffer_t buffer, size_t offset, size_t size);
GGML_BACKEND_API size_t ggml_backend_cuda_vmm_buffer_release_spare(ggml_backend_buffer_t buffer, size_t offset, size_t size);
GGML_BACKEND_API bool   ggml_backend_cuda_vmm_buffer_commit_spare(ggml_backend_buffer_t buffer, size_t offset, size_t size, bool zero);
// as commit, one allocation per unbacked run (fewer, larger mappings); such a run is released only as a whole
GGML_BACKEND_API bool   ggml_backend_cuda_vmm_buffer_commit_chunk(ggml_backend_buffer_t buffer, size_t offset, size_t size);
// map the memory backing src [src_offset, +size) also at dst [dst_offset, +size) (granule aligned, unbacked in dst): no allocation
GGML_BACKEND_API bool   ggml_backend_cuda_vmm_buffer_alias(ggml_backend_buffer_t dst, size_t dst_offset, ggml_backend_buffer_t src, size_t src_offset, size_t size);
GGML_BACKEND_API size_t ggml_backend_cuda_vmm_spare_trim(ggml_backend_buffer_type_t buft);
GGML_BACKEND_API size_t ggml_backend_cuda_vmm_buffer_get_backed_size(ggml_backend_buffer_t buffer);
GGML_BACKEND_API size_t ggml_backend_cuda_vmm_get_granularity(ggml_backend_buffer_type_t buft);

// conduct allreduce operation between devices
GGML_BACKEND_API bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

#ifdef  __cplusplus
}
#endif
