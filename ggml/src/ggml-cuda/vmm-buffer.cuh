#pragma once

#include "common.cuh"

#include <vector>

#if defined(GGML_USE_VMM) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#define GGML_CUDA_VMM_BUFFER

// device buffer: address space for the whole size, physical memory backed per granule on request
// granules that are not backed map one shared zero granule: reads see zeros, writes are not allowed
struct ggml_cuda_vmm_region {
    int    device;
    int    physical_device;
    size_t granularity;
    size_t size;   // multiple of granularity

    CUdeviceptr                  base = 0;
    CUmemGenericAllocationHandle zero = 0;

    std::vector<CUmemAccessDesc> access;
    std::vector<bool>            backed;
    size_t                       n_backed = 0;
};

// nullptr if the address space or the zero granule cannot be had
ggml_cuda_vmm_region * ggml_cuda_vmm_region_create(int device, int physical_device, size_t granularity, size_t size,
        const std::vector<CUmemAccessDesc> & access);

void ggml_cuda_vmm_region_free(ggml_cuda_vmm_region * r);

// back every granule that [offset, offset + size) touches; the new memory is zeroed
// false (nothing changed) if the device has not enough memory
bool ggml_cuda_vmm_region_commit(ggml_cuda_vmm_region * r, size_t offset, size_t size);

// give back the granules that lie wholly inside [offset, offset + size); returns the bytes given back
size_t ggml_cuda_vmm_region_release(ggml_cuda_vmm_region * r, size_t offset, size_t size);

// memset the backed granules only (the zero granule must stay zero)
void ggml_cuda_vmm_region_memset_backed(ggml_cuda_vmm_region * r, uint8_t value);

#endif
