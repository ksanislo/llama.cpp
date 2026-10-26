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
    // per granule: first granule of its mapping (each mapping maps its own allocation from offset 0; a mapping is unmapped
    // only as a whole), and whether it maps another region's memory (ggml_cuda_vmm_region_alias, not counted as backed)
    std::vector<size_t>          map_start;
    std::vector<size_t>          map_len;   // at map_start: granules of the mapping
    std::vector<bool>            aliased;
    size_t                       n_alias = 0;
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

// as release, but the physical granules stay allocated in a spare list of the device (no cuMemRelease)
size_t ggml_cuda_vmm_region_release_spare(ggml_cuda_vmm_region * r, size_t offset, size_t size);

// as commit, the device's spare granules first (no cuMemCreate for them); zero = false leaves the memory as it is
bool ggml_cuda_vmm_region_commit_spare(ggml_cuda_vmm_region * r, size_t offset, size_t size, bool zero);

// as commit, one allocation and one mapping for each unbacked run of [offset, offset + size) instead of one per granule
// (granule by granule if the device cannot give a run in one piece); such a run is released only as a whole
bool ggml_cuda_vmm_region_commit_chunk(ggml_cuda_vmm_region * r, size_t offset, size_t size);

// map the memory backing src [src_offset, src_offset + size) also at dst [dst_offset, ...) (unbacked there, granule aligned):
// both ranges then read and write the same memory, nothing is allocated; one mapping per allocation of src, which must lie
// wholly in the range (cuMemMap maps a handle from offset 0 only); false (nothing changed) otherwise or if dst is backed
bool ggml_cuda_vmm_region_alias(ggml_cuda_vmm_region * dst, size_t dst_offset, const ggml_cuda_vmm_region * src, size_t src_offset, size_t size);

// frees the spare granules of a physical device, returns the bytes
size_t ggml_cuda_vmm_spare_trim(int physical_device);

// memset the backed granules only (the zero granule must stay zero)
void ggml_cuda_vmm_region_memset_backed(ggml_cuda_vmm_region * r, uint8_t value);

#endif
