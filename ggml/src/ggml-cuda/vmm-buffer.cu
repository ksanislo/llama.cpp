#include "vmm-buffer.cuh"

#if defined(GGML_CUDA_VMM_BUFFER)

static CUresult ggml_cuda_vmm_create(CUmemGenericAllocationHandle * handle, size_t size, int physical_device) {
    CUmemAllocationProp prop = {};
    prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id   = physical_device;

    return cuMemCreate(handle, size, &prop, 0);
}

static void ggml_cuda_vmm_map_zero(ggml_cuda_vmm_region * r, size_t g0, size_t g1) {
    for (size_t g = g0; g < g1; ++g) {
        CU_CHECK(cuMemMap(r->base + g*r->granularity, r->granularity, 0, r->zero, 0));
    }
    CU_CHECK(cuMemSetAccess(r->base + g0*r->granularity, (g1 - g0)*r->granularity, r->access.data(), r->access.size()));
}

// the mappings change under the address range, so no kernel may use it meanwhile
static void ggml_cuda_vmm_sync(const ggml_cuda_vmm_region * r) {
    ggml_cuda_set_device(r->device);
    CUDA_CHECK(cudaDeviceSynchronize());
}

ggml_cuda_vmm_region * ggml_cuda_vmm_region_create(int device, int physical_device, size_t granularity, size_t size,
        const std::vector<CUmemAccessDesc> & access) {
    GGML_ASSERT(granularity > 0 && !access.empty());

    auto * r = new ggml_cuda_vmm_region;
    r->device          = device;
    r->physical_device = physical_device;
    r->granularity     = granularity;
    r->size            = GGML_PAD(std::max<size_t>(size, 1), granularity);
    r->access          = access;
    r->backed.assign(r->size/granularity, false);

    ggml_cuda_set_device(device);

    if (cuMemAddressReserve(&r->base, r->size, 0, 0, 0) != CUDA_SUCCESS) {
        GGML_LOG_ERROR("%s: cannot reserve %.2f MiB of address space on device %d\n", __func__, r->size/1024.0/1024.0, device);
        delete r;
        return nullptr;
    }

    if (ggml_cuda_vmm_create(&r->zero, granularity, physical_device) != CUDA_SUCCESS) {
        GGML_LOG_ERROR("%s: cannot allocate the zero granule on device %d\n", __func__, device);
        CU_CHECK(cuMemAddressFree(r->base, r->size));
        delete r;
        return nullptr;
    }

    ggml_cuda_vmm_map_zero(r, 0, r->backed.size());

    // every unbacked granule aliases this one
    CUDA_CHECK(cudaMemset((void *) r->base, 0, granularity));
    CUDA_CHECK(cudaDeviceSynchronize());

    return r;
}

void ggml_cuda_vmm_region_free(ggml_cuda_vmm_region * r) {
    if (r == nullptr) {
        return;
    }

    ggml_cuda_vmm_sync(r);

    CU_CHECK(cuMemUnmap(r->base, r->size));
    CU_CHECK(cuMemAddressFree(r->base, r->size));
    CU_CHECK(cuMemRelease(r->zero));

    delete r;
}

bool ggml_cuda_vmm_region_commit(ggml_cuda_vmm_region * r, size_t offset, size_t size) {
    if (size == 0 || offset >= r->size) {
        return true;
    }

    const size_t g0 = offset/r->granularity;
    const size_t g1 = std::min(r->backed.size(), (offset + size + r->granularity - 1)/r->granularity);

    std::vector<size_t> todo;
    for (size_t g = g0; g < g1; ++g) {
        if (!r->backed[g]) {
            todo.push_back(g);
        }
    }

    if (todo.empty()) {
        return true;
    }

    ggml_cuda_set_device(r->device);

    // get all the memory first, so that a failure changes nothing
    std::vector<CUmemGenericAllocationHandle> handles(todo.size());
    for (size_t i = 0; i < todo.size(); ++i) {
        if (ggml_cuda_vmm_create(&handles[i], r->granularity, r->physical_device) != CUDA_SUCCESS) {
            for (size_t j = 0; j < i; ++j) {
                CU_CHECK(cuMemRelease(handles[j]));
            }
            return false;
        }
    }

    ggml_cuda_vmm_sync(r);

    for (size_t i = 0; i < todo.size(); ) {
        size_t n = 1;
        while (i + n < todo.size() && todo[i + n] == todo[i] + n) {
            n++;
        }

        const CUdeviceptr ptr = r->base + todo[i]*r->granularity;

        CU_CHECK(cuMemUnmap(ptr, n*r->granularity));
        for (size_t j = 0; j < n; ++j) {
            CU_CHECK(cuMemMap(ptr + j*r->granularity, r->granularity, 0, handles[i + j], 0));
            CU_CHECK(cuMemRelease(handles[i + j])); // the mapping keeps the memory
            r->backed[todo[i + j]] = true;
        }
        CU_CHECK(cuMemSetAccess(ptr, n*r->granularity, r->access.data(), r->access.size()));

        // free cells are read by the attention (masked), so they must not hold NaN
        CUDA_CHECK(cudaMemsetAsync((void *) ptr, 0, n*r->granularity, cudaStreamPerThread));

        i += n;
    }
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));

    r->n_backed += todo.size();

    return true;
}

size_t ggml_cuda_vmm_region_release(ggml_cuda_vmm_region * r, size_t offset, size_t size) {
    if (size == 0 || offset >= r->size) {
        return 0;
    }

    const size_t g0 = (offset + r->granularity - 1)/r->granularity;
    const size_t g1 = std::min(r->backed.size(), (offset + size)/r->granularity);

    size_t n_freed = 0;
    bool synced = false;

    for (size_t g = g0; g < g1; ) {
        if (!r->backed[g]) {
            g++;
            continue;
        }

        size_t n = 1;
        while (g + n < g1 && r->backed[g + n]) {
            n++;
        }

        if (!synced) {
            ggml_cuda_vmm_sync(r);
            synced = true;
        }

        CU_CHECK(cuMemUnmap(r->base + g*r->granularity, n*r->granularity));
        ggml_cuda_vmm_map_zero(r, g, g + n);

        for (size_t j = 0; j < n; ++j) {
            r->backed[g + j] = false;
        }

        n_freed += n;
        g += n;
    }

    r->n_backed -= n_freed;

    return n_freed*r->granularity;
}

void ggml_cuda_vmm_region_memset_backed(ggml_cuda_vmm_region * r, uint8_t value) {
    ggml_cuda_set_device(r->device);

    for (size_t g = 0; g < r->backed.size(); ) {
        if (!r->backed[g]) {
            g++;
            continue;
        }

        size_t n = 1;
        while (g + n < r->backed.size() && r->backed[g + n]) {
            n++;
        }

        CUDA_CHECK(cudaMemsetAsync((void *) (r->base + g*r->granularity), value, n*r->granularity, cudaStreamPerThread));

        g += n;
    }
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

#endif // defined(GGML_CUDA_VMM_BUFFER)
