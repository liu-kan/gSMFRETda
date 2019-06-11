#ifndef rmm_cuh_INCLUDED
#define rmm_cuh_INCLUDED
#include <cuda_runtime_api.h>
#include <cstring>
#include <rmm/rmm.h>
#include <cassert>
using namespace std;

#define cudaSucceeded(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define rmmSucceeded(ans) { rmmAssert((ans), __FILE__, __LINE__); }
inline void rmmAssert(rmmError_t code, const char *file, int line, bool abort=true) {
    if (code != RMM_SUCCESS) {
        fprintf(stderr, "RMMassert: %s %d\n", file, line);
        if (abort) exit(code);
    }
}

cudaError_t _rmmAlloc(void **ptr, size_t sz, cudaStream_t stream) {
    rmmError_t res = RMM_ALLOC(ptr, sz, stream);
    rmmSucceeded(res);
    if (res != RMM_SUCCESS) return cudaErrorMemoryAllocation;
    return cudaSuccess;
}

cudaError_t _rmmReAlloc(void **ptr, size_t sz, cudaStream_t stream) {
    rmmError_t res = RMM_REALLOC(ptr, sz, stream);
    rmmSucceeded(res);
    if (res != RMM_SUCCESS) return cudaErrorMemoryAllocation;
    return cudaSuccess;
}

cudaError_t _rmmFree(void *ptr, cudaStream_t stream) {
    rmmError_t res = RMM_FREE(ptr, stream);
    rmmSucceeded(res);
    if (res != RMM_SUCCESS) return cudaErrorMemoryAllocation;
    return cudaSuccess;
}

enum Allocator {
    cudaDefault = 0,
    rmmDefault,
    rmmManaged,
    rmmDefaultPool,
    rmmManagedPool
};

void setAllocator(const std::string alloc) {
    // if (alloc == "cudaDefault") {
    //     gpuAlloc = cudaMalloc;
    //     gpuFree = cudaFree;
    //     return;
    // }
    // else {
        rmmOptions_t options{CudaDefaultAllocation, 0, false};
        if (alloc == "rmmManaged")
            options.allocation_mode = CudaManagedMemory;
        else if (alloc == "rmmDefaultPool")
            options.allocation_mode = PoolAllocation;
        else if (alloc == "rmmManagedPool")
            options.allocation_mode = 
                static_cast<rmmAllocationMode_t>(PoolAllocation | 
                                                 CudaManagedMemory);
        else assert(alloc == "rmmDefault");
        rmmInitialize(&options);
        // gpuAlloc = _rmmAlloc;
        // gpuFree = _rmmFree;
        return;
    // }
}



#endif