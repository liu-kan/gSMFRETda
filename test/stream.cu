#include "helper_cuda.h"

#include <rmm/detail/error.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/default_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <rmm/mr/device/pool_memory_resource.hpp>
#include "rmm/detail/error.hpp"
#include "rmm/mr/device/cuda_memory_resource.hpp"
#include "rmm/mr/device/default_memory_resource.hpp"
#include "rmm/mr/device/device_memory_resource.hpp"
using Pool = rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>;

#define N (1024*1024) 
#define FULL_DATA_SIZE (N*20)
__global__ void kernel( int *a, int *b, int *c ) 
{ 
    int idx = threadIdx.x + blockIdx.x * blockDim.x; 
    if (idx < N) {
         int idx1 = (idx + 1) % 256; 
         int idx2 = (idx + 2) % 256; 
         float as = (a[idx] + a[idx1] + a[idx2]) / 3.0f; 
         float bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f; 
        //  printf("a=%d\n",a[idx]);
         c[idx] = (as + bs) / 2;
    }
}
int main( void ) { 
    cudaDeviceProp prop; 
    int whichDevice;
    checkCudaErrors( cudaGetDevice( &whichDevice ) );
    checkCudaErrors( cudaGetDeviceProperties( &prop, whichDevice ) );
    if (!prop.deviceOverlap) { 
        printf( "Device will not handle overlaps, so no " 
        "speed up from streams\n" );
        return 0; 
    }
        cudaEvent_t start, stop;
        float elapsedTime;
        // start the timers
        checkCudaErrors( cudaEventCreate( &start ) ); checkCudaErrors( cudaEventCreate( &stop ) );
        checkCudaErrors( cudaEventRecord( start, 0 ) );
        // initialize the streams 
        cudaStream_t stream0, stream1;
        checkCudaErrors( cudaStreamCreate( &stream0 ) );
        checkCudaErrors( cudaStreamCreate( &stream1 ) );
        int *host_a, *host_b, *host_c; 
        int *dev_a0, *dev_b0, *dev_c0; 
        int *dev_a1, *dev_b1, *dev_c1; 
        auto const max_pool =
        static_cast<std::size_t>(rmm::mr::detail::available_device_memory());
        Pool *mr=new Pool{rmm::mr::get_current_device_resource(),static_cast<std::size_t>(max_pool*0.75),
          static_cast<std::size_t>(max_pool*0.9)};

        //GPU buffers for stream1
        // allocate the memory on the GPU 
        // checkCudaErrors( cudaMalloc( (void**)&dev_a0, N * sizeof(int) ) );
        // rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource();
        dev_a0= mr->allocate(N * sizeof(int), stream0);

        // rmm::device_buffer buff(N * sizeof(int), stream0);
        checkCudaErrors(cudaStreamSynchronize(stream0));
        // dev_a0=(int *)buff.data();
        checkCudaErrors( cudaMalloc( (void**)&dev_b0, N * sizeof(int) ) );
        checkCudaErrors( cudaMalloc( (void**)&dev_c0, N * sizeof(int) ) );
checkCudaErrors( cudaMalloc( (void**)&dev_a1, N * sizeof(int) ) );
checkCudaErrors( cudaMalloc( (void**)&dev_b1, N * sizeof(int) ) );
checkCudaErrors( cudaMalloc( (void**)&dev_c1, N * sizeof(int) ) );
// allocate page-locked memory, used to stream 
checkCudaErrors( cudaHostAlloc( (void**)&host_a, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault ) );
checkCudaErrors( cudaHostAlloc( (void**)&host_b, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault ) );
checkCudaErrors( cudaHostAlloc( (void**)&host_c, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault ) );
for (int i=0; i<FULL_DATA_SIZE; i++) {
     host_a[i] = i; host_b[i] = rand();
}
// now loop over full data, in bite-sized chunks 
for (int i=0; i<FULL_DATA_SIZE; i+= N*2) { // copy the locked memory to the device, async 
    checkCudaErrors( cudaMemcpyAsync( dev_a0, host_a+i, N * sizeof(int),
    cudaMemcpyHostToDevice, stream0 ) );
    checkCudaErrors( cudaMemcpyAsync( dev_b0, host_b+i, N * sizeof(int),
    cudaMemcpyHostToDevice, stream0 ) );
    kernel<<<N/256,256,0,stream0>>>( dev_a0, dev_b0, dev_c0 );
    // copy the data from device to locked memory 
    checkCudaErrors( cudaMemcpyAsync( host_c+i, dev_c0, N * sizeof(int),
    cudaMemcpyDeviceToHost,
    stream0 ) );

    // copy the locked memory to the device, async 
    checkCudaErrors( cudaMemcpyAsync( dev_a1, host_a+i+N, N * sizeof(int),
cudaMemcpyHostToDevice, stream1 ) );
checkCudaErrors( cudaMemcpyAsync( dev_b1, host_b+i+N, N * sizeof(int),
cudaMemcpyHostToDevice,
stream1 ) );
kernel<<<N/256,256,0,stream1>>>( dev_a1, dev_b1, dev_c1 ); // copy the data from device to locked memory 
checkCudaErrors( cudaMemcpyAsync( host_c+i+N, dev_c1, N * sizeof(int),
cudaMemcpyDeviceToHost,
stream1 ) );
}
checkCudaErrors( cudaStreamSynchronize( stream0 ) ); checkCudaErrors( cudaStreamSynchronize( stream1 ) );

checkCudaErrors( cudaEventRecord( stop, 0 ) ); checkCudaErrors( cudaEventSynchronize( stop ) ); 
checkCudaErrors( cudaEventElapsedTime( &elapsedTime, start, stop ) );
printf( "Time taken: %3.1f ms\n", elapsedTime );
// cleanup the streams and memory 
checkCudaErrors( cudaFreeHost( host_a ) ); checkCudaErrors( cudaFreeHost( host_b ) );
checkCudaErrors( cudaFreeHost( host_c ) );

// checkCudaErrors( cudaFree( dev_a0 ) ); 
// rmm::device_memory_resource::deallocate(dev_a0, N * sizeof(int), stream0);
mr->deallocate(dev_a0, N * sizeof(int), stream0);
delete(mr);
checkCudaErrors( cudaFree( dev_b0 ) ); checkCudaErrors( cudaFree( dev_c0 ) ); 
checkCudaErrors( cudaFree( dev_a1 ) ); checkCudaErrors( cudaFree( dev_b1 ) ); checkCudaErrors( cudaFree( dev_c1 ) );
 checkCudaErrors( cudaStreamDestroy( stream0 ) ); checkCudaErrors( cudaStreamDestroy( stream1 ) );
return 0;
}