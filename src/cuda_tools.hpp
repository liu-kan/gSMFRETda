#include <cuda_runtime.h>
#include <cuda.h>
#include <memory>
static void HandleError( cudaError_t err )
{
	// CUDA error handeling from the "CUDA by example" book
	if (err != cudaSuccess)
    {
		printf( "%s  \n", cudaGetErrorString( err ) );
		exit( EXIT_FAILURE );
	}
}
#define HANDLE_ERROR( err ) (HandleError( err))