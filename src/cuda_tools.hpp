#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <memory>
#include <iostream>
//static void HandleError( cudaError_t err )
//{
//	// CUDA error handeling from the "CUDA by example" book
//	if (err != cudaSuccess)
//    {
//		printf( "%s  \n", cudaGetErrorString( err ) );
//		exit( EXIT_FAILURE );
//	}
//}
//#define HANDLE_ERROR( err ) (HandleError( err))


static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)
