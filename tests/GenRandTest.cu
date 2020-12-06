#include <cuda_runtime_api.h>
#include "GenRandTest.hpp"
#include "binom.cuh"
#include "gen_rand.cuh"

void GenRandTest::SetUp(){
  printf("GenRandTest SetUp()\n");
  int nDevices=0;
  cudaGetDeviceCount(&nDevices);
  ASSERT_GE(nDevices, 1) << "You need at least 1 NVIDIA GPU to run tests!";
  CUDA_CHECK_RETURN(cudaSetDevice(nDevices-1));
  // cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, mc_kernel, 0, 0);
  // blockSize=128;
  // printf("blockSize = %d\n", blockSize);
}

void GenRandTest::TearDown() {
  cudaDeviceReset();
  std::cout<<"cudaDeviceReset done!\n";
}
void GenRand::init_randstate(int N){
  CURAND_CALL(curandGetDirectionVectors64(
              &hostVectors64, CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6));
  CURAND_CALL(curandGetScrambleConstants64(&hostScrambleConstants64));  
  int NN;
  NN = N * reSampleTimes;
  gridSize = (NN + blockSize - 1) / blockSize;
  CUDA_CHECK_RETURN(cudaMalloc((void **)&devStates, NN * sizeof(rk_state)));
  CUDA_CHECK_RETURN(cudaMalloc((void **)&devQStates, NN * sizeof(curandStateScrambledSobol64)));
  CUDA_CHECK_RETURN(cudaMalloc((void **)&devDirectionVectors64, 
                                NN * VECTOR_SIZE * sizeof(long long int)));
  CUDA_CHECK_RETURN(cudaMalloc((void **)&devScrambleConstants64, 
                                NN * sizeof(long long int)));
  int n = 0;
  int tNN = NN;
  while (tNN > 0) {
      int size = (tNN > 20000) ? 20000 : tNN;
      unsigned long long int *buf = devScrambleConstants64;
      CUDA_CHECK_RETURN(cudaMemcpy(buf + n * 20000, hostScrambleConstants64,
                                        size * sizeof(unsigned long long int),
                                        cudaMemcpyHostToDevice));
      // std::cout << "n = " << n << ", size = " << size << std::endl;
      buf = devDirectionVectors64;
      CUDA_CHECK_RETURN(
          cudaMemcpy(buf + n * 20000 * sizeof(curandDirectionVectors64_t) /
                                    sizeof(unsigned long long int),
                          hostVectors64, size * sizeof(curandDirectionVectors64_t),
                          cudaMemcpyHostToDevice));
      tNN -= size;
      n++;
  }
  setup_kernel<<<gridSize, blockSize>>>(
      devStates, 0, /*time(NULL)*/ NN, devDirectionVectors64,
      devScrambleConstants64, devQStates);
  std::cout<<"setup_kernel done!\n";
  CUDA_CHECK_RETURN(cudaDeviceSynchronize());
}
