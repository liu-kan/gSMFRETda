#include <cuda_runtime_api.h>
#include "GenRandTest.hpp"
#include "binom.cuh"
#include "gen_rand.cuh"
#include <random>
#include "hist.hpp"

void GenRandTest::SetUp(){
  printf("GenRandTest SetUp()\n");
  int nDevices=0;
  cudaGetDeviceCount(&nDevices);
  ASSERT_GE(nDevices, 1) << "You need at least 1 NVIDIA GPU to run tests!";
  CUDA_CHECK_RETURN(cudaSetDevice(0));
  // cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, mc_kernel, 0, 0);
  // blockSize=128;
  // printf("blockSize = %d\n", blockSize);
}

void GenRandTest::TearDown() {
  cudaDeviceReset();
  std::cout<<"cudaDeviceReset done!\n";
}
void GenRand::init_randstate(int N){
  cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, setup_kernel, 0, 0);
  std::cout<<"blockSize: "<<blockSize<<std::endl;
  CURAND_CALL(curandGetDirectionVectors64(
              &hostVectors64, CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6));
  CURAND_CALL(curandGetScrambleConstants64(&hostScrambleConstants64));  
  int NN;
  NN = N;
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
void GenRand::init_mem(int N){
  cudaError_t e = cudaMalloc((void **)&int_res, N * sizeof(int));
  ASSERT_EQ(e, cudaSuccess) << "cudaMalloc failed!";
}

__global__ void
test_drawDisIdx_kernel(int n,float* p, int N, int* int_res,
          curandStateScrambledSobol64* devQStates) 
{
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  if (tidx < N) {
    int_res[tidx] = drawDisIdx(n, p, devQStates + tidx);
  }
}

void GenRand::test_drawDisIdx(int n){
  // using namespace boost::histogram;
  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<float> dis(0.0, 1.0);
  float psum=0;
  float *rawp=new float[n*30];
  for (int i = 0; i < n*30; i++) {
    rawp[i]=dis(gen);
  }
  
  std::ostringstream os;
  getoss(rawp,n,os);

  std::cout << os.str() << std::flush;

  // test_drawDisIdx_kernel<<<gridSize, blockSize>>>(n,p,N,int_res);
}