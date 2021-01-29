#include <cuda_runtime_api.h>
#include "GenRandTest.hpp"
#include "binom.cuh"
#include "gen_rand.cuh"
#include <random>
#include "hist.hpp"
#include <vector>
#include <dlib/statistics.h>
void GenRandTest::SetUp(){
  printf("GenRandTest SetUp()\n");
  int nDevices=0;
  cudaGetDeviceCount(&nDevices);
  ASSERT_GE(nDevices, 1) << "You need at least 1 NVIDIA GPU to run tests!";  
}

void GenRandTest::TearDown() {
    printf("GenRandTest TearDown()\n");
}

GenRand::GenRand() {
    randstateN = 0;
    CUDA_CHECK_RETURN(cudaSetDevice(0));
    devDirectionVectors64=NULL;
    devScrambleConstants64=NULL;   
    int_res=NULL;
}
GenRand::~GenRand() {
    cudaDeviceReset();
    std::cout << "cudaDeviceReset done!\n";
}
/**
 * @brief init randstate
 * 
 * @param N Number of samples
 */
void GenRand::init_randstate(int N){
  // cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, setup_kernel, 0, 0);
  blockSize=256;
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
                                NN * sizeof(unsigned long long int)));
  int n = 0;
  int tNN = NN;
  while (tNN > 0) {
      int size = (tNN > 20000) ? 20000 : tNN;
      std::cout << "n= " << n << std::endl;
      CUDA_CHECK_RETURN(cudaMemcpy(devScrambleConstants64 + n * 20000, hostScrambleConstants64,
                                        size * sizeof(unsigned long long),
                                        cudaMemcpyHostToDevice));
      // std::cout << "n = " << n << ", size = " << size << std::endl;
      //buf = devDirectionVectors64;
      CUDA_CHECK_RETURN(
          cudaMemcpy(devDirectionVectors64 + n * 20000 * sizeof(curandDirectionVectors64_t) /
                                    sizeof(unsigned long long),
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
void GenRand::init_mem(int N,int n){
    cudaError_t e = cudaMalloc((void**)&gp, n * sizeof(float));
    ASSERT_EQ(e, cudaSuccess) << "cudaMalloc failed!";
    e = cudaMalloc((void **)&int_res, N * sizeof(int));
    ASSERT_EQ(e, cudaSuccess) << "cudaMalloc failed!";
    randstateN = N;
}
void GenRand::free_mem() {    
    std::cout << " devDirectionVectors64: " << static_cast<void*>(devDirectionVectors64) << "\n";
    CUDA_CHECK_RETURN(cudaFree(devDirectionVectors64));
    CUDA_CHECK_RETURN(cudaFree(devScrambleConstants64));
    if (randstateN > 0) {
        CUDA_CHECK_RETURN(cudaFree(int_res));        
    }

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

/**
 * @brief 
 * 
 * @param n Number of states use in hist
 */
void GenRand::test_drawDisIdx(int n){
  // using namespace boost::histogram;
  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<float> dis(0.0, 1.0);

  int scount = n * 30;
  float *rawp=new float[scount];
  int* res_c = new int[randstateN];
  float* p = new float[n];
  for (int i = 0; i < scount; i++) {
    rawp[i]=dis(gen);
  }  
  std::ostringstream os;
  getoss(rawp, scount, n,os,p);
  std::vector<float> tp(p, p+n);
  CUDA_CHECK_RETURN(cudaMemcpy(gp, p,n * sizeof(float),cudaMemcpyHostToDevice));
  std::cout << os.str() << std::flush;  
  test_drawDisIdx_kernel<<<gridSize, blockSize>>>(n,gp,randstateN,int_res,devQStates);
  CUDA_CHECK_RETURN(cudaMemcpy(res_c, int_res, randstateN * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK_RETURN(cudaDeviceSynchronize());  
  os.str("");
  os.clear();
  getoss_i(res_c, randstateN, n, os, p);
  std::cout << os.str() << std::flush;
  std::vector<float> sp(p, p + n);  
  float r2 = dlib::r_squared(sp, tp);
  std::cout << "r2: " << r2 << std::endl;
  if(randstateN>10000)
    EXPECT_GE(r2, 0.78);
  delete[] p;
  delete[] rawp;
  delete[] res_c;
  
}