#include <cuda_runtime_api.h>
#include "GenRandTest.hpp"
#include "binom.cuh"
#include "gen_rand.cuh"
#include <random>
#include "hist.hpp"
#include <vector>
#include <dlib/statistics.h>
#include "ParamsTest.hpp"
#include "eigenhelper.hpp"
#include <algorithm>

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
    float_res=NULL;
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
      std::cout << "hostScrambleConstants64 n= " << n << std::endl;
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
  randstateN = N;
}
void GenRand::init_mem(int N,int n){
    cudaError_t e = cudaMalloc((void**)&gp, n * sizeof(float));
    ASSERT_EQ(e, cudaSuccess) << "cudaMalloc failed!";
    e = cudaMalloc((void **)&int_res, N * sizeof(int));
    ASSERT_EQ(e, cudaSuccess) << "cudaMalloc failed!";
    e = cudaMalloc((void **)&float_res, N * sizeof(float));
    ASSERT_EQ(e, cudaSuccess) << "cudaMalloc failed!";    
    randstateN = N;
}
void GenRand::free_mem() {    
    if (randstateN > 0) {
        CUDA_CHECK_RETURN(cudaFree(int_res));        
    }
}
void GenRand::free_randstate() {
    CUDA_CHECK_RETURN(cudaFree(devDirectionVectors64));
    CUDA_CHECK_RETURN(cudaFree(devScrambleConstants64));
    CUDA_CHECK_RETURN(cudaFree(devStates));
    CUDA_CHECK_RETURN(cudaFree(devQStates));
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

__global__ void
test_drawTau_kernel(float k, float* float_res, int N,
          curandStateScrambledSobol64* devQStates) 
{
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  if (tidx < N) {
    float_res[tidx] = drawTau(k, devQStates + tidx, 0);
  }
}

__global__ void
test_drawJ_Si2Sj_kernel(int n, float* P_i2j, float* gpp_i2j,int i, int N, int* int_res,
    curandStateScrambledSobol64* devQStates)
{
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx < N) {
        int_res[tidx] = drawJ_Si2Sj(P_i2j + tidx * n, gpp_i2j, n, i, devQStates+ tidx);
    }
}

void GenRand::test_drawJ_Si2Sj(int n) {
    float* g_P_i2j;
    float* gpp_i2j;
    CUDA_CHECK_RETURN(cudaMalloc((void**)&g_P_i2j,n* randstateN * sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&gpp_i2j, n * n * sizeof(float)));
    float* matK = new float[n * n];
    float* matP_i2j = new float[n * n];
    int* res_c = new int[randstateN];
    float* p = new float[n];
    std::ostringstream os;

    std::vector<float> args;
    MatCheck matCheck;
    matCheck.genRandMatk(n, args);
    vecFloatMapper evargs(args.data(), n * n + n);
    RowVectorXf eargs = evargs.block(0, 0, 1, n);
    float* peargs = eargs.data();
    RowVectorXf kargs = evargs.block(0, n, 1, n * n - n);
    RowVectorXf vargs = evargs.block(0, n * n, 1, n);
    float* pvargs = vargs.data();
    bool r = genMatK(matK, n, kargs);
    genP_i2j(matK, matP_i2j, n);
    matXfMapper matP_i2jmp(matP_i2j, n, n);
    CUDA_CHECK_RETURN(cudaMemcpy(gpp_i2j, matP_i2j, sizeof(float) * n * n,cudaMemcpyHostToDevice));
    for (int ni = 0; ni < n; ni++) {
        test_drawJ_Si2Sj_kernel <<<gridSize, blockSize >>> (n, g_P_i2j, gpp_i2j, ni,randstateN, int_res, devQStates);
        os.str("");
        os.clear();
        CUDA_CHECK_RETURN(cudaMemcpy(res_c, int_res, randstateN * sizeof(int), cudaMemcpyDeviceToHost));
        getoss_i(res_c, randstateN, n, os, p);
        std::vector<float> sp(p, p + n);
        std::vector<float> tp(matP_i2jmp.data()+ni*n, matP_i2jmp.data() + ni * n + n);
        float r2 = dlib::r_squared(sp, tp);
        std::cout << "test_drawJ_Si2Sj r2: " << r2 << std::endl;
        if (randstateN > 10000)
            EXPECT_GE(r2, 0.711);
    }

    delete[] matK;
    delete[] matP_i2j;
    delete[] res_c;
    delete[] p;
    CUDA_CHECK_RETURN(cudaFree(g_P_i2j));
    CUDA_CHECK_RETURN(cudaFree(gpp_i2j));
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
  std::cout << "test_drawDisIdx r2: " << r2 << std::endl;
  if(randstateN>10000)
    EXPECT_GE(r2, 0.711);
  delete[] p;
  delete[] rawp;
  delete[] res_c;  
}

//https://www.math.pku.edu.cn/teachers/lidf/docs/statcomp/html/_statcompbook/intro-graph.html
void GenRand::test_drawTau(float k){
    // using namespace boost::histogram;
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::exponential_distribution<float> dis(k);  
    int scount = randstateN;
    float *rawp=new float[scount];
    float* res_c = new float[randstateN];    
    for (int i = 0; i < scount; i++) {
      rawp[i]=dis(gen);
    }  
    float minp=*std::min_element(rawp,rawp+scount);
    float maxp=*std::max_element(rawp,rawp+scount);
    int n=100;
    std::ostringstream os;
    float* p = new float[n];

    test_drawTau_kernel<<<gridSize, blockSize>>>(k,float_res,randstateN,devQStates);
    CUDA_CHECK_RETURN(cudaMemcpy(res_c, float_res, randstateN * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());  

    float minpc=*std::min_element(res_c,res_c+scount);
    float maxpc=*std::max_element(res_c,res_c+scount);
    float minpp=std::max(minp,minpc);
    float maxpp=std::min(maxp,maxpc);
    getoss(rawp, scount, n,os,p,minpp,maxpp);
    std::vector<float> tp(p, p+n);
    std::cout << os.str() << std::flush;  
    os.str("");
    os.clear();
    getoss(res_c, scount, n,os,p,minpp,maxpp);
    std::cout << os.str() << std::flush;
    std::vector<float> sp(p, p + n);  
    float r2 = dlib::r_squared(sp, tp);
    std::cout << "test_drawTau r2: " << r2 << std::endl;
    if(randstateN>10000)
      EXPECT_GE(r2, 0.711);
    delete[] p;
    delete[] rawp;
    delete[] res_c;  
  }
  