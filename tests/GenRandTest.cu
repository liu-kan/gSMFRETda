// #define ENABLE_ASSERTS
// #define DLIB_NO_GUI_SUPPORT

#include <cuda_runtime_api.h>
#include "GenRandTest.hpp"
#include "binom.cuh"
#include "gen_rand.cuh"
#include <random>
#include "hist.hpp"
#include <stdio.h>
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
        // getoss<int,float>(res_c, randstateN, n, os, p, (float)0.0, (float)n);
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
  if (n<4)
    return;
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
  // getoss<int,float>(res_c, randstateN, n, os, p,(float)0.0, (float)n);
  std::cout << os.str() << std::flush;
  std::vector<float> sp(p, p + n);  
  float r2 = dlib::r_squared(sp, tp);
  std::cout << "test_drawDisIdx r2: " << r2 << std::endl;
  if(randstateN>1000 && n>2)
    EXPECT_GE(r2, 0.6);
  delete[] p;
  delete[] rawp;
  delete[] res_c;  
}

//https://www.math.pku.edu.cn/teachers/lidf/docs/statcomp/html/_statcompbook/intro-graph.html
void GenRand::test_drawTau(float k){
  if (k<0.2)
    return;  
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
 
__global__ void
test_binomial_kernel(int n, float p, int* int_res, int N, rk_state* devStates) 
{
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  if (tidx < N) {
      int_res[tidx] = (int)rk_binomial(devStates + tidx, n, p);
  }
}

__global__ void
test_multinomial_kernel(int N,int K,float *pp,int *g_n_res,int scount,rk_state* devStates){
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  if (tidx < scount) {
      csd_int_multinomial (devStates+tidx, K, N, pp, g_n_res+K*tidx);
  }
}

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

void GenRand::test_binomial(int n, float p){  
  if (p>1)
    p=1/p;
  if (p<0.2)
    return;
  const gsl_rng_type * rngT;
  gsl_rng * rng ;
  /* create a generator chosen by the
     environment variable GSL_RNG_TYPE */
  gsl_rng_env_setup();
  rngT = gsl_rng_default;
  rng = gsl_rng_alloc (rngT);

  float* pp = new float[n];
  int scount = randstateN;
  int *rawp  = new int[scount];
  int *res_c = new int[scount];

  for (int i = 0; i < scount; i++) {
    rawp[i]=gsl_ran_binomial(rng, p, n);
  }

  // std::cout<<std::endl << "rawp:" <<std::endl;
  // for (int i=0;i<scount;i++)
  //   std::cout << rawp[i] << ' ';    
  // std::cout<<std::endl;

  std::ostringstream os;
  getoss_i(rawp, scount, n, os, pp);
  // getoss<int,float>(rawp, scount, n, os, pp,(float)0.0, (float)n);
  std::cout << os.str() << std::flush;
  os.str("");
  os.clear();
  std::vector<float> tp(pp, pp + n );

  test_binomial_kernel<<<gridSize, blockSize>>>(n,p,int_res,scount,devStates);
  CUDA_CHECK_RETURN(cudaMemcpy(res_c, int_res, scount * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK_RETURN(cudaDeviceSynchronize());  
  
  // std::cout<<std::endl << "res_c:" <<std::endl;
  // for (int i=0;i<scount;i++)
  //   std::cout << res_c[i] << ' ';    
  // std::cout<<std::endl;  

  getoss_i(res_c, scount, n, os, pp);
  // getoss<int,float>(res_c, scount, n, os, pp,(float)0, (float)n);
  std::cout << os.str() << std::flush;  
  std::vector<float> sp(pp, pp + n );
  std::cout << "sp:" <<std::endl;
  for (auto i: sp)
    std::cout << i << ' ';
  std::cout<<std::endl<< "tp:" <<std::endl;
  for (auto i: tp)
    std::cout << i << ' ';    
  std::cout<<std::endl;
  float r2 = dlib::r_squared(sp, tp);
  std::cout << "test_binomial r2: " << r2 << std::endl;
  // if(randstateN>10000)
    EXPECT_GE(r2, 0.711);
  delete[] pp;
  delete[] rawp;
  delete[] res_c;  
}

#include <cmath>
#include <algorithm> 
#include <gsl/gsl_statistics_int.h> 
void GenRand::test_multinomial(int K, float p){  
  if (p<2 )
    return;
  std::default_random_engine generator;
  std::uniform_real_distribution<double> uni_dist(0,p);
  std::uniform_real_distribution<double> uni_dist_N(0.7,30);
  int scount = randstateN;
  double mu=uni_dist(generator)*K;
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<> d{mu,mu/2};
  double *p0  = new double[scount];
  for (int i=0;i<scount;i++){
    p0[i] = std::round(d(gen));
  }

  double minp=*std::min_element(p0,p0+scount);
  double maxp=*std::max_element(p0,p0+scount);

  int N=std::ceil(uni_dist_N(generator)*K);
  std::ostringstream os;
  double* ppd = new double[K];
  float* pp = new float[K];
  getoss_d(p0, scount, K,os,ppd,minp,maxp);
  std::transform(ppd, ppd + K, pp, [](double d) {return (float)d;});
  std::cout << os.str() << std::flush;  
  os.str("");
  os.clear();
  
  // printf("K=%d, scount=%d\n",K,scount);
  int* g_n_res=NULL;
  float* gpp=NULL;
  cudaError_t e = cudaMalloc((void**)&g_n_res, K * sizeof(int)*scount);
  ASSERT_EQ(e, cudaSuccess) << "cudaMalloc failed!";
  e = cudaMalloc((void**)&gpp, K * sizeof(float));
  ASSERT_EQ(e, cudaSuccess) << "cudaMalloc failed!";
  CUDA_CHECK_RETURN(cudaMemcpy(gpp, pp, K*sizeof(float), cudaMemcpyHostToDevice));
  test_multinomial_kernel<<<gridSize, blockSize>>>(N,K,gpp,g_n_res,scount,devStates);
  CUDA_CHECK_RETURN(cudaDeviceSynchronize());
  int *c_n_res=new int[ K *scount];
  // long *cg_n_res=new long[ K *scount];
  
  CUDA_CHECK_RETURN(cudaMemcpy(c_n_res, g_n_res, scount * K*sizeof(int), cudaMemcpyDeviceToHost));
  
  // std::transform(cg_n_res, cg_n_res + K*scount, c_n_res, [](long d) {return (int)d;});
  if(scount>1000){
    for(int i=0;i<K;i++){
      double meansp=gsl_stats_int_mean(c_n_res+i,K,scount);
      double varsp=gsl_stats_int_variance(c_n_res+i,K,scount);
      double mean=N*pp[i];
      double var=N*pp[i]*(1-pp[i]);
      printf("meanki=%lf,np=%lf,N=%d,pp[%d]=%lf\n",meansp,mean,N,i,pp[i]);
      EXPECT_LE(abs(meansp-mean)/mean,0.3);
      EXPECT_LE(abs(varsp-var)/var,0.3);
      for(int j=i+1;j<K;j++){
        double covarsp=gsl_stats_int_covariance(c_n_res+i,K,c_n_res+j,K,scount);
        double covar=-N*pp[i]*pp[j];
        EXPECT_LE(abs(covarsp-covar)/covar,0.3);
      }
    }
  }
  delete [] p0;
  delete[] pp;
  delete[] ppd;
  delete[] c_n_res;
  // delete[] cg_n_res;
  CUDA_CHECK_RETURN(cudaFree(g_n_res));     
  CUDA_CHECK_RETURN(cudaFree(gpp));
}
