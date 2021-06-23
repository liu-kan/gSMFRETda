#pragma once
#include <curand_kernel.h>
#include "cuda_tools.hpp"
#include <gtest/gtest.h>
typedef struct {
  unsigned int xor128[4];
  double gauss;
  int has_gauss; // !=0: gauss contains a gaussian deviate
  int has_binomial; // !=0: following parameters initialized for binomial
  /* The rk_state structure has been extended to store the following
   * information for the binomial generator. If the input values of n or p
   * are different than nsave and psave, then the other parameters will be
   * recomputed. RTK 2005-09-02 */
  int nsave, m;
  double psave, r, q, fm, p1, xm, xl, xr, c, laml, lamr, p2, p3, p4;
} rk_state;

class GenRandTest : public ::testing::TestWithParam<std::tuple<int, int>>  {
    protected:
        void SetUp() override;
        void TearDown() override ;
};

class GenRand{
    public:
        void init_randstate(int N);
        void init_mem(int N,int n);
        void free_mem();
        void test_drawDisIdx(int n);
        void free_randstate();
        void test_drawJ_Si2Sj(int n);
        GenRand();
        ~GenRand();
    private:        
        int gridSize;
        int minGridSize, blockSize;
        rk_state* devStates;
        curandStateScrambledSobol64* devQStates;
        curandDirectionVectors64_t* hostVectors64;
        unsigned long long int* hostScrambleConstants64;
        unsigned long long int* devDirectionVectors64;
        unsigned long long int* devScrambleConstants64;
        int *int_res;
        float* gp;
        int randstateN;
};