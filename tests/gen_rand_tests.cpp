#include <gtest/gtest.h>
#include <tuple>
#include <iostream>
#include <vector>

#include "GenRandTest.hpp"

std::vector<int> ints_n;
std::vector<int> ints_N;
std::vector<float> floats;

TEST_P(GenRandTest, drawDisIdx) {
  GenRand gr;
  int N = std::get<0>(GetParam());
  int n = std::get<1>(GetParam());
  gr.init_randstate(N);
  gr.init_mem(N,n); 
  gr.test_drawDisIdx(n);
  gr.free_mem();
  gr.free_randstate();
}


TEST_P(GenRandTest, binomial) {
  GenRand gr;
  int N = std::get<0>(GetParam());
  int n = std::get<1>(GetParam());
  float f = std::get<2>(GetParam());
  gr.init_randstate(N);
  gr.init_mem(N,n); 
  gr.test_binomial(n,f);
  gr.free_mem();
  gr.free_randstate();
}

TEST_P(GenRandTest, multinomial) {
  GenRand gr;
  int N = std::get<0>(GetParam());
  int K = std::get<1>(GetParam());
  float p = std::get<2>(GetParam());
  gr.init_randstate(N);
  gr.init_mem(N,K); 
  gr.test_multinomial(K,p);
  gr.free_mem();
  gr.free_randstate();
}

TEST_P(GenRandTest, drawTau) {
  GenRand gr;
  int N = std::get<0>(GetParam());
  int n = 4;
  float f = std::get<2>(GetParam());
  gr.init_randstate(N);
  gr.init_mem(N,n); 
  gr.test_drawTau(f);
  gr.free_mem();
  gr.free_randstate();
}

TEST_P(GenRandTest, drawJ_Si2Sj) {
    GenRand gr;
    int N = std::get<0>(GetParam());
    int n = std::get<1>(GetParam());
    gr.init_randstate(N);
    gr.init_mem(N, n);
    gr.test_drawJ_Si2Sj(n);
    gr.free_mem();
    gr.free_randstate();
}

INSTANTIATE_TEST_CASE_P(CombinBurstSizeAndParams,
  GenRandTest,
  ::testing::Combine(::testing::ValuesIn(ints_N),
                     ::testing::ValuesIn(ints_n),
                     ::testing::ValuesIn(floats)));


int main(int argc, char **argv) {    
    ints_n.push_back(3);
    ints_n.push_back(4);
    ints_n.push_back(5);
    ints_n.push_back(6);
    ints_N.push_back(1000);
    ints_N.push_back(10000);
    ints_N.push_back(20000);
    ints_N.push_back(30000);
    floats.push_back(0.41);
    floats.push_back(0.54);
    floats.push_back(2.1);
    floats.push_back(3.3);
    floats.push_back(334.3);
    floats.push_back(4343);
    floats.push_back(72343);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
  }