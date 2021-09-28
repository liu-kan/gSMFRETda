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
    ints_n.push_back(2);
    ints_n.push_back(3);
    ints_n.push_back(4);
    ints_n.push_back(5);
    ints_N.push_back(1000);
    ints_N.push_back(10000);
    ints_N.push_back(20000);
    ints_N.push_back(30000);
    floats.push_back(32.6);
    floats.push_back(312);
    floats.push_back(3125);
    floats.push_back(23125.7);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
  }