#include <gtest/gtest.h>
#include <tuple>
#include <iostream>
#include <vector>

#include "GenRandTest.hpp"

std::vector<int> ints_n;
std::vector<int> ints_N;
std::vector<float> floats;

//TEST_P(GenRandTest, Basic) {
//    std::cout << "Basic int: "        << std::get<0>(GetParam())
//              << "int: \"" << std::get<1>(GetParam())
//              << "\"\n";
//    GenRand gr;
//    gr.init_randstate(std::get<0>(GetParam()));
//}

TEST_P(GenRandTest, drawDisIdx) {
  GenRand gr;
  int N = std::get<0>(GetParam());
  int n = std::get<1>(GetParam());
  gr.init_randstate(N);
  gr.init_mem(N,n); 
  gr.test_drawDisIdx(n);
  gr.free_mem();
}

INSTANTIATE_TEST_CASE_P(CombinBurstSizeAndParams,
  GenRandTest,
  ::testing::Combine(::testing::ValuesIn(ints_N),
                     ::testing::ValuesIn(ints_n)));

int main(int argc, char **argv) {
    ints_n.push_back(2);
    ints_n.push_back(3);
    ints_n.push_back(4);
    ints_n.push_back(5);
    ints_N.push_back(1000);
    ints_N.push_back(10000);
    ints_N.push_back(20000);
    ints_N.push_back(30000);
    floats.push_back(3.12);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
  }