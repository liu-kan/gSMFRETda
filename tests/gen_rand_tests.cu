#include <gtest/gtest.h>
#include <tuple>
#include <iostream>
#include <vector>

#include "GenRandTest.hpp"

std::vector<int> ints_n;
std::vector<int> ints_N;
std::vector<float> floats;

TEST_P(GenRandTest, Basic) {
    std::cout << "int: "        << std::get<0>(GetParam())
              << "int: \"" << std::get<1>(GetParam())
              << "\"\n";
    GenRand gr;
    gr.init_randstate(std::get<0>(GetParam()));
}

TEST_P(GenRandTest, drawDisIdx) {
  GenRand gr;
  gr.init_randstate(std::get<0>(GetParam()));
  gr.test_drawDisIdx(std::get<1>(GetParam()));
}

INSTANTIATE_TEST_CASE_P(CombinBurstSizeAndParams,
  GenRandTest,
  ::testing::Combine(::testing::ValuesIn(ints_N),
                     ::testing::ValuesIn(ints_n)));

int main(int argc, char **argv) {
    ints_n.push_back(2);
    ints_n.push_back(3);
    ints_n.push_back(5);
    ints_N.push_back(1000);
    floats.push_back(3.12);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
  }