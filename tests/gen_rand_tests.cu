#include <gtest/gtest.h>
#include <tuple>
#include <iostream>
#include <vector>

#include "GenRandTest.hpp"

std::vector<int> ints;
std::vector<float> floats;

TEST_P(GenRandTest, Basic) {
    std::cout << "int: "        << std::get<0>(GetParam())
              << "  float: \"" << std::get<1>(GetParam())
              << "\"\n";
    GenRand gr;
    gr.init_randstate(std::get<0>(GetParam()));
}

INSTANTIATE_TEST_CASE_P(CombinBurstSizeAndParams,
  GenRandTest,
  ::testing::Combine(::testing::ValuesIn(ints),
                     ::testing::ValuesIn(floats)));

int main(int argc, char **argv) {
    for (int i = 0; i < 2; ++i) {
      ints.push_back(i * 100);
      floats.push_back(i*3.12);
    }
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
  }