#include "ParamsTest.hpp"
#include <gtest/gtest.h>
#include <iostream>
#include <tuple>

std::vector<int> ints_n;

TEST_P(ParamsTest, paramsPassCheck) {
  int n = GetParam();
  paraMatCheck_test(n);
}

INSTANTIATE_TEST_CASE_P(statesNumAndParams, ParamsTest,
                        ::testing::Values(2, 3, 4, 5, 6));
