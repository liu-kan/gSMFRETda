#include <gtest/gtest.h>
#include <tuple>
#include <iostream>
#include <vector>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include "cuda_tools.hpp"

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
#include "binom.cuh"
#include "gen_rand.cuh"

std::vector<int> ints;
std::vector<float> floats;

class GenRandTest : public ::testing::TestWithParam<std::tuple<int, float>>  {
protected:
    void SetUp() override {
        // pdamc.set_gpuid();
        // pdamc.init_randstate(numBurst,0)
        a=1;
    }

    // void TearDown() override {}

    // mc pdamc(0,1,0,0,0);
    float a;
};

TEST_P(GenRandTest, Basic) {
    std::cout << "int: "        << std::get<0>(GetParam())
              << "  float: \"" << std::get<1>(GetParam())
              << "\"\n";
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