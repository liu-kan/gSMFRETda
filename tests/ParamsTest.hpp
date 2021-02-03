#include <gtest/gtest.h>
#include <vector>
class ParamsTest : public ::testing::TestWithParam<int> {
protected:
  void SetUp() override;
  void TearDown() override;
  void paraMatCheck_test(int n);
};

class MatCheck{
  public:
    void genRandMatk(int n, std::vector<float> &args);
};