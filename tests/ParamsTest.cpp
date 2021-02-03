#include "ParamsTest.hpp"
#include "eigenhelper.hpp"
#include <random>
#include <vector>

void ParamsTest::SetUp() { printf("ParamsTest SetUp()\n"); }

void ParamsTest::TearDown() { printf("ParamsTest TearDown()\n"); }

void MatCheck::genRandMatk(int n,std::vector<float> &args){
    std::random_device
      rd; // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<float> dis(0.0, 1.0);
  for (int i = 0; i < n; i++) {
    args.push_back(dis(gen));
  }
  for (int i = 0; i < n * (n - 1); i++) {
    args.push_back(dis(gen) * 1000);
  }
  for (int i = 0; i < n; i++) {
    args.push_back(dis(gen) * 10);
  }
}

void ParamsTest::paraMatCheck_test(int n) {
  std::cout << "n: " << n << std::endl;
  std::vector<float> args;
  float *matK = new float[n * n];
  float* matP_i2j= new float[n * n];
  float *matP = new float[n];
  MatCheck matCheck;
  matCheck.genRandMatk( n,args);
  std::cout << "args: ";
  for (auto i_args : args)
    std::cout << i_args << ' ';
  std::cout << std::endl;
  std::string sep = "\n----------------------------------------\n";
  Eigen::IOFormat CleanFmt(n, 0, ", ", "\n", "[", "]");

  vecFloatMapper evargs(args.data(), n * n + n);
  // RowVectorXf eargs=evargs(seqN(0,n));
  RowVectorXf eargs = evargs.block(0, 0, 1, n);
  float *peargs = eargs.data();
  // RowVectorXf kargs=evargs(seqN(n,n*n-n));
  RowVectorXf kargs = evargs.block(0, n, 1, n * n - n);
  // RowVectorXf vargs=evargs(seqN(n*n,n));
  RowVectorXf vargs = evargs.block(0, n * n, 1, n);
  float *pvargs = vargs.data();
  int sa = 1;

  bool r = genMatK(matK, n, kargs);
  genP_i2j(matK, matP_i2j, n);
  r = r && genMatP(matP, matK, n);
  matXfMapper matKmp(matK, n, n);
  matXfMapper matP_i2jmp(matP_i2j, n, n);
  for (int i = 0; i < n; i++) {
    ASSERT_EQ( matKmp(0,i), *(matK + i * n));
  }
  for (int j = 0; j < n; j++) {
    ASSERT_LT(abs(matP_i2jmp.col(j).sum()-1),0.00001);
    if (n == 3) {
        int i = (j + 1) % n;
        int ii = (j + 2) % n;
        ASSERT_LT(abs(matP_i2jmp(i, j) / matP_i2jmp(ii, j) - matKmp(i, j) / matKmp(ii, j)), 0.0001);
    }
  }
  std::cout << matKmp.format(CleanFmt) << sep;
  delete[] matK;
  delete[] matP_i2j;
  delete[] matP;
}