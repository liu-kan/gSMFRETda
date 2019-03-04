#ifndef eigenhelper_HPP_INCLUDED
#define eigenhelper_HPP_INCLUDED
#include <Eigen/Core>
#include <Eigen/Eigen>

#include <vector>
using namespace std;
using namespace Eigen;

typedef Eigen::Array<int64_t,1,Eigen::Dynamic,Eigen::RowMajor> arrI64;
typedef Eigen::Array<int,1,Eigen::Dynamic,Eigen::RowMajor> arrI;
typedef Eigen::Map<arrI64> arrI64Mapper;
typedef Eigen::Array<unsigned char,1,Eigen::Dynamic,Eigen::RowMajor> arrUchar;
typedef Eigen::Map<arrUchar> arrUcharMapper;
typedef Eigen::Map<RowVectorXf> vecFloatMapper;
typedef Eigen::Map<MatrixXf> matXfMapper;

bool genMatK(Eigen::MatrixXf** matK,int n, RowVectorXf& args);
bool genMatP(Eigen::MatrixXf** matP,Eigen::MatrixXf* matK);
#endif