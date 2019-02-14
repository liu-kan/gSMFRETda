#ifndef eigenhelper_HPP_INCLUDED
#define eigenhelper_HPP_INCLUDED
#include <Eigen/Eigen>
#include <Eigen/Core>

#include <vector>
using namespace std;
using namespace Eigen;

typedef Eigen::Map<Eigen::Array<int,1,Eigen::Dynamic,Eigen::RowMajor>> arrIntMapper;

typedef Eigen::Array<unsigned char,1,Eigen::Dynamic,Eigen::RowMajor> arrUchar;
typedef Eigen::Map<arrUchar> arrUcharMapper;
typedef Eigen::Map<RowVectorXf> vecFloatMapper;

bool genMatK(Eigen::MatrixXf** matK,int n, RowVectorXf& args);
bool genMatP(Eigen::MatrixXf** matP,Eigen::MatrixXf* matK);
#endif