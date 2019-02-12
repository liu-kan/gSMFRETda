#ifndef eigenhelper_HPP_INCLUDED
#define eigenhelper_HPP_INCLUDED

#include <Eigen/Core>

typedef Eigen::Map<Eigen::Array<int,1,Eigen::Dynamic,Eigen::RowMajor>> arrIntMapper;

typedef Eigen::Array<unsigned char,1,Eigen::Dynamic,Eigen::RowMajor> arrUchar;
typedef Eigen::Map<arrUchar> arrUcharMapper;

#endif