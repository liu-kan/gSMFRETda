#pragma once

#include <sstream>             // std::ostringstream

void getoss(float *rawp, int size, int n, std::ostringstream &os, float* p,float low=0,float up=1);
void getoss_d(double *rawp, int size, int n, std::ostringstream &os, double* p,double low=0,double up=1);
void getoss_i(int* rawp, int size, int n, std::ostringstream& os, float* p,float low=0,float up=-1);