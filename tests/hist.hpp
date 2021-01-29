#pragma once

#include <sstream>             // std::ostringstream
auto mkhist(float* SgDivSr,int size, int binnum,float lv,float uv);
void getoss(float *rawp, int size, int n, std::ostringstream &os, float* p);
void getoss_i(int* rawp, int size, int n, std::ostringstream& os, float* p);