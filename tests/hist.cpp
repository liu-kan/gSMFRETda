#include <algorithm>           // std::for_each
#include <boost/format.hpp>    // only needed for printing
#include <boost/histogram.hpp> // make_histogram, regular, weight, indexed
#include <cassert>             // assert (used to test this example for correctness)
#include <functional>          // std::ref
#include <iostream>            // std::cout, std::flush

#include "hist.hpp"
using namespace boost::histogram;
auto mkhist(float* SgDivSr,int size, int binnum,float lv,float uv){
  
  auto h = make_histogram( axis::regular<>(binnum, lv, uv,"x") );
  for (int i = 0; i < size; i++) {
    float it = SgDivSr[i];
    h(it);
  }
  return h;
}
void getoss(float *rawp,int size, int n, std::ostringstream &os,float *p,float low,float up){
    auto h=mkhist(rawp, size,n,low,up);
    int i = 0,sum=0;
    for (auto&& x : indexed(h, coverage::inner)) {
        // os << boost::format("bin %2i [%4.1f, %4.1f): %i\n") % x.index() % x.bin().lower() % x.bin().upper() % *x;
        sum += x;
        p[i++] = x;
    }    
    for (; i > 0;) {
        p[--i] /= sum;
        // os << boost::format("p[%i]=%f ") % i % p[i];
    }
    os << boost::format("\n");
    
}

auto mkhist_i(int* SgDivSr, int size, int binnum, float lv, float uv) {
    auto h = make_histogram(axis::regular<>(binnum, lv, uv, "x"));
    for (int i = 0; i < size; i++) {
        int it = SgDivSr[i];
        h(it);
    }
    return h;
}

void getoss_i(int* rawp, int size, int n, std::ostringstream& os, float* p,float low,float up){
    auto h=mkhist_i(rawp, size, n, 0, n);
    if (up>low)        
        h = mkhist_i(rawp, size, n, low, up);
    int i = 0, sum = 0;
    for (auto&& x : indexed(h, coverage::inner)) {
        // os << boost::format("bin %2i [%4.1f, %4.1f): %i\n") % x.index() % x.bin().lower() % x.bin().upper() % *x;
        sum += x;
        p[i++] = x;
    }
    for (; i > 0;) {
        p[--i] /= sum;
        // os << boost::format("p[%i]=%f ") % i % p[i];
    }
    os << boost::format("\n");

}
