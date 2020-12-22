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
void getoss(float *rawp, int n, std::ostringstream &os){
    auto h=mkhist(rawp,n*30,n,0,1);
    for (auto&& x : indexed(h, coverage::inner)) {
    os << boost::format("bin %2i [%4.1f, %4.1f): %i\n")
            % x.index() % x.bin().lower() % x.bin().upper() % *x;
    }
}