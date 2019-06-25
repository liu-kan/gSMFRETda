#ifndef stream_thr_hpp
#define stream_thr_hpp

#include "mc.hpp"
#include <string>

class streamWorker 
{
public:
  streamWorker(mc* pdamc,int streamNum,std::vector<float> *d,int fretHistNum);
  void run(int sid,int sz_burst);
auto mkhist(std::vector<float>* SgDivSr,int binnum,float lv,float uv);
private:
  mc* pdamc;
  int streamNum;
  std::vector<float> *SgDivSr;
  int fretHistNum;
};

#endif