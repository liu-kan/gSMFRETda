#ifndef stream_thr_hpp
#define stream_thr_hpp

#include "mc.hpp"
#include <string>
#include <condition_variable>
#include <mutex>
#include <thread> 
class streamWorker 
{
public:
  streamWorker(mc* pdamc,string* _url,std::vector<float> *d,int fretHistNum,
    std::mutex *m, std::condition_variable *cv);
  void run(int sid,int sz_burst);
  auto mkhist(std::vector<float>* SgDivSr,int binnum,float lv,float uv);
private:
  mc* pdamc;
  int streamNum;
  std::vector<float> *SgDivSr;
  int fretHistNum;
  string* url;
  std::mutex *_m;
  std::condition_variable *_cv;  
};

#endif