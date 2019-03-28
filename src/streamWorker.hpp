#ifndef stream_thr_hpp
#define stream_thr_hpp

#include "mc.hpp"
#include <string>

class streamWorker 
{
public:
  streamWorker(mc* pdamc,string* url);
  void run(int sid);

private:
  mc* pdamc;
  string* url;
};

#endif