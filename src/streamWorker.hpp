#ifndef stream_thr_hpp
#define stream_thr_hpp

#include <Poco/Runnable.h>
#include <Poco/Thread.h>
#include "mc.hpp"
#include "protobuf/args.pb.h"

class streamWorker : public Runnable
{
public:
  streamWorker(mc* pdamc,string* url);
  virtual void run();

private:
  mc* pdamc;
  string* url;
  int sock;
  int s_n;
  int ps_n;
};

#endif