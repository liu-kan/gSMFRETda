#pragma once
#include <string>
void genuid(std::string* id,int gid,int sid,char *gpuuid);
#include <thread>
#include <mutex>
#include <condition_variable>
#include <iostream>     // std::cout, std::ios
#include <sstream>      // std::ostringstream
enum debugLevel:unsigned char{
  cpu=1,
  gpu=1<<1,
  net=1<<2,
  info=1<<3,
  parameter=1<<4,
  kernel=1<<5
};

class AtomicWriter {
    std::ostringstream st;
    unsigned char debug=false;
 public:
    AtomicWriter(unsigned char used,debugLevel wanted){
      debug=wanted & used;
    }
    template <typename T> 
    AtomicWriter& operator<<(T const& t) {
      if(debug)
       st << t;
      return *this;
    }
    ~AtomicWriter() {
      if(debug){
       std::string s = st.str();
       std::cout << s;
      //  std::cerr << s;
       //fprintf(stderr,"%s", s.c_str());
       // write(2,s.c_str(),s.size());
      }
    }
};

