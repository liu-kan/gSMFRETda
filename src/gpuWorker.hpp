#pragma once

#include <string>
#include <condition_variable>
#include <mutex>
#include <thread> 
#include "mc.hpp"

class gpuWorker {
    public:
        gpuWorker(mc* _pdamc,int streamNum,std::vector<float>* _d,int _fretHistNum,
            std::mutex *m, std::condition_variable *cv,int *dataready,int *sn,
    std::vector<float> *params, int *ga_start, int *ga_stop,int *N);
        void run(int sid);        
    private:
        mc* pdamc;
        int streamNum;
        std::vector<float> *SgDivSr;
        int fretHistNum;
        std::mutex *_m;
        std::condition_variable *_cv;
        int *dataready;
        int *s_n;
        std::vector<float> *params;
        int *N;int *ga_start; int *ga_stop;
        // auto mkhist(std::vector<float>* SgDivSr,int binnum,float lv,float uv);
};
