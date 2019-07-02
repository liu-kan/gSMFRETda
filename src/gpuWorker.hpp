#ifndef gpu_thr_hpp
#include <string>
#include <condition_variable>
#include <mutex>
#include <thread> 
#include "mc.hpp"
class gpuWorker {
    public:
        gpuWorker(mc* _pdamc,int streamNum,std::vector<float>* _d,int _fretHistNum,
            std::mutex *m, std::condition_variable *cv);
        void run(int sid);        
    private:
        mc* pdamc;
        int streamNum;
        std::vector<float> *SgDivSr;
        int fretHistNum;
        std::mutex *_m;
        std::condition_variable *_cv;
        auto mkhist(std::vector<float>* SgDivSr,int binnum,float lv,float uv);
}
#endif