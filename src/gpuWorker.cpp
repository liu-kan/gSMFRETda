#include "gpuWorker.hpp"
#include <assert.h>
#include <nanomsg/nn.h>
#include <nanomsg/reqrep.h>
#include <nanomsg/pair.h>
#include <nanomsg/tcp.h>
#include "protobuf/args.pb.h"
#include "tools.hpp"
#include <iostream>
#include <chrono>
using namespace std::chrono_literals;

gpuWorker::gpuWorker(mc* _pdamc,int _streamNum, std::vector<float>* _d,int _fretHistNum,
        std::mutex *m, std::condition_variable *cv){
    pdamc=_pdamc;
    _m=m;
    _cv=cv;
    streamNum=_streamNum;
    pdamc->set_gpuid();       
    SgDivSr=_d;
    fretHistNum=_fretHistNum;
}
// template <typename Tag, typename Storage>
auto gpuWorker::mkhist(std::vector<float>* SgDivSr,int binnum,float lv,float uv){
    auto h = make_s(static_tag(), std::vector<float>(), reg(binnum, lv, uv));
    for (auto it = SgDivSr->begin(), end = SgDivSr->end(); it != end;) 
        h(*it++);
    // auto h = make_histogram(
    //   axis::regular<>(binnum, 0.0, 1.0, "x")
    // );    
    // std::for_each(SgDivSr->begin(), SgDivSr->end(), std::ref(h));
    return h;
}
void gpuWorker::run(int sz_burst){
    auto fretHist=mkhist(SgDivSr,fretHistNum,0,1);
    do {            
      for(int sid=0;sid<streamNum;sid++){
        std::unique_lock<std::mutex> lk(_m[sid],std::defer_lock);
        if(!lck.try_lock_for(500ms))
          continue;
        // wait(cv[sidx],dataready[sidx]==1 or 2)
        if(!cv.wait_for(lck,500ms,[]{return (dataready[sidx]==1|| 
            dataready[sidx]==2);})){
          lck.unlock();
          continue;
        }
        pdamc->set_nstates(s_n[sid],sid);
        pdamc->set_params(s_n[sid],sid,params[sid]);
        int N=pdamc->setBurstBd(ga_start[sid],ga_stop[sid], sid);
        pdamc->run_kernel(N,sid);
        dataready[sidx]=3;
        if(pdamc->streamQuery(sid)){
          dataready[sidx]=4;
          lck.unlock();
          cv[sid].notify_one();
        }
        else{
          lck.unlock();
          continue;
        }
        vector<float> mcE(pdamc->hmcE[sid], 
          pdamc->hmcE[sid] + N);//*pdamc->reSampleTimes
        auto mcHist=mkhist(&mcE,fretHistNum,0,1);
        vector<float> vMcHist(fretHistNum);
        vector<float> vOEHist(fretHistNum);
        int ihist=0;
        for (auto x : indexed(fretHist))
          vOEHist[ihist++]=*x;
        ihist=0;
        for (auto x : indexed(mcHist))
          vMcHist[ihist++]=*x;      
        int effN=fretHistNum;           
        float chisqr=0;
        for(ihist=0;ihist<fretHistNum;ihist++){
          if(vOEHist[ihist]>0)
            chisqr+=pow((float(vOEHist[ihist]-vMcHist[ihist])),2)
              /float(vOEHist[ihist]);
          else
            effN--;      
        }
        chisqr=chisqr/(effN-s_n*(s_n+1));
        gSMFRETda::pb::res chi2res;
        chi2res.set_s_n(s_n);
        chi2res.set_idx(gpuNodeId);
        for (auto v : params)
          chi2res.add_params(v);
        chi2res.set_e(chisqr);
        string sres;
        chi2res.SerializeToString(&sres);
        sres="r"+sres;
        bytes = nn_send (sock, sres.c_str(), sres.length(), 0);
        rbuf = NULL;
        bytes = nn_recv (sock, &rbuf, NN_MSG, 0); 
        countcalc++;
      }
    }while(countcalc<3);
    // std::cout<<"end\n";
}