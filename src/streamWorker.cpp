#include "streamWorker.hpp"
#include <assert.h>
#include <nanomsg/nn.h>
#include <nanomsg/reqrep.h>
#include <nanomsg/tcp.h>
#include "args.pb.h"
#include "tools.hpp"
#include <iostream>
#include <chrono>
using namespace std::chrono_literals;
#include <boost/histogram.hpp>
#include <vector>
using namespace boost::histogram;
streamWorker::streamWorker(mc* _pdamc,string* _url,std::vector<float>* _d,int _fretHistNum,
  std::mutex *m, std::condition_variable *cv,int *_dataready,int *_sn,
  std::vector<float> *_params, int *_ga_start, int *_ga_stop,int *_N,unsigned char debugl=0){    
    pdamc=_pdamc;    
    url=_url;       
    SgDivSr=_d;
    fretHistNum=_fretHistNum;
    _m=m;
    _cv=cv;    
    dataready=_dataready;
    s_n=_sn;
    params=_params;
    ga_start=_ga_start;
    ga_stop=_ga_stop;
    N=_N;
    debug=debugl;
}
// template <typename Tag, typename Storage>
auto streamWorker::mkhist(std::vector<float>* SgDivSr,int binnum,float lv,float uv){
    // auto h = make_s(static_tag(), std::vector<float>(), reg(binnum, lv, uv));
    auto h = make_histogram( axis::regular<>(binnum, lv, uv) );
    for (auto it = SgDivSr->begin(), end = SgDivSr->end(); it != end;) 
        h(*it++);
    // auto h = make_histogram(
    //   axis::regular<>(binnum, 0.0, 1.0, "x")
    // );    
    // std::for_each(SgDivSr->begin(), SgDivSr->end(), std::ref(h));
    return h;
}
void streamWorker::run(int sid,int sz_burst){  
    // AtomicWriter(debug,debugLevel::net) <<"net th#"<<sid<<" start connecting "<<url->c_str()<<"\n";  
    int sock;    //local
    // thread_local int s_n;
    unsigned long keepalivecount=0;
    int ps_n;
    // AtomicWriter(debug,debugLevel::net) <<"net th#"<<sid<<" start creating sock \n";
    sock = nn_socket (AF_SP, NN_REQ);
    //TODO nn_setsockopt
    // AtomicWriter(debug,debugLevel::net) <<"net th#"<<sid<<" sock created:"<<sock<<"\n";
    assert (sock >= 0);
    int nneo=nn_connect(sock, url->c_str());
    // AtomicWriter(debug,debugLevel::net) <<"net th#"<<sid<<" conneting:"<<nneo<<"\n";
    assert (nneo>= 0);
    // AtomicWriter(debug,debugLevel::net) <<"net th#"<<sid<<" conneted\n";
    std::string gpuNodeId;
    genuid(&gpuNodeId,pdamc->devid, sid,pdamc->gpuuuid);
    // AtomicWriter(debug,debugLevel::net) <<"net th#"<<sid<<" genuid "<<gpuNodeId.c_str()<<" gotten\n";
    // int countcalc=0;
    auto fretHist=mkhist(SgDivSr,fretHistNum,0,1);
    // AtomicWriter(debug,debugLevel::cpu) <<"frethist done\n";
    bool ending=false;
    int32_t params_idx;
    do {        
      // AtomicWriter(debug,debugLevel::net) <<"net th# "<<sid<<" try lock\n";    
      std::unique_lock<std::mutex> lck(_m[sid],std::defer_lock);

      if(!lck.try_lock()){            
            continue;
      }
      // AtomicWriter(debug,debugLevel::net) <<"net th#"<<sid<<" locked\n";   
      char *rbuf = NULL;
      int bytes;
      if(dataready[sid]==0){
          gSMFRETda::pb::p_cap cap;
          cap.set_cap(sz_burst);
          string scap;
          cap.SerializeToString(&scap);
          scap="c"+scap;
          nn_send (sock, scap.c_str(), scap.length(), 0);
          bytes = nn_recv (sock, &rbuf, NN_MSG, 0);  
          gSMFRETda::pb::p_n sn;
          sn.ParseFromArray(rbuf,bytes);
          nn_freemsg (rbuf);
          rbuf = NULL;
          s_n[sid]=sn.s_n();
          dataready[sid]=1;

          lck.unlock(); 
          continue;
      }
      else if (dataready[sid]==1){
        gSMFRETda::pb::p_str gpuidStr;
        gpuidStr.set_str(gpuNodeId);
        string sgpuid;
        gpuidStr.SerializeToString(&sgpuid);
        sgpuid="p"+sgpuid;
        nn_send (sock, sgpuid.c_str(), sgpuid.length(), 0);
        AtomicWriter(debug,debugLevel::cpu) << "p"+gpuNodeId <<'\n';
        
        gSMFRETda::pb::p_ga ga;
        bytes = nn_recv (sock, &rbuf, NN_MSG, 0);  
        ga.ParseFromArray(rbuf,bytes); 
        int pidx=ga.idx();
        if (pidx<0){
          lck.unlock();
          continue;
        }
        params_idx=pidx;
        ps_n=s_n[sid]*(s_n[sid]+1);
        params[sid].resize(ps_n);          
        AtomicWriter(debug,debugLevel::net) << "params_idx recv: "<<params_idx <<'\n';
        for(int pi=0;pi<ps_n;pi++)
          params[sid][pi]=ga.params(pi);
        nn_freemsg (rbuf);
        rbuf=NULL;
        ga_start[sid]=ga.start(); ga_stop[sid]=ga.stop();
        dataready[sid]=2;
      
        lck.unlock();
        _cv[sid].notify_one(); 
        continue;
      }      
      // bool calcR2=false;
      // while(!calcR2){
      //     if(!lck.try_lock()){
      //       std::this_thread::sleep_for(200ms);     
      //       continue;
      //     }
        if(!_cv[sid].wait_for(lck,500ms,[this,sid]{return dataready[sid]==4;})){
          AtomicWriter(debug,debugLevel::cpu) <<dataready[sid]<<" dataready !=4\n";
          if(++keepalivecount%60==0){
            gSMFRETda::pb::p_n pidx;
            pidx.set_s_n(params_idx);
            string spidx;
            pidx.SerializeToString(&spidx);
            spidx="k"+spidx;
            nn_send (sock, spidx.c_str(), spidx.length(), 0);
            nn_recv(sock, &rbuf,NN_MSG,0);
            nn_freemsg(rbuf);
            rbuf=NULL;
          }
          lck.unlock();
          continue;
        }
        else{
          AtomicWriter(debug,debugLevel::cpu) <<dataready[sid]<<" dataready ==4\n";
          // calcR2=true;
          vector<float> mcE(pdamc->hmcE[sid], 
            pdamc->hmcE[sid] + N[sid]*pdamc->reSampleTimes);//
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
          for(ihist=1;ihist<fretHistNum;ihist++){
            if(vOEHist[ihist]>0)
              chisqr+=pow((float(vOEHist[ihist]-vMcHist[ihist]/pdamc->reSampleTimes)),2)
                /float(vOEHist[ihist]);
            else
              effN--;      
          }
          chisqr=chisqr/(effN-s_n[sid]*(s_n[sid]+1));
          // chisqr=(effN-s_n[sid]*(s_n[sid]+1));
          gSMFRETda::pb::res chi2res;
          // chi2res.set_s_n(s_n[sid]);
          chi2res.set_idx(gpuNodeId);
          chi2res.set_ridx(params_idx);          
          // for (auto v : params[sid])
          //   chi2res.add_params(v);
          chi2res.set_e(chisqr);
          string sres;
          chi2res.SerializeToString(&sres);          
          nn_send (sock, ("r"+sres).c_str(), sres.length()+1, 0);
          // bytes = nn_send (sock, sres.c_str(), sres.length(), 0);           
          gSMFRETda::pb::p_sid gas; 
          AtomicWriter(debug,debugLevel::net) << chisqr<<"\t params_idx send: "<<params_idx <<'\n'; 
          bytes = nn_recv (sock, &rbuf, NN_MSG, 0);           
          gas.ParseFromArray(rbuf,bytes); 
          AtomicWriter(debug,debugLevel::net) << params_idx<<" p_sid recv: "<<gas.sid() <<'\n'; 
          if(gas.sid()==-1)
            ending=true;
          nn_freemsg (rbuf);
          rbuf=NULL;          
          dataready[sid]=0;          
          lck.unlock();
          // countcalc++;
        }
      // }
    }while(!ending);
    // notice gpu sid ended
    pdamc->workerNum--;
    nn_close(sock);

}