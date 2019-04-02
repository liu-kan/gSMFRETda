#include "streamWorker.hpp"
#include <assert.h>
#include <nanomsg/nn.h>
#include <nanomsg/reqrep.h>
#include <nanomsg/tcp.h>
#include "protobuf/args.pb.h"
#include "tools.hpp"
#include <iostream>

using namespace boost::histogram;
streamWorker::streamWorker(mc* _pdamc,string* _url,std::vector<float>* _d,
  int _fretHistNum){
    pdamc=_pdamc;
    url=_url;       
    SgDivSr=_d;
    fretHistNum=_fretHistNum;
    
}
// template <typename Tag, typename Storage>
auto streamWorker::mkhist(std::vector<float>* SgDivSr,int binnum,float lv,float uv){
    auto h = make_s(static_tag(), std::vector<float>(), reg(binnum, lv, uv));
    for (auto it = SgDivSr->begin(), end = SgDivSr->end(); it != end;) 
        h(*it++);
    // auto h = make_histogram(
    //   axis::regular<>(binnum, 0.0, 1.0, "x")
    // );    
    // std::for_each(SgDivSr->begin(), SgDivSr->end(), std::ref(h));
    return h;
}
void streamWorker::run(int sid,int sz_burst){
    int sock;
    int s_n;
    int ps_n;
    sock = nn_socket (AF_SP, NN_REQ);
    assert (sock >= 0);
    assert (nn_connect(sock, url->c_str()) >= 0);
    s_n=0;
    ps_n=0;
    std::string gpuNodeId;
    genuid(&gpuNodeId);
    
    auto fretHist=mkhist(SgDivSr,fretHistNum,0,1);
    do {            
      gSMFRETda::pb::p_cap cap;
      cap.set_cap(sz_burst);
      cap.set_idx(gpuNodeId);
      string scap;
      cap.SerializeToString(&scap);
      scap="c"+scap;
      int bytes = nn_send (sock, scap.c_str(), scap.length(), 0);
      // free(sbuf);

      char *rbuf = NULL;
      bytes = nn_recv (sock, &rbuf, NN_MSG, 0);  
      // printf("%s\n",rbuf);      
      gSMFRETda::pb::p_n sn;
      sn.ParseFromArray(rbuf,bytes);
      nn_freemsg (rbuf);
      s_n=sn.s_n();
      // printf("%d\n",sn.s_n());
      pdamc->set_nstates(s_n,sid);
      // std::string idxencoded = base64_encode(reinterpret_cast<const unsigned char*>(gpuNodeId.c_str()),
      //    gpuNodeId.length());
      nn_send (sock, ("p"+gpuNodeId).c_str(), gpuNodeId.length()+1, 0);
      std::cout<< "p"+gpuNodeId <<endl;
      rbuf = NULL;
      gSMFRETda::pb::p_ga ga;
      bytes = nn_recv (sock, &rbuf, NN_MSG, 0);  
      ga.ParseFromArray(rbuf,bytes); 
      ps_n=s_n*(s_n+1);
      vector<float> params(ps_n);
      for(int pi=0;pi<ps_n;pi++)
        params[pi]=ga.params(pi);
      pdamc->set_params(s_n,sid,params);
      nn_freemsg (rbuf);
      int N=pdamc->setBurstBd(ga.start(),ga.stop(), sid);
      pdamc->run_kernel(N,sid);
      vector<float> mcE(pdamc->hmcE[sid], 
        pdamc->hmcE[sid] + N*pdamc->reSampleTimes);
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
    }while(0);
    // std::cout<<"end\n";
}