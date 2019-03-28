#include "streamWorker.hpp"
#include <assert.h>
#include <nanomsg/nn.h>
#include <nanomsg/reqrep.h>
#include <nanomsg/tcp.h>
#include "protobuf/args.pb.h"
#include "tools.hpp"
#include <iostream>

streamWorker::streamWorker(mc* _pdamc,string* _url){
    pdamc=_pdamc;
    url=_url;       
}
void streamWorker::run(int sid){
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
    do {            
      gSMFRETda::pb::p_cap cap;
      cap.set_cap(-1);
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
      // cout<< ga.idx()<<endl;
      // vector<float> fData={1,3,5,8,7,9};
      // *ga.mutable_params() = {fData.begin(), fData.end()};
      // size = ga.ByteSize(); 
      // buffer = malloc(size);
      // ga.SerializeToArray(buffer,size);
      // bytes = nn_send (sock, buffer, size, 0);
      // free(buffer);
    }while(0);

}