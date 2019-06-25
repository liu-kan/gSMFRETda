#include "gpuWorker.hpp"
#include <assert.h>
#include <nanomsg/nn.h>
#include <nanomsg/reqrep.h>
#include <nanomsg/pair.h>
#include <nanomsg/tcp.h>
#include "protobuf/args.pb.h"
#include "tools.hpp"
#include <iostream>


netWorker::netWorker(string* _url){
    url=_url;
}
void netWorker::run(int tid,int sz_burst){
    int sock;
    int s_n;
    int ps_n;
    sock = nn_socket (AF_SP, NN_REQ);
    assert (sock >= 0);
    assert (nn_connect(sock, url->c_str()) >= 0);
    int gsock = nn_socket (AF_SP, NN_PAIR);
    assert (gsock >= 0);    
    string TIPC=gpuipc+std::to_string(tid)+".ipc";
    assert (nn_connect(gsock, TIPC.c_str()) >= 0);    
    s_n=0;
    ps_n=0;
    std::string gpuNodeId;
    genuid(&gpuNodeId);
    int countcalc=0;
    int sid=-1,gaidx=-1;
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
        string ss_n="c"+rbuf;
        bytes = nn_send (gsock, ss_n.c_str(), ss_n.length(), 0);
        nn_freemsg (rbuf);
        s_n=sn.s_n();
        // printf("%d\n",sn.s_n());
        rbuf = NULL;
        bytes = nn_recv (gsock, &rbuf, NN_MSG, 0);  
        // printf("%s\n",rbuf);      
        gSMFRETda::pb::p_sid psid;
        psid.ParseFromArray(rbuf,bytes);
        nn_freemsg (rbuf);
        sid=psid.sid();

        nn_send (sock, ("p"+gpuNodeId).c_str(), gpuNodeId.length()+1, 0);
        std::cout<< "p"+gpuNodeId <<std::endl;
        rbuf = NULL;
        gSMFRETda::pb::p_ga ga;
        bytes = nn_recv (sock, &rbuf, NN_MSG, 0);  
        ga.ParseFromArray(rbuf,bytes); 
        nn_freemsg (rbuf);
        gaidx=ga.idx();
        ga.set_idx(sid);
        string spg_a;
        ga.SerializeToString(&spg_a);
        spg_a="p"+spg_a;
        nn_send (gsock, spg_a.c_str(), spg_a.length(), 0);
        rbuf = NULL;      
        bytes = nn_recv (gsock, &rbuf, NN_MSG, 0);  
        nn_freemsg (rbuf);  
        
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
    }while(countcalc<3);
    // std::cout<<"end\n";
}