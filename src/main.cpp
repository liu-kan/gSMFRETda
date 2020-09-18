#include <iostream>
#include <cstdint>
#include "loadHdf5.hpp"
#include "eigenhelper.hpp"
#include "mc.hpp"
#include "streamWorker.hpp"
#include "gpuWorker.hpp"
#include <vector> 
#include <thread>
#include <mutex>
#include <condition_variable>
#include "args.pb.h"
#include <chrono>
using namespace std::chrono_literals;
using namespace std;
#include "tools.hpp"
#include "3rdparty/gengetopt/cmdline.h"

void share_var_init(int streamNum,std::mutex **_m, std::condition_variable **_cv,
  int **s_n, vector<float> **params, int **ga_start, int **ga_stop,
  int **dataready,int **N){
    *_m=new std::mutex[streamNum]();
    *_cv=new std::condition_variable[streamNum]();    
    *s_n=new int[streamNum](); 
    *params=new vector<float>[streamNum](); 
    *ga_start=new int[streamNum](); 
    *ga_stop=new int[streamNum](); 
    *N=new int[streamNum];
    *dataready=new int[streamNum]();
    std::fill_n(*dataready, streamNum, 0);
}

void share_var_free(int streamNum,std::mutex *_m, std::condition_variable *_cv,
  int *s_n, vector<float> *params, int *ga_start, int *ga_stop, 
  int *dataready,int *N){
    // delete[] _m ;
    // delete[] _cv;    
    delete[] s_n; 
    delete[] params ; 
    delete[] ga_start ; 
    delete[] ga_stop ; 
    delete[] N;
    delete[] dataready;
}

int main(int argc, char* argv[])
{
    // auto result = parse(argc, argv);
    gengetopt_args_info args_info;
    if (cmdline_parser (argc, argv, &args_info) != 0)
      exit(1) ;    
    if ( args_info.gpuinfo_flag ){
      showGPUsInfo();
      exit(1);
    }
    if(args_info.inputs_num<1 && !args_info.input_given ){
      std::cout<<"You need either appoint -i or add hdf5 filename in the end of cmdline!"<<std::endl;
      exit(1);
    }
    string H5FILE_NAME;
    if (args_info.input_given)
      H5FILE_NAME=args_info.input_arg ;    
    else{
      H5FILE_NAME=args_info.inputs[0];
    }
    std::cout<<"Using "<<H5FILE_NAME<<std::endl;
    string url=args_info.url_arg ;    
    int streamNum=args_info.snum_arg;    
    int fretHistNum=args_info.fret_hist_num_arg;    
    unsigned char debuglevel=0;
    if(args_info.debug_flag)
      debuglevel=~0;
    if(args_info.debugcpu_flag)
      debuglevel|=debugLevel::cpu;
    if(args_info.debuggpu_flag)
      debuglevel|=debugLevel::gpu;
    if(args_info.debugnet_flag)
      debuglevel|=debugLevel::net;            
    int gpuid=args_info.gpuid_arg;    
    vector<int64_t> istart;vector<int64_t> istop;    
    vector<int64_t> stop;vector<int64_t> start;
    vector<int64_t> times_ms;
    vector<unsigned char> mask_ad;vector<unsigned char> mask_dd;
    vector<float> T_burst_duration;vector<float> SgDivSr;
    float clk_p,bg_ad_rate,bg_dd_rate;
    std::uintmax_t hdf5size=loadhdf5(H5FILE_NAME,start,stop,istart,istop,times_ms,mask_ad,mask_dd,T_burst_duration,
        SgDivSr,clk_p,bg_ad_rate,bg_dd_rate);
    std::cout<<H5FILE_NAME<<" loaded."<<std::endl;
    assert (mask_ad.size() == times_ms.size());
    assert (T_burst_duration.size() == SgDivSr.size());
    assert (start.size() == istart.size());
    cout<<T_burst_duration.size()<<endl;
    std::vector<int> phCount(start.size());
    int64_t *burst_ad, *burst_dd;
    burst_data(istart,istop,times_ms,mask_ad,mask_dd, 
                phCount,&burst_ad, &burst_dd);
    mc pdamc(gpuid,streamNum,debuglevel,hdf5size,args_info.profiler_flag);
    streamNum=pdamc.streamNum;
    pdamc.init_data_gpu(istart,start,stop,
        phCount,times_ms.size(),burst_ad, burst_dd,
        T_burst_duration,SgDivSr,
        clk_p,bg_ad_rate,bg_dd_rate);
    std::mutex *_m;std::condition_variable *_cv;
    // std::vector<std::unique_ptr<std::mutex>> _m
    // std::vector<std::mutex> _ms(streamNum);
    // std::vector<std::condition_variable> _cvs(streamNum);
    int *s_n; vector<float> *params; int *ga_start; int *ga_stop; float *chisqr;
    int *dataready,*N;
    share_var_init(streamNum,&_m,&_cv,
      &s_n, &params, &ga_start, &ga_stop,&dataready,&N);
    // std::cout<<"dataready[2]:"<<dataready[2]<<std::endl;
    streamWorker worker(&pdamc,&url,&SgDivSr,fretHistNum,_m,_cv,
      dataready,s_n, params, ga_start, ga_stop,N,debuglevel);
    gpuWorker gpuworker(&pdamc,streamNum,&SgDivSr,fretHistNum,_m,_cv,
      dataready,s_n, params, ga_start, ga_stop,N,debuglevel);
    std::cout<<"workers created\n";
    std::vector<std::thread> threads;
    for(int i=0;i<streamNum;i++){
      threads.push_back(std::thread(&streamWorker::run,&worker,i,pdamc.sz_burst));
    }
    std::cout<<"net threads looped\n";
    std::this_thread::sleep_for(2s);
    std::thread thGpu(&gpuWorker::run,&gpuworker,pdamc.sz_burst);
    std::cout<<"gpu thread looped\n";
    for (auto& th : threads) {
      std::cout<<"Joining pid: "<<th.get_id()<<std::endl;
      th.join();
    }
    std::cout<<"net threads joined\n";    
    thGpu.join();
    std::cout<<"gpu thread joined\n";    
   	// for (std::thread & th : threads)
    // {
    //   if (th.joinable())
    //     th.join();
    // }
    // pdamc.set_gpuid();
    // worker.run(0,pdamc.sz_burst);
    delete[](burst_ad);
    delete[](burst_dd);    
    share_var_free(streamNum,_m,_cv,s_n, params, ga_start, ga_stop,dataready,N);
    return 0;   
}