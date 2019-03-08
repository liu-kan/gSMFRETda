#include <iostream>
#include <cstdint>
#include "cxxopts.hpp"
#include "loadHdf5.hpp"

#include "eigenhelper.hpp"
#include "mc.hpp"
#include <assert.h>
#include <nanomsg/nn.h>
#include <nanomsg/reqrep.h>
#include <nanomsg/tcp.h>
#include "protobuf/args.pb.h"

using namespace std;
cxxopts::ParseResult
parse(int argc, char* argv[])
{
  try
  {
    cxxopts::Options options(argv[0], " - Compute smFRET PDA by GPU");
    options
      .allow_unrecognised_options()
      .add_options()
      ("u,url", "params server url tcp://ip:port", cxxopts::value<std::string>()->default_value("tcp://127.0.0.1:7777"))
      ("i,input", "Input HDF5", cxxopts::value<string>())
      ("h,help", "Print help")
    ;
    auto result = options.parse(argc, argv);    
    if (result.count("input")<1)
    {
        cout << "Input file of HDF5 arg -i/--input must be assigned " << endl;
        exit(1);
    }
		else if (result.count("help"))
		{
			std::cout << options.help({""}) << std::endl;
			exit(0);
		}
    return result;
  } catch (const cxxopts::OptionException& e)
  {
    cout << "error parsing options: " << e.what() << endl;
    exit(1);
  }
}

int main(int argc, char* argv[])
{
    auto result = parse(argc, argv);
    // auto arguments = result.arguments();
    // cout << "Saw " << arguments[0].key() << " arguments" << endl;
    string H5FILE_NAME=result["input"].as<string>();    
    string url=result["url"].as<string>();    
    vector<uint32_t> istart;vector<uint32_t> istop;    
    vector<int64_t> stop;vector<int64_t> start;
    vector<int64_t> times_ms;
    vector<unsigned char> mask_ad;vector<unsigned char> mask_dd;
    vector<float> T_burst_duration;vector<float> SgDivSr;
    float clk_p,bg_ad_rate,bg_dd_rate;
    loadhdf5(H5FILE_NAME,start,stop,istart,istop,times_ms,mask_ad,mask_dd,T_burst_duration,
        SgDivSr,clk_p,bg_ad_rate,bg_dd_rate);
    // cout << "mask_ad len " << mask_ad.size() << " mask_ad last " << mask_ad[34367292]<< endl;
    assert (mask_ad.size() == times_ms.size());
    assert (T_burst_duration.size() == SgDivSr.size());
    assert (T_burst_duration.size() == istart.size());
    cout<<T_burst_duration.size()<<endl;    
    mc pdamc(0);
    pdamc.init_data_gpu(start,stop,istart,istop,times_ms,mask_ad,mask_dd,T_burst_duration,
         SgDivSr,clk_p,bg_ad_rate,bg_dd_rate);

    int sock = nn_socket (AF_SP, NN_REQ);
    assert (sock >= 0);
    assert (nn_connect(sock, url.c_str()) >= 0);
    while (1){
      // s1.send("connected")
      char *rbuf = NULL;
      // nn_recv (sock, &rbuf, NN_MSG, 0);  
      // printf("%s\n",rbuf);
      // nn_freemsg (rbuf);
      
      gSMFRETda::pb::p_cap cap;
      cap.set_cap(1024);
      int size = cap.ByteSize(); 
      void *sbuf = malloc(size);
      cap.SerializeToArray(sbuf,size);
      int bytes = nn_send (sock, sbuf, size, 0);
      free(sbuf);

      rbuf = NULL;
      bytes = nn_recv (sock, &rbuf, NN_MSG, 0);  
      // printf("%s\n",rbuf);
      
      gSMFRETda::pb::p_n sn;
      sn.ParseFromArray(rbuf,bytes);
      nn_freemsg (rbuf);
      printf("%d\n",sn.s_n());
      nn_send (sock, "ok", 2, 0);
      
      rbuf = NULL;
      gSMFRETda::pb::p_ga ga;
      bytes = nn_recv (sock, &rbuf, NN_MSG, 0);  
      ga.ParseFromArray(rbuf,bytes);
      nn_freemsg (rbuf);
      printf("%f\n",ga.params(3));
      // vector<float> fData={1,3,5,8,7,9};
      // *ga.mutable_params() = {fData.begin(), fData.end()};
      // size = ga.ByteSize(); 
      // buffer = malloc(size);
      // ga.SerializeToArray(buffer,size);
      // bytes = nn_send (sock, buffer, size, 0);
      // free(buffer);
    }
        
    // vector<float> args={0.2,0.3,0.4,1,1,1,1,1,1,0.9,0.9,0.9};
    // for (int i=0;i<args.size();i++)cout<< args[i];cout<<endl;
    // pdamc.set_nstates(3);
    // pdamc.set_params(args);
    // pdamc.run_kernel(0,1570);
    // cout<<pdamc.eargs<<endl;
    // cout<<pdamc.vargs<<endl;
    return 0;   
}