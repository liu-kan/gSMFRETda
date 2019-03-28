#include <iostream>
#include <cstdint>
#include "cxxopts.hpp"
#include "loadHdf5.hpp"
#include <thread> 
#include "eigenhelper.hpp"
#include "mc.hpp"
#include "streamWorker.hpp"
#include <vector> 

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
      ("g,gpuid", "The index of the GPU will be used", cxxopts::value<string>()->default_value("0"))
      ("s,snum", "Stream Number", cxxopts::value<string>()->default_value("16"))
      ("d,debug", "debug the gpu kernel 0 or 1", cxxopts::value<string>()->default_value("0"))
      ("h,help", "Print help")      
    ;
    auto result = options.parse(argc, argv);    
		if (result.count("help"))
		{
			std::cout << options.help({""}) << std::endl;
			exit(0);
		}
    else if (result.count("input")<1)
    {
        cout << "Input file of HDF5 arg -i/--input must be assigned " << endl;
        exit(1);
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
    int streamNum=std::stoi(result["snum"].as<string>());    
    int debugmc=std::stoi(result["debug"].as<string>());
    bool debugbool=(debugmc==0?false:true);
    int gpuid=std::stoi(result["gpuid"].as<string>());    
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
    mc pdamc(gpuid,streamNum,debugbool);
    pdamc.init_data_gpu(start,stop,istart,istop,times_ms,mask_ad,mask_dd,T_burst_duration,
         SgDivSr,clk_p,bg_ad_rate,bg_dd_rate);        
    // vector<float> args={0.2,0.3,0.4,1,1,1,1,1,1,0.9,0.9,0.9};
    // for (int i=0;i<args.size();i++)cout<< args[i];cout<<endl;
    // pdamc.set_nstates(3);
    // pdamc.set_params(args);
    // pdamc.run_kernel(0,1570);
    // cout<<pdamc.eargs<<endl;
    // cout<<pdamc.vargs<<endl;
    streamWorker worker(&pdamc,&url);
    std::vector<std::thread> threads;
    for(int i=0;i<streamNum;i++){
      threads.push_back(std::thread(&streamWorker::run,&worker,i));
    }
    for (auto& th : threads) th.join();
    return 0;   
}