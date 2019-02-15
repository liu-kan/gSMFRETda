#include <iostream>
#include <cstdint>
#include "cxxopts.hpp"
#include "loadHdf5.hpp"

#include "eigenhelper.hpp"
#include "mc.hpp"
#include <assert.h>
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
      ("i,input", "Input HDF5", cxxopts::value<string>())
    ;

    auto result = options.parse(argc, argv);    
    if (result.count("input")<1)
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
    vector<float> args={0.2,0.3,0.4,1,2,3,4,5,6,0.9,0.9,0.9};
    for (int i=0;i<args.size();i++)cout<< args[i];cout<<endl;
    pdamc.set_params(3,args);
    cout<<pdamc.eargs<<endl;
    cout<<pdamc.vargs<<endl;
    return 0;   
}