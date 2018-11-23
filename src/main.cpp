#include <iostream>
#include <cstdint>
#include "cxxopts.hpp"
#include "loadHdf5.hpp"

cxxopts::ParseResult
parse(int argc, char* argv[])
{
  try
  {
    cxxopts::Options options(argv[0], " - Compute smFRET PDA by GPU");

    options
      .allow_unrecognised_options()
      .add_options()
      ("i,input", "Input HDF5", cxxopts::value<std::string>())
    ;

    auto result = options.parse(argc, argv);    
    if (result.count("input")<1)
    {
        std::cout << "Input file of HDF5 arg -i/--input must be assigned " << std::endl;
        exit(1);
    }
    return result;

  } catch (const cxxopts::OptionException& e)
  {
    std::cout << "error parsing options: " << e.what() << std::endl;
    exit(1);
  }
}

int main(int argc, char* argv[])
{
    auto result = parse(argc, argv);
    // auto arguments = result.arguments();
    // std::cout << "Saw " << arguments[0].key() << " arguments" << std::endl;
    std::string H5FILE_NAME=result["input"].as<std::string>();    
    std::vector<uint32_t> istart;std::vector<uint32_t> istop;    
    std::vector<int64_t> stop;std::vector<int64_t> start;
    std::vector<int64_t> times_ms;
    std::vector<unsigned char> mask_ad;std::vector<unsigned char> mask_dd;
    std::vector<float> T_burst_duration;std::vector<float> SgDivSr;
    float clk_p,bg_ad_rate,bg_dd_rate;
    loadhdf5(H5FILE_NAME,start,stop,istart,istop,times_ms,mask_ad,mask_dd,T_burst_duration,SgDivSr,clk_p,bg_ad_rate,bg_dd_rate);
    // std::cout << "mask_ad len " << mask_ad.size() << " mask_ad last " << mask_ad[34367292]<< std::endl;
    return 0;   
}