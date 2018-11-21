#ifndef loadhdf5_HPP_INCLUDED
#define loadhdf5_HPP_INCLUDED

#include <cstdint>
#include <vector>
// using namespace HighFive;

bool loadhdf5(std::string H5FILE_NAME, std::vector<int64_t>& start,std::vector<int64_t>& stop,
    std::vector<uint32_t>& istart,std::vector<uint32_t>& istop,
    std::vector<int64_t>& times_ms,
    std::vector<bool>& mask_ad,std::vector<bool>& mask_dd,
    std::vector<float>& T_burst_duration,std::vector<float>& SgDivSr,
    float& clk_p,float& bg_ad_rate,float& bg_dd_rate);

#endif