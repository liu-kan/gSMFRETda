#ifndef loadhdf5_HPP_INCLUDED
#define loadhdf5_HPP_INCLUDED

#include <cstdint>
#include <vector>
// using namespace HighFive;
using namespace std;
bool loadhdf5(string H5FILE_NAME, vector<int64_t>& start,vector<int64_t>& stop,
    vector<uint32_t>& istart,vector<uint32_t>& istop,
    vector<int64_t>& times_ms,
    vector<unsigned char>& mask_ad,vector<unsigned char>& mask_dd,
    vector<float>& T_burst_duration,vector<float>& SgDivSr,
    float& clk_p,float& bg_ad_rate,float& bg_dd_rate);

#endif