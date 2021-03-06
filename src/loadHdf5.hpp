#pragma once

#include <cstdint>
#include <vector>

using namespace std;

typedef float retype;
std::uintmax_t loadhdf5(string H5FILE_NAME, vector<int64_t>& start,vector<int64_t>& stop,
    vector<int64_t>& istart,vector<int64_t>& istop,
    vector<int64_t>& times_ms,
    vector<unsigned char>& mask_ad,vector<unsigned char>& mask_dd,
    vector<float>& T_burst_duration,vector<float>& SgDivSr,
    float& clk_p,float& bg_ad_rate,float& bg_dd_rate);
bool savehdf5(string FILE_NAME, string DATASET_NAME, vector<retype>& r);

void burst_data(vector<int64_t>& istart,vector<int64_t>& istop,vector<int64_t>& times_ms, vector<unsigned char>& mask_ad,vector<unsigned char>& mask_dd, 
                std::vector<int>& phCount,int64_t **burst_ad, int64_t **burst_dd);
