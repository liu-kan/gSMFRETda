#include <iostream>
#include <string>
#include <vector>

#include <highfive/H5File.hpp>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include "loadHdf5.hpp"

bool loadhdf5(std::string H5FILE_NAME, std::vector<int64_t>& start,std::vector<int64_t>& stop,
    std::vector<uint32_t>& istart,std::vector<uint32_t>& istop,
    std::vector<int64_t>& times_ms,
    std::vector<bool>& mask_ad,std::vector<bool>& mask_dd,
    std::vector<float>& T_burst_duration,std::vector<float>& SgDivSr,
    float& clk_p,float& bg_ad_rate,float& bg_dd_rate)
{
    using namespace HighFive;    
    std::vector<int8_t> mask_ad_i;std::vector<int8_t> mask_dd_i;
    try {
        File file(H5FILE_NAME, File::ReadOnly);
        std::string DATASET_NAME("/sub_bursts_l/start");
        DataSet dataset = file.getDataSet(DATASET_NAME);        
        dataset.read(start);
        DATASET_NAME="/sub_bursts_l/stop";
        dataset = file.getDataSet(DATASET_NAME);        
        dataset.read(stop);        
        DATASET_NAME="/sub_bursts_l/istart";
        dataset = file.getDataSet(DATASET_NAME);        
        dataset.read(istart);
        DATASET_NAME="/sub_bursts_l/istop";
        dataset = file.getDataSet(DATASET_NAME);        
        dataset.read(istop);        
        DATASET_NAME="times";
        dataset = file.getDataSet(DATASET_NAME);        
        dataset.read(times_ms);
        DATASET_NAME="mask_ad";
        dataset = file.getDataSet(DATASET_NAME);        
        dataset.read(mask_ad_i);        
        DATASET_NAME="mask_dd";
        dataset = file.getDataSet(DATASET_NAME);        
        dataset.read(mask_dd_i);                
        DATASET_NAME="T_burst_duration";
        dataset = file.getDataSet(DATASET_NAME);        
        dataset.read(T_burst_duration);          
        DATASET_NAME="SgDivSr";
        dataset = file.getDataSet(DATASET_NAME);        
        dataset.read(SgDivSr);                                
        DATASET_NAME="clk_p";
        dataset = file.getDataSet(DATASET_NAME);        
        dataset.read(clk_p);                                        
        DATASET_NAME="bg_ad_rate";
        dataset = file.getDataSet(DATASET_NAME);        
        dataset.read(bg_ad_rate);
        DATASET_NAME="bg_dd_rate";
        dataset = file.getDataSet(DATASET_NAME);        
        dataset.read(bg_dd_rate);
    } catch (Exception& err) {
        std::cerr << err.what() << std::endl;
        return false;
    }    
    int64_t size=mask_ad_i.size();
    for (int i=0;i<size;i++){
        mask_dd.push_back(static_cast<bool>(mask_dd_i[i]));
        mask_ad.push_back(static_cast<bool>(mask_ad_i[i]));
    }
    return true; // successfully terminated
}
