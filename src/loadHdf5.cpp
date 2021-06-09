#include <iostream>
#include <string>
#include <vector>

#include <highfive/H5File.hpp>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
// #include <numeric>
#include "loadHdf5.hpp"
#include <boost/filesystem.hpp>
using namespace boost::filesystem;
#include "eigenhelper.hpp"


std::uintmax_t loadhdf5(std::string H5FILE_NAME, std::vector<int64_t>& start,std::vector<int64_t>& stop,
    std::vector<int64_t>& istart,std::vector<int64_t>& istop,
    std::vector<int64_t>& times_ms,
    std::vector<unsigned char>& mask_ad,std::vector<unsigned char>& mask_dd,
    std::vector<float>& T_burst_duration,std::vector<float>& SgDivSr,
    float& clk_p,float& bg_ad_rate,float& bg_dd_rate)
{
    std::uintmax_t fsize=0;
    path fpath(H5FILE_NAME);
    // try {
        fsize=file_size(fpath);
    // } catch(const filesystem_error& e) {
    //     std::cout << e.what() << '\n';
    //     return 0;
    // }  
    using namespace HighFive;    
    // std::vector<unsigned char> mask_ad_i;std::vector<unsigned char> mask_dd_i;
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
        // dataset.read(mask_ad_i);        
        dataset.read(mask_ad);      
        DATASET_NAME="mask_dd";
        dataset = file.getDataSet(DATASET_NAME);        
        // dataset.read(mask_dd_i);                
        dataset.read(mask_dd);                
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
        fsize=0;
    }    
    // // int64_t size=mask_ad_i.size();
    // // for (int i=0;i<size;i++){
    // //     mask_dd.push_back(static_cast<bool>(mask_dd_i[i]));
    // //     mask_ad.push_back(static_cast<bool>(mask_ad_i[i]));
    // // }
    // // std::vector<unsigned char> bytebits;
    // // std::vector<bool> bits;
    // // int x=34367295,y=34367397;
    // fillbits<unsigned char>(mask_ad,mask_ad_i);
    // fillbits<unsigned char>(mask_dd,mask_dd_i);
    // // std::cout<<"getbits "<<getbits<bool>(bits,bytebits,x,y)<<std::endl;
    // // std::cout<<"fillbits "<<size<<" bytesize "<<bytebits.size()<<std::endl;
    // // // std::cout<<"mask_ad_i[304344] "<<static_cast<bool>(mask_ad_i[304340])<<std::endl;
    // // bool b1;int ii=0;
    // // for (auto i:bits){
    // //     std::cout<<"bytebits["<<x+ii<<"] "<<static_cast<bool>(i)<<std::endl;        
    // //     ii++;
    // // }
    return fsize; // successfully terminated
}

void burst_data(vector<int64_t>& istart,vector<int64_t>& istop,vector<int64_t>& times_ms, vector<unsigned char>& mask_ad,vector<unsigned char>& mask_dd, 
                std::vector<int>& phCount,int64_t **burst_ad, int64_t **burst_dd)
{
    size_t sz_burst=istart.size();
    if (phCount.size()!=sz_burst)
        phCount.resize(sz_burst);
    for(int idx=0;idx<sz_burst;idx++){    
        phCount[idx]=istop[idx]-istart[idx]+1; 
    }
    // uint64_t count=std::accumulate(phCount.begin(), phCount.end(), 0);
    // assert (count == times_ms.size());
    // std::cout<<"count: "<<count<<" times_ms.size: "<<times_ms.size()<<std::endl;
    *burst_ad=new int64_t[times_ms.size()];
    *burst_dd=new int64_t[times_ms.size()];
    // uint64_t offset=0;
    for(int idx=0;idx<sz_burst;idx++){
        arrUcharMapper mask_adA(mask_ad.data()+istart[idx],phCount[idx]); 
        arrUcharMapper mask_ddA(mask_dd.data()+istart[idx],phCount[idx]);
        arrI64Mapper times_msA(times_ms.data()+istart[idx],phCount[idx]);
        arrI64 eburst_ad=mask_adA.cast<int64_t>()*times_msA;     
        arrI64 eburst_dd=mask_ddA.cast<int64_t>()*times_msA; 
        memcpy((*burst_ad)+istart[idx],eburst_ad.data(),sizeof(int64_t)*phCount[idx]);
        memcpy((*burst_dd)+istart[idx],eburst_dd.data(),sizeof(int64_t)*phCount[idx]);
        // offset+=phCount[idx];
    }
}

bool savehdf5(string FILE_NAME, string DATASET_NAME, vector<retype>& r){   
    using namespace HighFive;
    try {
        // Create a new file using the default property lists.
        HighFive::File file(FILE_NAME, File::ReadWrite|File::Overwrite);
        // Create the dataset
        DataSet dataset =
            file.createDataSet<retype>(DATASET_NAME, DataSpace::From(r));
        // write it
        dataset.write(r);
    } catch (Exception& err) {
        // catch and print any HDF5 error
        std::cerr << err.what() << std::endl;
        return false;
    }
    return true;

}