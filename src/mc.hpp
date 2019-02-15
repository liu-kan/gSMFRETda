#ifndef mc_HPP_INCLUDED
#define mc_HPP_INCLUDED
#include <vector>
#include <cstdint>
using namespace std;
#include "eigenhelper.hpp"
class mc
{
    protected:
        int devid;
        long sz_tag;
        unsigned char *g_mask_ad,*g_mask_dd;
        float *gchi2,*hchi2;        
        int sz_burst;
        int64_t *g_start,*g_stop,*g_times_ms;
        uint32_t *g_istart,*g_istop;   
        float *g_burst_duration,*g_SgDivSr,clk_p,bg_ad_rate,bg_dd_rate;
        MatrixXf *matK,*matP;
        int s_n;
    public:
        RowVectorXf eargs,vargs,kargs;
        void free_data_gpu();
        void run_kernel();
        void init_data_gpu(vector<int64_t>& start,vector<int64_t>& stop,
            vector<uint32_t>& istart,vector<uint32_t>& istop,
            vector<int64_t>& times_ms,
            vector<unsigned char>& mask_ad,vector<unsigned char>& mask_dd,
            vector<float>& T_burst_duration,vector<float>& SgDivSr,
            float& clk_p,float& bg_ad_rate,float& bg_dd_rate);
        ~mc();
        mc(int devid);
        bool set_params(vector<float>& args);
};

#endif