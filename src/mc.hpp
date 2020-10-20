/**
 * @file mc.hpp
 * @author liukan (liuk@apm.ac.cn)
 * @brief  header of cuda code for gSMFRETda
 */
#pragma once
#include <atomic>
#include <vector>
#include <cstdint>
#include <queue>
using namespace std;
#include "eigenhelper.hpp"
#include <curand_kernel.h>
#include "cuda_tools.hpp"
#include "loadHdf5.hpp"
#define DEBUGMC 0
// #include "Poco/Mutex.h"
#include "helper_cuda.h"

#include "mrImp.hpp"
typedef struct {
    unsigned int xor128[4];
    double gauss;
    int has_gauss; // !=0: gauss contains a gaussian deviate
    int has_binomial; // !=0: following parameters initialized for binomial
    /* The rk_state structure has been extended to store the following
     * information for the binomial generator. If the input values of n or p
     * are different than nsave and psave, then the other parameters will be
     * recomputed. RTK 2005-09-02 */
    int nsave, m;
    double psave, r, q, fm, p1, xm, xl, xr, c, laml, lamr, p2, p3, p4;
} rk_state;
int showGPUsInfo(int dn=-1,char *gpuuid=NULL,int *asyncEngineCount=NULL);
class mc
{
    protected:
        mrImp *mr;
        long sz_tag;
        // unsigned char *g_mask_ad,*g_mask_dd;
        // float *gchi2,*hchi2;
        float **hpe,**hpk,**hpp,**hpv;
        float **gpe,**gpk,**gpp,**gpv;        
        int64_t *g_start,*g_stop;//,*g_times_ms;
        int* g_phCount;
        int64_t *g_burst_ad,  *g_burst_dd;
        int64_t *g_istart,*g_istop;  // 64
        float *g_burst_duration,*g_SgDivSr,clk_p,bg_ad_rate,bg_dd_rate;
        float **g_P_i2j;
        arrFF **matK;
        arrF** matP; 
        
        int *begin_burst;
        int *end_burst;
        rk_state** devStates;
        curandStateScrambledSobol64** devQStates;
        curandDirectionVectors64_t* hostVectors64;
        unsigned long long int* hostScrambleConstants64;
        unsigned long long int** devDirectionVectors64;
        unsigned long long int** devScrambleConstants64;
        int *oldN;
        cudaStream_t* streams;
        int blockSize;   // The launch configurator returned block size 
        int minGridSize; // The minimum grid size needed to achieve the 
                         // maximum occupancy for a full device launch 
        int *gridSize;   // The actual grid size needed, based on input size   
        retype** mcE;
        unsigned char debug;
        std::queue<int> streamFIFO;
        int getStream();
        void givebackStream(int i); 
        // Poco::FastMutex streamLock;
        int nDevices;
        bool profiler;
    public:      
        int streamNum;  
        int devid;  
        char gpuuuid[33];
        atomic_int workerNum;
        retype** hmcE;
        int *s_n;
        int sz_burst;
        bool streamQuery(int sid);
        int reSampleTimes;        
        int set_nstates(int n,int sid);
        /**
         * @brief Set which GPU for calculating
         * 
         */
        void set_gpuid();
        int  setBurstBd(int cstart,int cstop, int sid);
        void set_reSampleTimes(int n);
        void free_data_gpu();
        void init_randstate(int N,int sid);
        void run_kernel(int N, int sid);
        void get_res(int sid, int N);
        void set_params_buff(int oldS_n,int N_sid,int sid);
        void init_data_gpu(vector<int64_t>& istart,vector<int64_t>& start,vector<int64_t>& stop,
                        std::vector<int>& phCount,long _sz_tag,int64_t *burst_ad, int64_t *burst_dd,
                        vector<float>& T_burst_duration,vector<float>& SgDivSr,
                        float& iclk_p,float& ibg_ad_rate,float& ibg_dd_rate);
        ~mc();
        /**
         * @brief Construct a new mc object
         * 
         * @param devid the GPU id to use
         * @param _streamNum total stream number
         * @param debug  debug flag
         * @param hdf5size HDF5 file size to setup gpumem pool size
         * @param profiler  Profiler cuda flag
         */
        mc(int devid,int _streamNum=16,unsigned char debug=DEBUGMC,std::uintmax_t hdf5size=0,bool profiler=false);
        bool set_params(int n,int sid,vector<float>& args);        
};

