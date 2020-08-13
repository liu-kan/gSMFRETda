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

class mc
{
    protected:
        int devid;
        long sz_tag;
        unsigned char *g_mask_ad,*g_mask_dd;
        // float *gchi2,*hchi2;
        float **hpe,**hpk,**hpp,**hpv;
        float **gpe,**gpk,**gpp,**gpv;        
        int64_t *g_start,*g_stop,*g_times_ms;
        uint32_t *g_istart,*g_istop;   
        float *g_burst_duration,*g_SgDivSr,clk_p,bg_ad_rate,bg_dd_rate;
        arrFF *matK;
        arrF* matP;                
        int *begin_burst;
        int *end_burst;
        rk_state** devStates;
        curandStateScrambledSobol64** devQStates;
        curandDirectionVectors64_t** hostVectors64;
        unsigned long long int** hostScrambleConstants64;
        unsigned long long int** devDirectionVectors64;
        unsigned long long int** devScrambleConstants64;
        
        cudaStream_t* streams;
        int streamNum;    
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
    public:        
        atomic_int workerNum;
        retype** hmcE;
        int *s_n;
        int sz_burst;        
        bool streamQuery(int sid);
        int reSampleTimes;        
        bool set_nstates(int n,int sid);
        void set_gpuid();
        int  setBurstBd(int cstart,int cstop, int sid);
        void set_reSampleTimes(int n);
        void free_data_gpu();
        void int_randstate(int N,int sid);
        void run_kernel(int N, int sid);
        void get_res(int sid, int N);
        void init_data_gpu(vector<int64_t>& start,vector<int64_t>& stop,
            vector<uint32_t>& istart,vector<uint32_t>& istop,
            vector<int64_t>& times_ms,
            vector<unsigned char>& mask_ad,vector<unsigned char>& mask_dd,
            vector<float>& T_burst_duration,vector<float>& SgDivSr,
            float& clk_p,float& bg_ad_rate,float& bg_dd_rate);
        ~mc();
        mc(int devid,int _streamNum=16,unsigned char debug=DEBUGMC);
        bool set_params(int n,int sid,vector<float>& args);        
};

