#include "mc.hpp"
#include "cuda_tools.hpp"
#include "loadHdf5.hpp"
#include <time.h>
#define VECTOR_SIZE 64

#include "binom.cuh"
#include "gen_rand.cuh"


__global__ void mc_kernel(float *chi2, int64_t* start,int64_t* stop,
    uint32_t* istart,uint32_t* istop,
    int64_t* times_ms,
    unsigned char* mask_ad,unsigned char* mask_dd,
    float* T_burst_duration,float* SgDivSr,
    float clk_p,float bg_ad_rate,float bg_dd_rate,long sz_tag,int sz_burst ,
    float* gpe,float* gpv,float* gpk,float* gpp,
    int N,int n,curandStateScrambledSobol64 *devQStates,rk_state *devStates, retype *intr){
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<N){
        arrUcharMapper mask_adA(mask_ad,sz_tag);
        // intr[idx]=drawDisIdx(n,gpp,devQStates+idx);
        // intr[idx]=drawTau(25,devQStates+idx);
        // float t=1;
        // draw_P_B_Tr(intr+idx,35,1,&t,6 ,devQStates+idx);
        // intr[idx]=drawE(3.0,6,devQStates+idx);        
        intr[idx]=drawA_fi_e(devStates+idx, 5, 0.7) ;
    }
    
}

__global__ void setup_kernel  (rk_state * state, unsigned long seed , int N,
    unsigned long long * sobolDirectionVectors, 
    unsigned long long *sobolScrambleConstants, 
    curandStateScrambledSobol64* stateQ)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<N){
        // curand_init ( seed, idx, 0, &state[idx] );        
        curand_init(sobolDirectionVectors + VECTOR_SIZE*idx, 
            sobolScrambleConstants[idx], 
            1234, 
            &stateQ[idx]);
        unsigned long long llseed=curand(stateQ+idx);    
        rk_seed(llseed,state+idx);
    }
} 

mc::mc(int id){
    devid=id;
    matK=NULL;matP=NULL;
    hpe=hpv=hpk=hpp=gpe=gpv=gpp=gpk=NULL;    
    devStates=NULL;
    devQStates=NULL;        
    CUDA_CHECK_RETURN(cudaSetDevice(devid));
    hostVectors64=NULL;
    hostScrambleConstants64=NULL;
    devDirectionVectors64=NULL;
    devScrambleConstants64=NULL;
    
}

void mc::init_data_gpu(vector<int64_t>& start,vector<int64_t>& stop,
        vector<uint32_t>& istart,vector<uint32_t>& istop,
        vector<int64_t>& times_ms,
        vector<unsigned char>& mask_ad,vector<unsigned char>& mask_dd,
        vector<float>& T_burst_duration,vector<float>& SgDivSr,
        float& iclk_p,float& ibg_ad_rate,float& ibg_dd_rate){    
    clk_p=iclk_p;bg_ad_rate=ibg_ad_rate;bg_dd_rate=ibg_dd_rate;    
    sz_tag=mask_ad.size();                    
    CUDA_CHECK_RETURN(cudaMallocHost((void **)&hchi2, sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&g_mask_ad, sizeof(unsigned char)*sz_tag));
    CUDA_CHECK_RETURN(cudaMemcpy(g_mask_ad, mask_ad.data(), sizeof(unsigned char)*sz_tag, 
        cudaMemcpyHostToDevice));    
    CUDA_CHECK_RETURN(cudaMalloc((void **)&g_mask_dd, sizeof(unsigned char)*sz_tag));
    CUDA_CHECK_RETURN(cudaMemcpy(g_mask_dd, mask_dd.data(), sizeof(unsigned char)*sz_tag,cudaMemcpyHostToDevice));
    sz_burst=start.size();         
    CUDA_CHECK_RETURN(cudaMalloc((void **)&g_start, sizeof(int64_t)*sz_burst));
    CUDA_CHECK_RETURN(cudaMemcpy(g_start, start.data(), sizeof(int64_t)*sz_burst,cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&g_stop, sizeof(int64_t)*sz_burst));
    CUDA_CHECK_RETURN(cudaMemcpy(g_stop, stop.data(), sizeof(int64_t)*sz_burst,cudaMemcpyHostToDevice));        
    CUDA_CHECK_RETURN(cudaMalloc((void **)&g_istart, sizeof(uint32_t)*sz_burst));
    CUDA_CHECK_RETURN(cudaMemcpy(g_istart, istart.data(), sizeof(uint32_t)*sz_burst,cudaMemcpyHostToDevice));    
    CUDA_CHECK_RETURN(cudaMalloc((void **)&g_istop, sizeof(uint32_t)*sz_burst));
    CUDA_CHECK_RETURN(cudaMemcpy(g_istop, istop.data(), sizeof(uint32_t)*sz_burst,cudaMemcpyHostToDevice));            
    CUDA_CHECK_RETURN(cudaMalloc((void **)&g_times_ms, sizeof(int64_t)*sz_tag));
    CUDA_CHECK_RETURN(cudaMemcpy(g_times_ms, times_ms.data(), sizeof(int64_t)*sz_tag,cudaMemcpyHostToDevice));        
    CUDA_CHECK_RETURN(cudaMalloc((void **)&g_burst_duration, sizeof(float)*sz_burst));
    CUDA_CHECK_RETURN(cudaMemcpy(g_burst_duration, T_burst_duration.data(), sizeof(float)*sz_burst,cudaMemcpyHostToDevice));        
    CUDA_CHECK_RETURN(cudaMalloc((void **)&g_SgDivSr, sizeof(float)*sz_burst));
    CUDA_CHECK_RETURN(cudaMemcpy(g_SgDivSr, SgDivSr.data(), sizeof(float)*sz_burst,cudaMemcpyHostToDevice));        
    CUDA_CHECK_RETURN(cudaMalloc((void **)&gchi2, sizeof(float)));
}

void mc::run_kernel(int cstart,int cstop){  
    int N=cstop-cstart;
    int dimension=128;  
    dim3 threads = dim3(dimension, 1);
    int blocksCount = ceil(N / dimension);
    dim3 blocks  = dim3(blocksCount, 1);    
    CUDA_CHECK_RETURN(cudaFree ( devStates));
    CUDA_CHECK_RETURN(cudaFree ( devQStates));    
    CUDA_CHECK_RETURN(cudaMalloc ( (void **)&devStates, N*sizeof (rk_state ) ));
    CUDA_CHECK_RETURN(cudaMalloc ( (void **)&devQStates, N*sizeof( curandStateScrambledSobol64) ));    
    CUDA_CHECK_RETURN(cudaFree (devDirectionVectors64));
    CUDA_CHECK_RETURN(cudaFree (devScrambleConstants64));
    // CUDA_CHECK_RETURN
    (curandGetDirectionVectors64( &hostVectors64, 
        CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6));
    // CUDA_CHECK_RETURN
    (curandGetScrambleConstants64( &hostScrambleConstants64));              
    CUDA_CHECK_RETURN(cudaMalloc((void **)&(devDirectionVectors64), 
    N * VECTOR_SIZE * sizeof(long long int)));        
    CUDA_CHECK_RETURN(cudaMemcpy(devDirectionVectors64, hostVectors64,
    N * VECTOR_SIZE * sizeof(long long int), 
    cudaMemcpyHostToDevice)); 
    CUDA_CHECK_RETURN(cudaMalloc((void **)&(devScrambleConstants64), 
    N * sizeof(long long int)));
    CUDA_CHECK_RETURN(cudaMemcpy(devScrambleConstants64, hostScrambleConstants64,
    N * sizeof(long long int), 
    cudaMemcpyHostToDevice));
    setup_kernel <<<blocks, threads>>>(devStates, 0,/*time(NULL)*/ N ,
        devDirectionVectors64, devScrambleConstants64, devQStates);

    retype *intr,*hintr;
    CUDA_CHECK_RETURN(cudaMalloc((void **)&intr, N * sizeof(retype)));
    CUDA_CHECK_RETURN(cudaMallocHost((void **)&hintr, N * sizeof(retype)));
    mc_kernel<<<blocks, threads>>>(gchi2, g_start,g_stop,
        g_istart,g_istop,
        g_times_ms,
        g_mask_ad,g_mask_dd,
        g_burst_duration,g_SgDivSr,
        clk_p,bg_ad_rate,bg_dd_rate,sz_tag,sz_burst ,
        gpe,gpv,gpk,gpp,N,s_n,devQStates,devStates, intr);
    CUDA_CHECK_RETURN(cudaMemcpy(hintr, intr,N * sizeof(retype), cudaMemcpyDeviceToHost));        
    std::vector<retype> my_vector(hintr, hintr + N);
    for (int ip=0;ip<N;ip++)
        cout<<my_vector.at(ip)<<" ";
    cout<<endl;
    savehdf5("r.hdf5", "/r",my_vector);
    CUDA_CHECK_RETURN(cudaFree(intr));
    CUDA_CHECK_RETURN(cudaFreeHost(hintr));
}

mc::~mc(){
    free_data_gpu();
    delete(matK);delete(matP);
}

void mc::free_data_gpu(){            
    CUDA_CHECK_RETURN(cudaFree(g_mask_ad));
    CUDA_CHECK_RETURN(cudaFree(g_mask_dd));
    CUDA_CHECK_RETURN(cudaFree(g_start));
    CUDA_CHECK_RETURN(cudaFree(g_stop));    
    CUDA_CHECK_RETURN(cudaFree(g_istart));
    CUDA_CHECK_RETURN(cudaFree(g_istop)); 
    CUDA_CHECK_RETURN(cudaFree(g_times_ms));
    CUDA_CHECK_RETURN(cudaFree(g_SgDivSr));
    CUDA_CHECK_RETURN(cudaFree(g_burst_duration));

    // CUDA_CHECK_RETURN(cudaFree(r_size));
    cudaDeviceSynchronize();
    cout<<"rsize:"<<*hchi2<<endl;
    CUDA_CHECK_RETURN(cudaFreeHost(hchi2));
    CUDA_CHECK_RETURN(cudaFreeHost(hpe));
    CUDA_CHECK_RETURN(cudaFreeHost(hpv));
    CUDA_CHECK_RETURN(cudaFreeHost(hpp));
    CUDA_CHECK_RETURN(cudaFreeHost(hpk));
    CUDA_CHECK_RETURN(cudaFree(gpe));
    CUDA_CHECK_RETURN(cudaFree(gpv));
    CUDA_CHECK_RETURN(cudaFree(gpp));
    CUDA_CHECK_RETURN(cudaFree(gpk));        
}

bool mc::set_nstates(int n){
    s_n=n;
    bool r;
    CUDA_CHECK_RETURN(cudaFreeHost(hpe));
    CUDA_CHECK_RETURN(cudaFreeHost(hpv));
    CUDA_CHECK_RETURN(cudaFreeHost(hpp));
    CUDA_CHECK_RETURN(cudaFreeHost(hpk));
    CUDA_CHECK_RETURN(cudaFree(gpe));
    CUDA_CHECK_RETURN(cudaFree(gpv));
    CUDA_CHECK_RETURN(cudaFree(gpp));
    CUDA_CHECK_RETURN(cudaFree(gpk));    
    CUDA_CHECK_RETURN(cudaMallocHost((void **)&hpe, n*sizeof(float)));
    CUDA_CHECK_RETURN(cudaMallocHost((void **)&hpv, n*sizeof(float)));
    CUDA_CHECK_RETURN(cudaMallocHost((void **)&hpp, n*sizeof(float)));
    CUDA_CHECK_RETURN(cudaMallocHost((void **)&hpk, n*n*sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&gpe, n*sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&gpv, n*sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&gpp, n*sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&gpk, n*n*sizeof(float)));    
    return r;
}

bool mc::set_params(vector<float>& args){
    int n=s_n;
    vecFloatMapper evargs(args.data(),n*n+n);    
    cout<<evargs<<endl;
    eargs=evargs(seqN(0,n));
    float *peargs=eargs.data();
    kargs=evargs(seqN(n,n*n-n));    
    vargs=evargs(seqN(n*n,n));    
    float *pvargs=vargs.data();
    bool r=genMatK(&matK,n,kargs);
    //&matK不可修改，但是matK的值可以修改    
    r=r&&genMatP(&matP,matK);    
    cout<<"p:"<<*matP<<endl;
    memcpy(hpe, peargs, sizeof(float)*n);
    memcpy(hpv, pvargs, sizeof(float)*n);
    memcpy(hpk, matK->data(), sizeof(float)*n*n);
    memcpy(hpp, matP->data(), sizeof(float)*n);
    CUDA_CHECK_RETURN(cudaMemcpy(gpe,hpe,sizeof(float)*n,cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(gpv,hpv, sizeof(float)*n,cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(gpk,hpk, sizeof(float)*n*n,cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(gpp,hpp, sizeof(float)*n,cudaMemcpyHostToDevice));    
    return r;
}