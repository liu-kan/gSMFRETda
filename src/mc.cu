#include "mc.hpp"
#include "cuda_tools.hpp"

__forceinline__ __device__ int drawDisIdx(int n,float* p){
    ;
}
__global__ void mc_kernel(float *chi2, int64_t* start,int64_t* stop,
    uint32_t* istart,uint32_t* istop,
    int64_t* times_ms,
    unsigned char* mask_ad,unsigned char* mask_dd,
    float* T_burst_duration,float* SgDivSr,
    float clk_p,float bg_ad_rate,float bg_dd_rate,long sz_tag,int sz_burst ){
    arrUcharMapper mask_adA(mask_ad,sz_tag);
    // *rsize=mask_adA.cols();
}

mc::mc(int id){
    devid=id;
    matK=NULL;matP=NULL;
    hpe=NULL;
    devStates=NULL;
    devQStates=NULL;        
    CUDA_CHECK_RETURN(cudaSetDevice(devid));
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

__global__ void setup_kernel ( curandState * state, curandStateSobol32_t* qstate, unsigned long seed , int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init ( seed, idx, 0, &state[idx] );
} 

void mc::run_kernel(int cstart,int cstop){  
    int N=cstop-cstart;
    int dimension=128;  
    dim3 threads = dim3(dimension, 1);
    int blocksCount = floor(N / threads.x) + 1;
    dim3 blocks  = dim3(blocksCount, 1);

    setup_kernel <<<blocks, threads>>> ( devStates, devQStates, time(NULL), N );
    mc_kernel<<<1,1>>>(gchi2, g_start,g_stop,
        g_istart,g_istop,
        g_times_ms,
        g_mask_ad,g_mask_dd,
        g_burst_duration,g_SgDivSr,
        clk_p,bg_ad_rate,bg_dd_rate,sz_tag,sz_burst );
    CUDA_CHECK_RETURN(cudaMemcpy(hchi2, gchi2, sizeof(float), cudaMemcpyDeviceToHost));        
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