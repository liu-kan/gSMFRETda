#include "mc.hpp"
#include "cuda_tools.hpp"
#include "loadHdf5.hpp"
#include <time.h>
#define VECTOR_SIZE 64

#include "binom.cuh"
#include "gen_rand.cuh"
#include "cuList.cuh"

#define gamma 0.34
#define beta 1.42
#define DexDirAem 0.08
#define Dch2Ach 0.07

template <typename T>
__forceinline__ __device__ void binTimeHist(arrF* hist, arrI64& x,
         cuList<T> bins ){
    int binlen=bins.len;
    hist->resize(1,binlen-1);
    hist->setZero();
    int datalen=x.cols();
    for (int i=0;i<datalen;i++){
        if(x(i)==0)
            continue;
        int idxbin=1;
        do{
            T v=*(bins.at(idxbin));
            if (x(i)<v){
                ((*hist)(idxbin-1))+=1;
                break;
            }
            idxbin++;
        }while(idxbin<binlen);
    }
}
__global__ void mc_kernel(float *chi2, int64_t* start,int64_t* stop,
    uint32_t* istart,uint32_t* istop,
    int64_t* times_ms,
    unsigned char* mask_ad,unsigned char* mask_dd,
    float* T,float* SgDivSr,
    float clk_p,float bg_ad_rate,float bg_dd_rate,long sz_tag,int sz_burst ,
    float* gpe,float* gpv,float* gpk,float* gpp,
    int N,int s_n,curandStateScrambledSobol64 *devQStates,rk_state *devStates, retype *mcE,int reSampleTimes/*,int tidx*/){
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<N /*&& idx==tidx*/){
        // arrUcharMapper mask_adA(mask_ad,sz_tag);
        // mcE[idx]=drawDisIdx(s_n,gpp,devQStates+idx);
        // mcE[idx]=drawTau(25,devQStates+idx);
        // mcE[idx]=drawTau(25,devQStates);
        // float t=1;
        // draw_P_B_Tr(mcE+idx,35,1,&t,6 ,devQStates+idx);
        // mcE[idx]=drawE(3.0,6,devQStates+idx);        
        // mcE[idx]=drawA_fi_e(devStates+idx, 5, 0.7) ;
        // mcE[tidx]=drawA_fi_e(devStates, 5, 0.7) ;
        // mcE[idx]=drawJ_Si2Sj(gpp,s_n,2,devQStates+idx);
        // cuList<int> l1;
        // for (int ti=0;ti<10;ti++){
        //     l1.append(ti);
        // }
        // arrI64 a(10);
        // a<<0,0,7,7,0,2,5,6,7,3;
        // arrI hist(9);
        // binTimeHist(&hist, a,l1);
        // mcE[idx]=hist(9);
        // l1.freeList();      

        arrUcharMapper mask_adA(mask_ad+istart[idx],istop[idx]-istart[idx]);
        arrUcharMapper mask_ddA(mask_dd+istart[idx],istop[idx]-istart[idx]);
        arrI64Mapper times_msA(times_ms+istart[idx],istop[idx]-istart[idx]);        
        arrI64 burst_dd=mask_ddA.cast<int64_t>()*times_msA;
        arrI64 burst_ad=mask_adA.cast<int64_t>()*times_msA;
        for (int sampleTime=0;sampleTime<reSampleTimes;sampleTime++){
            int si=drawDisIdx(s_n,gpp,devQStates+idx);
            cuList<int> sidx;
            cuList<int64_t> bins;
            bins.append(start[idx]);
            sidx.append(si);            
            float mcSpendTime=0;
            matXfMapper matK(gpk,s_n,s_n);
            while (T[idx]>mcSpendTime){
                int sj=drawJ_Si2Sj(gpp,s_n,si,devQStates+idx);
                sidx.append(sj);
                mcSpendTime+=drawTau(matK(si,sj),devQStates+idx);
                si=sj;                
                if(mcSpendTime>=T[idx]){
                    bins.append(stop[idx]);
                }
                else{
                    bins.append(*(bins.at(0))+mcSpendTime/clk_p);
                }
            }            
            arrF f_ia(bins.len-1);
            binTimeHist(&f_ia,burst_ad,bins);
            arrF f_id(bins.len-1);
            binTimeHist(&f_id,burst_dd,bins);            
            arrI f_i(bins.len-1);
            arrF f_if(bins.len-1);
            f_if=(gamma-Dch2Ach)*f_id + (1-DexDirAem)*f_ia;
            arrF t_diff(bins.len-1);
            bins.diff(&t_diff);
            t_diff=t_diff*clk_p;
            for (int s_trans=0;s_trans<bins.len-1;s_trans++){

            }
            sidx.freeList();
            bins.freeList();
        }
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
void mc::set_reSampleTimes(int t){
    reSampleTimes=t;
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

    retype *mcE,*hmcE;
    CUDA_CHECK_RETURN(cudaMalloc((void **)&mcE, N * sizeof(retype)));
    CUDA_CHECK_RETURN(cudaMallocHost((void **)&hmcE, N * sizeof(retype)));
    // int ti=0;
    // for( ;ti<N;ti++)
    mc_kernel<<<blocks, threads>>>(gchi2, g_start,g_stop,
        g_istart,g_istop,
        g_times_ms,
        g_mask_ad,g_mask_dd,
        g_burst_duration,g_SgDivSr,
        clk_p,bg_ad_rate,bg_dd_rate,sz_tag,sz_burst ,
        gpe,gpv,gpk,gpp,N,s_n,devQStates,devStates, mcE, reSampleTimes/*,ti*/);
    CUDA_CHECK_RETURN(cudaMemcpy(hmcE, mcE,N * sizeof(retype), cudaMemcpyDeviceToHost));        
    std::vector<retype> my_vector(hmcE, hmcE + N);
    for (int ip=0;ip<N;ip++)
        cout<<my_vector.at(ip)<<" ";
    cout<<endl;
    savehdf5("r.hdf5", "/r",my_vector);
    CUDA_CHECK_RETURN(cudaFree(mcE));
    CUDA_CHECK_RETURN(cudaFreeHost(hmcE));
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
    bool r=true;
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
    // cout<<evargs<<endl;
    eargs=evargs(seqN(0,n));
    float *peargs=eargs.data();
    kargs=evargs(seqN(n,n*n-n));    
    vargs=evargs(seqN(n*n,n));    
    float *pvargs=vargs.data();
    bool r=genMatK(&matK,n,kargs);
    //&matK不可修改，但是matK的值可以修改    
    r=r&&genMatP(&matP,matK);    
    // cout<<"p:"<<*matP<<endl;
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