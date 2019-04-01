#include "mc.hpp"
#include "cuda_tools.hpp"

#include <time.h>
#define VECTOR_SIZE 64
#include <algorithm> 
#include "binom.cuh"
#include "gen_rand.cuh"
#include "cuList.cuh"

//__forceinline__ 
template <typename T>
__device__ void binTimeHist(arrF* hist, arrI64& x,
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
__global__ void mc_kernel(int64_t* start,int64_t* stop,
    uint32_t* istart,uint32_t* istop,
    int64_t* times_ms,
    unsigned char* mask_ad,unsigned char* mask_dd,
    float* T,/*float* SgDivSr,*/
    float clk_p,float bg_ad_rate,float bg_dd_rate,long sz_tag,int sz_burst ,
    float* gpe,float* gpv,float* gpk,float* gpp,
    int N,int s_n,curandStateScrambledSobol64 *devQStates,
    rk_state *devStates, retype *mcE,int reSampleTimes,
    float gamma=0.34,float beta=1.42,float DexDirAem=0.08, 
    float Dch2Ach=0.07,float r0=52){
    
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;    
    if (tidx<N*reSampleTimes){
        int idx=tidx/reSampleTimes;
        arrUcharMapper mask_adA(mask_ad+istart[idx],istop[idx]-istart[idx]);
        arrUcharMapper mask_ddA(mask_dd+istart[idx],istop[idx]-istart[idx]);
        arrI64Mapper times_msA(times_ms+istart[idx],istop[idx]-istart[idx]);        
        arrI64 burst_dd=mask_ddA.cast<int64_t>()*times_msA;
        arrI64 burst_ad=mask_adA.cast<int64_t>()*times_msA;
        // for (int sampleTime=0;sampleTime<reSampleTimes;sampleTime++){
            int sampleTime=tidx%reSampleTimes;
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
            arrF f_i(bins.len-1);
            arrF f_if(bins.len-1);
            f_if=(gamma-Dch2Ach)*f_id + (1-DexDirAem)*f_ia;
            if (bg_dd_rate<1e-4){
                f_i=(f_if+0.5).floor();
            }
            else{
                arrF t_diff(bins.len-1);
                bins.diff(&t_diff);
                t_diff=t_diff*clk_p;
                arrF rf_ia(bins.len-1);
                rf_ia=(1-DexDirAem)*f_ia;
                arrF bg_a(bins.len-1);
                draw_P_B_Tr(bg_a.data(),rf_ia.data(),bins.len-1,t_diff.data(),bg_ad_rate ,devQStates+idx);
                arrF rf_id(bins.len-1);
                rf_id=(gamma-Dch2Ach)*f_id;
                arrF bg_d(bins.len-1);
                draw_P_B_Tr(bg_d.data(),rf_id.data(),bins.len-1,t_diff.data(),bg_dd_rate ,devQStates+idx);   
                f_i=(f_if - bg_d - bg_a+0.5).floor();             
            }
            float F=f_if.sum();
            for (int s_trans=0;s_trans<bins.len-1;s_trans++){
                float de=drawE(gpe[*(sidx.at(s_trans))],r0,
                    gpv[*(sidx.at(s_trans))],devQStates+idx);
                long ai=drawA_fi_e(devStates+idx, f_i(s_trans), de) ;
                mcE[idx*reSampleTimes+sampleTime]+=ai;
            }
            mcE[idx*reSampleTimes+sampleTime]/=F;
            sidx.freeList();
            bins.freeList();
        // }
    }    
}

mc::mc(int id,int _streamNum, bool de){    
    debug=de;
    streamNum=_streamNum;
    streams=new cudaStream_t[streamNum];
    devid=id;
    CUDA_CHECK_RETURN(cudaSetDevice(devid));
    for(int sid=0;sid<_streamNum;sid++){
        cudaStreamCreate ( &(streams[sid])) ;
        streamFIFO.push(sid);
    }
    matK=NULL;matP=NULL;        
    hpv=(float **)malloc(sizeof(float*)*streamNum);
    hpk=(float **)malloc(sizeof(float*)*streamNum);
    hpp=(float **)malloc(sizeof(float*)*streamNum);
    hpe=(float **)malloc(sizeof(float*)*streamNum);    
    gpv=(float **)malloc(sizeof(float*)*streamNum);
    gpk=(float **)malloc(sizeof(float*)*streamNum);
    gpp=(float **)malloc(sizeof(float*)*streamNum);
    gpe=(float **)malloc(sizeof(float*)*streamNum);    
    std::memset(hpv, 0, sizeof(float*)*streamNum);
    std::memset(hpk, 0, sizeof(float*)*streamNum);
    std::memset(hpp, 0, sizeof(float*)*streamNum);
    std::memset(hpe, 0, sizeof(float*)*streamNum);    
    std::memset(gpv, 0, sizeof(float*)*streamNum);
    std::memset(gpk, 0, sizeof(float*)*streamNum);
    std::memset(gpp, 0, sizeof(float*)*streamNum);
    std::memset(gpe, 0, sizeof(float*)*streamNum);        
    s_n=new int[streamNum];
    gridSize = new int[streamNum];
    begin_burst=new int[streamNum];
    end_burst=new int[streamNum];
    std::fill_n(s_n, streamNum, 0); 
    std::fill_n(gridSize, streamNum, 0); 
    std::fill_n(begin_burst, streamNum, 0); 
    std::fill_n(end_burst, streamNum, 0); 
    mcE=(retype **)malloc(sizeof(retype*)*streamNum);
    hmcE=(retype **)malloc(sizeof(retype*)*streamNum);
    std::memset(mcE, 0, sizeof(retype*)*streamNum);
    std::memset(hmcE, 0, sizeof(retype*)*streamNum);
    devStates=(rk_state**)malloc(sizeof(rk_state*)*streamNum);
    devQStates=(curandStateScrambledSobol64**)malloc(sizeof(curandStateScrambledSobol64*)*streamNum);
    hostVectors64=(curandDirectionVectors64_t**)malloc(sizeof(curandDirectionVectors64_t*)*streamNum);
    hostScrambleConstants64=(unsigned long long int**)malloc(sizeof(unsigned long long int*)*streamNum);
    devDirectionVectors64=(unsigned long long int**)malloc(sizeof(unsigned long long int*)*streamNum);
    devScrambleConstants64=(unsigned long long int**)malloc(sizeof(unsigned long long int*)*streamNum);
    std::memset(devStates, 0, sizeof(rk_state*)*streamNum);
    std::memset(devQStates, 0, sizeof(curandStateScrambledSobol64*)*streamNum);
    std::memset(hostVectors64, 0, sizeof(curandDirectionVectors64_t*)*streamNum);
    std::memset(devDirectionVectors64, 0, sizeof(unsigned long long int*)*streamNum);
    std::memset(hostScrambleConstants64, 0, sizeof(unsigned long long int*)*streamNum);
    std::memset(devScrambleConstants64, 0, sizeof(unsigned long long int*)*streamNum);    
    reSampleTimes=5;
    cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, mc_kernel, 0, 0);     
    
}
void mc::set_reSampleTimes(int t){
    reSampleTimes=t;
}
int mc::getStream(){
    int r=-1;
    if(!streamFIFO.empty()){
        int r=streamFIFO.front();
        streamFIFO.pop();
    }
    return r;
}

void mc::givebackStream(int i){    
    streamFIFO.push(i);
}

void mc::int_randstate(int N){
    int NN=N;
    if (N==-1){
        NN=sz_burst;
    }    
    for (int i=0;i<streamNum;i++){
        CUDA_CHECK_RETURN(cudaFree ( devStates[i]));
        CUDA_CHECK_RETURN(cudaFree ( devQStates[i]));    
        CUDA_CHECK_RETURN(cudaMalloc ( (void **)&(devStates[i]), N*sizeof (rk_state ) ));
        CUDA_CHECK_RETURN(cudaMalloc ( (void **)&(devQStates[i]), N*sizeof( curandStateScrambledSobol64) ));    
        CUDA_CHECK_RETURN(cudaFree (devDirectionVectors64[i]));
        CUDA_CHECK_RETURN(cudaFree (devScrambleConstants64[i]));
        CUDA_CHECK_RETURN(cudaMalloc((void **)&(devDirectionVectors64[i]), 
        N * VECTOR_SIZE * sizeof(long long int)));       
        CUDA_CHECK_RETURN(cudaMalloc((void **)&(devScrambleConstants64[i]), 
        N * sizeof(long long int)));
        // CUDA_CHECK_RETURN
        (curandGetDirectionVectors64( &(hostVectors64[i]), 
            CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6));
        // CUDA_CHECK_RETURN
        (curandGetScrambleConstants64( &(hostScrambleConstants64[i])));              

        CUDA_CHECK_RETURN(cudaMemcpyAsync(devDirectionVectors64[i],
             hostVectors64[i],
            N * VECTOR_SIZE * sizeof(long long int), 
            cudaMemcpyHostToDevice,streams[i])); 
        CUDA_CHECK_RETURN(cudaMemcpyAsync(devScrambleConstants64[i], hostScrambleConstants64[i],
        N * sizeof(long long int), 
        cudaMemcpyHostToDevice,streams[i]));
        gridSize[i] = (NN + blockSize - 1) / blockSize;
        // setup_kernel <<<blocks,     threads,0,streams[i]>>>(devStates[i], 0,/*time(NULL)*/ NN ,          
        setup_kernel <<<gridSize[i], blockSize,0,streams[i]>>>(devStates[i], 0,/*time(NULL)*/ NN,        
            devDirectionVectors64[i], devScrambleConstants64[i], devQStates[i]);    
    }
}

void mc::init_data_gpu(vector<int64_t>& start,vector<int64_t>& stop,
        vector<uint32_t>& istart,vector<uint32_t>& istop,
        vector<int64_t>& times_ms,
        vector<unsigned char>& mask_ad,vector<unsigned char>& mask_dd,
        vector<float>& T_burst_duration,vector<float>& SgDivSr,
        float& iclk_p,float& ibg_ad_rate,float& ibg_dd_rate){    
    clk_p=iclk_p;bg_ad_rate=ibg_ad_rate;bg_dd_rate=ibg_dd_rate;    
    sz_tag=mask_ad.size();                    
    // CUDA_CHECK_RETURN(cudaMallocHost((void **)&hchi2, sizeof(float)));
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
    // CUDA_CHECK_RETURN(cudaMalloc((void **)&gchi2, sizeof(float)));    
}

int mc::setBurstBd(int cstart,int cstop, int sid){
    int rcstart=cstart;
    int rcstop=cstop;
    if (cstop>=sz_burst||cstop<0)  
        rcstop=sz_burst;
    if(cstart>=rcstop){
        rcstart=0;
    }
    int N=rcstop-rcstart;
    if(cstart==-1){
        gridSize[sid]=-blockSize;
        N=-1;
    }
    if(end_burst[sid]-begin_burst[sid]!=N){
        begin_burst[sid]=rcstart;
        end_burst[sid]=rcstop;
        // int dimension=256;  
        // dim3 threads = dim3(dimension, 1);
        // int blocksCount = ceil(float(N)/dimension);
        // dim3 blocks  = dim3(blocksCount, 1);    
        gridSize[sid] = (N*reSampleTimes + blockSize - 1) / blockSize; 
        CUDA_CHECK_RETURN(cudaFree(mcE[sid]));
        CUDA_CHECK_RETURN(cudaFreeHost(hmcE[sid]));
        CUDA_CHECK_RETURN(cudaMalloc((void **)&(mcE[sid]), N *reSampleTimes* sizeof(retype)));        
        CUDA_CHECK_RETURN(cudaMallocHost((void **)&(hmcE[sid]), N *reSampleTimes* sizeof(retype)));
    }
    CUDA_CHECK_RETURN(cudaMemset(mcE[sid], 0, N *reSampleTimes* sizeof(retype)));
    return N;    
}

void mc::run_kernel(int N, int sid){ 
    // mc_kernel<<<blocks, threads,0,streams[sid]>>>(gchi2, g_start,g_stop,
    mc_kernel<<<gridSize[sid],blockSize,0,streams[sid]>>>(g_start,g_stop,            
        g_istart,g_istop,
        g_times_ms,
        g_mask_ad,g_mask_dd,
        g_burst_duration,/*g_SgDivSr,*/
        clk_p,bg_ad_rate,bg_dd_rate,sz_tag,sz_burst ,
        gpe[sid],gpv[sid],gpk[sid],gpp[sid],N,s_n[sid],
        devQStates[sid],devStates[sid], mcE[sid], reSampleTimes/*,ti*/);
    CUDA_CHECK_RETURN(cudaMemcpyAsync(hmcE[sid], mcE[sid],N *reSampleTimes* sizeof(retype), cudaMemcpyDeviceToHost));
    cudaStreamSynchronize(streams[sid]);
    if (debug){
        std::vector<retype> my_vector(hmcE[sid], hmcE[sid] + N*reSampleTimes);
        for (int ip=0;ip<N;ip++)
            cout<<my_vector.at(ip)<<" ";
        cout<<endl;
        // savehdf5("r.hdf5", "/r",my_vector);
    }
}

mc::~mc(){
    free_data_gpu();    
    delete(matK);delete(matP);
    for(int sid=0;sid<streamNum;sid++)
        cudaStreamDestroy ( streams[sid]);

    delete[](streams);
    delete[](s_n);
    delete[](begin_burst);
    delete[](end_burst);
    delete[](gridSize);
    free(hmcE);
    free(mcE);
    free(hpe);
    free(hpv);
    free(hpp);
    free(hpk);    
    free(gpe);
    free(gpv);
    free(gpp);
    free(gpk);        

    free(devStates);
    free(devQStates);    
    free(hostVectors64);
    free(hostScrambleConstants64);
    free(devDirectionVectors64);
    free(devScrambleConstants64);        
    
}

void mc::free_data_gpu(){            
    cudaDeviceSynchronize();
    // CUDA_CHECK_RETURN(cudaFree(g_mask_ad));
    // CUDA_CHECK_RETURN(cudaFree(g_mask_dd));
    // CUDA_CHECK_RETURN(cudaFree(g_start));
    // CUDA_CHECK_RETURN(cudaFree(g_stop));    
    // CUDA_CHECK_RETURN(cudaFree(g_istart));
    // CUDA_CHECK_RETURN(cudaFree(g_istop)); 
    // CUDA_CHECK_RETURN(cudaFree(g_times_ms));
    // CUDA_CHECK_RETURN(cudaFree(g_SgDivSr));
    // CUDA_CHECK_RETURN(cudaFree(g_burst_duration));    
    
    for (int sid=0;sid<streamNum;sid++){
        CUDA_CHECK_RETURN(cudaFreeHost(hpe[sid]));
        CUDA_CHECK_RETURN(cudaFreeHost(hpv[sid]));
        CUDA_CHECK_RETURN(cudaFreeHost(hpp[sid]));
        CUDA_CHECK_RETURN(cudaFreeHost(hpk[sid]));
        CUDA_CHECK_RETURN(cudaFree(gpe[sid]));
        CUDA_CHECK_RETURN(cudaFree(gpv[sid]));
        CUDA_CHECK_RETURN(cudaFree(gpp[sid]));
        CUDA_CHECK_RETURN(cudaFree(gpk[sid]));    

        CUDA_CHECK_RETURN(cudaFree(mcE[sid]));
        CUDA_CHECK_RETURN(cudaFreeHost(hmcE[sid]));
        CUDA_CHECK_RETURN(cudaFree(devStates[sid]));
        CUDA_CHECK_RETURN(cudaFree(devQStates[sid]));
        // CUDA_CHECK_RETURN(cudaFreeHost(hostVectors64[sid]));
        // CUDA_CHECK_RETURN(cudaFreeHost(hostScrambleConstants64[sid]));
        CUDA_CHECK_RETURN(cudaFree(devDirectionVectors64[sid]));
        CUDA_CHECK_RETURN(cudaFree(devScrambleConstants64[sid]));            
    }

}

bool mc::set_nstates(int n,int sid){
    bool r=false;
    if (s_n[sid]!=n){
        s_n[sid]=n;
        r=true;
        CUDA_CHECK_RETURN(cudaFreeHost(hpe[sid]));
        CUDA_CHECK_RETURN(cudaFreeHost(hpv[sid]));
        CUDA_CHECK_RETURN(cudaFreeHost(hpp[sid]));
        CUDA_CHECK_RETURN(cudaFreeHost(hpk[sid]));
        CUDA_CHECK_RETURN(cudaFree(gpe[sid]));
        CUDA_CHECK_RETURN(cudaFree(gpv[sid]));
        CUDA_CHECK_RETURN(cudaFree(gpp[sid]));
        CUDA_CHECK_RETURN(cudaFree(gpk[sid]));    
        CUDA_CHECK_RETURN(cudaMallocHost((void **)&(hpe[sid]), n*sizeof(float)));
        CUDA_CHECK_RETURN(cudaMallocHost((void **)&(hpv[sid]), n*sizeof(float)));
        CUDA_CHECK_RETURN(cudaMallocHost((void **)&(hpp[sid]), n*sizeof(float)));
        CUDA_CHECK_RETURN(cudaMallocHost((void **)&(hpk[sid]), n*n*sizeof(float)));
        CUDA_CHECK_RETURN(cudaMalloc((void **)&(gpe[sid]), n*sizeof(float)));
        CUDA_CHECK_RETURN(cudaMalloc((void **)&(gpv[sid]), n*sizeof(float)));
        CUDA_CHECK_RETURN(cudaMalloc((void **)&(gpp[sid]), n*sizeof(float)));
        CUDA_CHECK_RETURN(cudaMalloc((void **)&(gpk[sid]), n*n*sizeof(float)));    
    }
    return r;
}

bool mc::set_params(int n,int sid,vector<float>& args){
    set_nstates(n,sid);    
    vecFloatMapper evargs(args.data(),n*n+n);        
    RowVectorXf eargs=evargs(seqN(0,n));
    float *peargs=eargs.data();
    RowVectorXf kargs=evargs(seqN(n,n*n-n));    
    RowVectorXf vargs=evargs(seqN(n*n,n));    
    float *pvargs=vargs.data();
    bool r=genMatK(&matK,n,kargs);
    //&matK不可修改，但是matK的值可以修改    
    r=r&&genMatP(&matP,matK);    
    // cout<<"p:"<<*matP<<endl;
    // memcpy(hpe[sid], peargs, sizeof(float)*n);
    // memcpy(hpv[sid], pvargs, sizeof(float)*n);
    // memcpy(hpk[sid], matK->data(), sizeof(float)*n*n);
    // memcpy(hpp[sid], matP->data(), sizeof(float)*n);
    CUDA_CHECK_RETURN(cudaMemcpyAsync(gpe[sid],peargs,sizeof(float)*n,
        cudaMemcpyHostToDevice,streams[sid]));
    CUDA_CHECK_RETURN(cudaMemcpyAsync(gpv[sid],pvargs, sizeof(float)*n,
        cudaMemcpyHostToDevice,streams[sid]));
    CUDA_CHECK_RETURN(cudaMemcpyAsync(gpk[sid],matK->data(), sizeof(float)*n*n,
        cudaMemcpyHostToDevice,streams[sid]));
    CUDA_CHECK_RETURN(cudaMemcpyAsync(gpp[sid],matP->data(), sizeof(float)*n,
        cudaMemcpyHostToDevice,streams[sid]));
    return r;
}