#include "mc.hpp"
#include "cuda_tools.hpp"
#include <algorithm>
#include <time.h>
#define VECTOR_SIZE 64
#include <algorithm> 
#include <numeric>
#include "binom.cuh"
#include "gen_rand.cuh"
#include "cuList.cuh"
#include "rmm.hpp"
void CUDART_CB myStreamCallback(cudaStream_t stream, cudaError_t status, void *data) {
    if (status) {
      std::cout << "Error: " << cudaGetErrorString(status) << "-->";
    }
  }
  
#define CUDAstream_CHECK_LAST_ERROR   cudaStreamAddCallback(streams[sid], myStreamCallback, nullptr, 0)

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
/*
input:x 时间序列, bin0/1开始和结束时间点，x0起始搜索idx
output:hist 保存hist值，x0保存下次的起始位置
*/
template <typename T>
__device__ void p2TimeHist(float* hist, arrI64& x,
         T bin0, T bin1 , int64_t* x0){
    *hist=0;
    int datalen=x.cols();
    for (int i=x0;i<datalen;i++){
        if(x(i)==0)
            continue;
        if (x(i)<bin0)
            continue;
        else if(x(i)<bin1){
            found=true;
            *hist=(*hist)+1;
        }
        else{
            break;
            *x0=i;
        }
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
    if (tidx<N){
        int idx=tidx;//%reSampleTimes;
        arrUcharMapper mask_adA(mask_ad+istart[idx],istop[idx]-istart[idx]);
        arrUcharMapper mask_ddA(mask_dd+istart[idx],istop[idx]-istart[idx]);
        arrI64Mapper times_msA(times_ms+istart[idx],istop[idx]-istart[idx]);        
        arrI64 burst_dd=mask_ddA.cast<int64_t>()*times_msA;
        arrI64 burst_ad=mask_adA.cast<int64_t>()*times_msA;
        
        // printf("%d \n", mcE[tidx]);
        for (int sampleTime=0;sampleTime<reSampleTimes;sampleTime++){
            // // int sampleTime=tidx/reSampleTimes;
            int si=drawDisIdx(s_n,gpp,devQStates+tidx);
            // cuList<int> sidx;
            // cuList<int64_t> bins;
            // bins.append(start[idx]);
            // sidx.append(si);            
            float mcSpendTime=0;
            matXfMapper matK(gpk,s_n,s_n);
            // float count=0;
            int64_t bin0=start[idx];
            int64_t bin1=start[idx];
            while (T[idx]>mcSpendTime){
                int sj=drawJ_Si2Sj(gpp,s_n,si,devQStates+tidx);                
                // sidx.append(sj);
                float st=drawTau(matK(si,sj),devQStates+tidx);
                // printf("%f\t",st);
                mcSpendTime=mcSpendTime+st;
                si=sj;                
                if(mcSpendTime>=T[idx]){
                //     bins.append(stop[idx]);
                    bin1=stop[idx];
                }
                else{
                //     bins.append(*(bins.at(0))+mcSpendTime/clk_p);
                    bin1=bin0+mcSpendTime/clk_p;
                }
                // count++;
            }            
            // arrF f_ia(bins.len-1);
            // binTimeHist(&f_ia,burst_ad,bins);
            // arrF f_id(bins.len-1);
            // binTimeHist(&f_id,burst_dd,bins);
            // arrF f_i(bins.len-1);
            // arrF f_if(bins.len-1);
            // f_if=(gamma-Dch2Ach)*f_id + (1-DexDirAem)*f_ia;
            if (bg_dd_rate<1e-4){
                // f_i=(f_if+0.5).floor();
            }
            else{
                // arrF t_diff(bins.len-1);
                // bins.diff(&t_diff);
                // t_diff=t_diff*clk_p;
                // arrF rf_ia(bins.len-1);
                // rf_ia=(1-DexDirAem)*f_ia;
                // arrF bg_a(bins.len-1);
                // draw_P_B_Tr(bg_a.data(),rf_ia.data(),bins.len-1,t_diff.data(),bg_ad_rate ,devQStates+tidx);
                // arrF rf_id(bins.len-1);
                // rf_id=(gamma-Dch2Ach)*f_id;
                // arrF bg_d(bins.len-1);
                // draw_P_B_Tr(bg_d.data(),rf_id.data(),bins.len-1,t_diff.data(),bg_dd_rate ,devQStates+tidx);   
                // f_i=(f_if - bg_d - bg_a+0.5).floor();             
            }
            // float F=f_if.sum();
            // if(F>0){
            //     for (int s_trans=0;s_trans<bins.len-1;s_trans++){
                    // float de=drawE(gpe[*(sidx.at(s_trans))],r0,gpv[*(sidx.at(s_trans))],devQStates+tidx);
                    // long ai=drawA_fi_e(devStates+tidx, f_i(s_trans), de) ;
                    // mcE[idx*reSampleTimes+sampleTime]+=ai;
                    // mcE[tidx]+=ai;
            //     }
            //     mcE[tidx]/=F;
            mcE[N*sampleTime+tidx]=count;
            // }
            // sidx.freeList();
            // bins.freeList();
        }
    }    
}
void mc::set_gpuid(){
    CUDA_CHECK_RETURN(cudaSetDevice(devid));
}
mc::mc(int id,int _streamNum, bool de){    
    debug=de;
    streamNum=_streamNum;
    streams=(cudaStream_t*)malloc (sizeof(cudaStream_t)*streamNum);
    devid=id;
    set_gpuid();
    setAllocator("rmmDefaultPool");
    for(int sid=0;sid<_streamNum;sid++){
        CUDA_CHECK_RETURN(cudaStreamCreateWithFlags ( &(streams[sid]),cudaStreamNonBlocking) );
        streamFIFO.push(sid);
    }
    matK=NULL;matP=NULL;        
    // hpv=(float **)malloc(sizeof(float*)*streamNum);
    // hpk=(float **)malloc(sizeof(float*)*streamNum);
    // hpp=(float **)malloc(sizeof(float*)*streamNum);
    // hpe=(float **)malloc(sizeof(float*)*streamNum);    
    gpv=(float **)malloc(sizeof(float*)*streamNum);
    gpk=(float **)malloc(sizeof(float*)*streamNum);
    gpp=(float **)malloc(sizeof(float*)*streamNum);
    gpe=(float **)malloc(sizeof(float*)*streamNum);    
    // std::memset(hpv, 0, sizeof(float*)*streamNum);
    // std::memset(hpk, 0, sizeof(float*)*streamNum);
    // std::memset(hpp, 0, sizeof(float*)*streamNum);
    // std::memset(hpe, 0, sizeof(float*)*streamNum);    
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
    // hmcE=mcE=NULL;
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
    blockSize=256;
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

void mc::int_randstate(int N,int sid){
    int NN=N;
    if (N==-1){
        NN=sz_burst;//*reSampleTimes;
    }    
    // for (int sid=0;sid<streamNum;sid++){
        // CUDA_CHECK_RETURN(cudaFree ( devStates[sid]));
        // CUDA_CHECK_RETURN(cudaFree ( devQStates[sid]));    
        CUDA_CHECK_RETURN(_rmmReAlloc ( (void **)&(devStates[sid]), NN*sizeof (rk_state ), streams[sid] ));
        CUDA_CHECK_RETURN(_rmmReAlloc ( (void **)&(devQStates[sid]), NN*sizeof( curandStateScrambledSobol64) , streams[sid] ));    
        // CUDA_CHECK_RETURN(cudaFree (devDirectionVectors64[sid]));
        // CUDA_CHECK_RETURN(cudaFree (devScrambleConstants64[sid]));
        CUDA_CHECK_RETURN(_rmmReAlloc((void **)&(devDirectionVectors64[sid]), 
        NN * VECTOR_SIZE * sizeof(long long int), streams[sid] ));       
        CUDA_CHECK_RETURN(_rmmReAlloc((void **)&(devScrambleConstants64[sid]), 
        NN * sizeof(long long int), streams[sid] ));
        
        curandStatus_t curandResult =curandGetDirectionVectors64( &(hostVectors64[sid]), 
            CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6);
        if (curandResult != CURAND_STATUS_SUCCESS)
        {
            string msg("Could not get direction vectors for quasi-random number generator: ");
            msg += curandResult;
            throw std::runtime_error(msg);
        }    
        curandResult=curandGetScrambleConstants64( &(hostScrambleConstants64[sid])); 
        if (curandResult != CURAND_STATUS_SUCCESS)
        {
            string msg("Could not get direction vectors for quasi-random number generator: ");
            msg += curandResult;
            throw std::runtime_error(msg);
        }
        CUDA_CHECK_RETURN(cudaMemcpyAsync(devDirectionVectors64[sid],
             hostVectors64[sid],
            NN * VECTOR_SIZE * sizeof(long long int), 
            cudaMemcpyHostToDevice,streams[sid])); 
        CUDA_CHECK_RETURN(cudaMemcpyAsync(devScrambleConstants64[sid], hostScrambleConstants64[sid],
        NN * sizeof(long long int), 
        cudaMemcpyHostToDevice,streams[sid]));
        gridSize[sid] = (NN + blockSize - 1) / blockSize;
        
        setup_kernel <<<gridSize[sid], blockSize,0,streams[sid]>>>(devStates[sid], 0,/*time(NULL)*/ NN,        
            devDirectionVectors64[sid], devScrambleConstants64[sid], devQStates[sid]);    
        //CUDAstream_CHECK_LAST_ERROR;
        // CUDA_CHECK_RETURN(cudaStreamSynchronize(streams[sid]));
    // }
}

void mc::init_data_gpu(vector<int64_t>& start,vector<int64_t>& stop,
        vector<uint32_t>& istart,vector<uint32_t>& istop,
        vector<int64_t>& times_ms,
        vector<unsigned char>& mask_ad,vector<unsigned char>& mask_dd,
        vector<float>& T_burst_duration,vector<float>& SgDivSr,
        float& iclk_p,float& ibg_ad_rate,float& ibg_dd_rate){    
    clk_p=iclk_p;bg_ad_rate=ibg_ad_rate;bg_dd_rate=ibg_dd_rate;    
    sz_tag=mask_ad.size();                    
    // CUDA_CHECK_RETURN(cudaHostAlloc((void **)&hchi2, sizeof(float)));
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
        gridSize[sid] = (N + blockSize - 1) / blockSize;
        if (debug)
            cout<<gridSize[sid]<<" g "<<N*reSampleTimes<<" tN "<<blockSize<<" bS "<<mcE[sid]<<endl;
        
        // CUDA_CHECK_RETURN(cudaFree((void*)mcE[sid]));
        if (debug)
            cout<<"mcE[sid]"<<sid<<":"<<mcE[sid]<<endl;
        CUDA_CHECK_RETURN(_rmmReAlloc((void **)&(mcE[sid]), N *reSampleTimes* sizeof(retype),streams[sid]));   
        if (debug)  
            cout<<"mcE[sid]"<<sid<<":"<<mcE[sid]<<endl;   
        CUDA_CHECK_RETURN(cudaFreeHost((void*)hmcE[sid]));
        CUDA_CHECK_RETURN(cudaHostAlloc((void **)&(hmcE[sid]), N *reSampleTimes* sizeof(retype),cudaHostAllocDefault));
    }    
    CUDA_CHECK_RETURN(cudaMemsetAsync(mcE[sid], 0, N *reSampleTimes* sizeof(retype),streams[sid]));
    // CUDA_CHECK_RETURN(cudaStreamSynchronize(streams[sid]));
    return N;    
}

void mc::run_kernel(int N, int sid){     
    if (debug)
        cout<<"mcE[sid]"<<sid<<":"<<mcE[sid]<<endl;   
    mc_kernel<<<gridSize[sid],blockSize,0,streams[sid]>>>(g_start,g_stop,            
        g_istart,g_istop,
        g_times_ms,
        g_mask_ad,g_mask_dd,
        g_burst_duration,/*g_SgDivSr,*/
        clk_p,bg_ad_rate,bg_dd_rate,sz_tag,sz_burst ,
        gpe[sid],gpv[sid],gpk[sid],gpp[sid],N,s_n[sid],
        devQStates[sid],devStates[sid], mcE[sid], reSampleTimes/*,ti*/);
    // CUDAstream_CHECK_LAST_ERROR;
    // CUDA_CHECK_RETURN(cudaStreamSynchronize(streams[sid]));
    if (debug){
        cout<<"sid:"<<sid<<endl;
        cout<<"hmcE[sid]"<<sid<<":"<<hmcE[sid]<<endl;   
    }
    CUDA_CHECK_RETURN(cudaMemcpyAsync(hmcE[sid], mcE[sid],N * reSampleTimes*sizeof(retype), 
        cudaMemcpyDeviceToHost,streams[sid]));

    CUDA_CHECK_RETURN(cudaStreamSynchronize(streams[sid]));
    if (debug){
        std::vector<retype> my_vector(hmcE[sid], hmcE[sid] + N*reSampleTimes   );
        auto maxPosition = max_element(std::begin(my_vector), std::end(my_vector));
        for (int ip=0;ip<10;ip++)
            printf("%2.4f \t",*(hmcE[sid]+ip));
        cout<<endl<<sid<<"======"<<my_vector.size()<<"\n";
        cout<<*maxPosition<<","<<accumulate( my_vector.begin(), my_vector.end(), 0.0)/my_vector.size() <<endl;
        // savehdf5("r.hdf5", "/r",my_vector);
    }
}

mc::~mc(){
    free_data_gpu();    
    delete(matK);delete(matP);
    for(int sid=0;sid<streamNum;sid++){
        cudaStreamSynchronize(streams[sid]);
        cudaStreamDestroy ( streams[sid]);
    }

    free(streams);
    delete[](s_n);
    delete[](begin_burst);
    delete[](end_burst);
    delete[](gridSize);
    free(hmcE);
    free(mcE);
    // free(hpe);
    // free(hpv);
    // free(hpp);
    // free(hpk);    
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
    cudaDeviceReset();
 
}

void mc::free_data_gpu(){            
    // cudaDeviceSynchronize();
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
        cudaStreamSynchronize(streams[sid]);
        CUDA_CHECK_RETURN(_rmmFree(gpe[sid],streams[sid]));
        CUDA_CHECK_RETURN(_rmmFree(gpv[sid],streams[sid]));
        CUDA_CHECK_RETURN(_rmmFree(gpp[sid],streams[sid]));
        CUDA_CHECK_RETURN(_rmmFree(gpk[sid],streams[sid]));    

        CUDA_CHECK_RETURN(_rmmFree(mcE[sid],streams[sid]));
        
        CUDA_CHECK_RETURN(_rmmFree(devStates[sid],streams[sid]));
        CUDA_CHECK_RETURN(_rmmFree(devQStates[sid],streams[sid]));
        // CUDA_CHECK_RETURN(cudaFreeHost(hostVectors64[sid]));
        // CUDA_CHECK_RETURN(cudaFreeHost(hostScrambleConstants64[sid]));
        CUDA_CHECK_RETURN(_rmmFree(devDirectionVectors64[sid],streams[sid]));
        CUDA_CHECK_RETURN(_rmmFree(devScrambleConstants64[sid],streams[sid]));   
        cout<<"free sid"<<sid<<endl;
        // CUDA_CHECK_RETURN(cudaFreeHost(hpe[sid]));
        // cout<<"free sid"<<sid<<endl;
        // CUDA_CHECK_RETURN(cudaFreeHost(hpv[sid]));
        // CUDA_CHECK_RETURN(cudaFreeHost(hpp[sid]));
        // CUDA_CHECK_RETURN(cudaFreeHost(hpk[sid]));
        CUDA_CHECK_RETURN(cudaFreeHost(hmcE[sid]));                 
    }

}

bool mc::set_nstates(int n,int sid){
    bool r=false;
    if (s_n[sid]!=n){
        s_n[sid]=n;
        r=true;
        cudaStreamSynchronize(streams[sid]);
        // CUDA_CHECK_RETURN(cudaFreeHost(hpe[sid]));
        // CUDA_CHECK_RETURN(cudaFreeHost(hpv[sid]));
        // CUDA_CHECK_RETURN(cudaFreeHost(hpp[sid]));
        // CUDA_CHECK_RETURN(cudaFreeHost(hpk[sid]));
        // // CUDA_CHECK_RETURN(cudaFree(gpe[sid]));
        // // CUDA_CHECK_RETURN(cudaFree(gpv[sid]));
        // // CUDA_CHECK_RETURN(cudaFree(gpp[sid]));
        // // CUDA_CHECK_RETURN(cudaFree(gpk[sid]));    
        // CUDA_CHECK_RETURN(cudaHostAlloc((void **)&(hpe[sid]), n*sizeof(float),cudaHostAllocDefault));
        // CUDA_CHECK_RETURN(cudaHostAlloc((void **)&(hpv[sid]), n*sizeof(float),cudaHostAllocDefault));
        // CUDA_CHECK_RETURN(cudaHostAlloc((void **)&(hpp[sid]), n*sizeof(float),cudaHostAllocDefault));
        // CUDA_CHECK_RETURN(cudaHostAlloc((void **)&(hpk[sid]), n*n*sizeof(float),cudaHostAllocDefault));
        CUDA_CHECK_RETURN(_rmmReAlloc((void **)&(gpe[sid]), n*sizeof(float),streams[sid]));
        CUDA_CHECK_RETURN(_rmmReAlloc((void **)&(gpv[sid]), n*sizeof(float),streams[sid]));
        CUDA_CHECK_RETURN(_rmmReAlloc((void **)&(gpp[sid]), n*sizeof(float),streams[sid]));
        CUDA_CHECK_RETURN(_rmmReAlloc((void **)&(gpk[sid]), n*n*sizeof(float),streams[sid]));   
        int_randstate(n,sid);
    }
    return r;
}

bool mc::set_params(int n,int sid,vector<float>& args){
    // set_nstates(n,sid);    
    vecFloatMapper evargs(args.data(),n*n+n);        
    RowVectorXf eargs=evargs(seqN(0,n));
    float *peargs=eargs.data();
    RowVectorXf kargs=evargs(seqN(n,n*n-n));    
    RowVectorXf vargs=evargs(seqN(n*n,n));    
    float *pvargs=vargs.data();
    bool r=genMatK(&matK,n,kargs);
    //&matK不可修改，但是matK的值可以修改    
    r=r&&genMatP(&matP,matK);    
    cout<<"k:"<<*matK<<endl;
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
    // CUDA_CHECK_RETURN(cudaStreamSynchronize(streams[sid]));
    return r;
}