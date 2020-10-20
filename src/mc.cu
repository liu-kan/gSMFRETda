#include "cuda_tools.hpp"
#include "mc.hpp"
#include <cstddef>
#include <time.h>

#define VECTOR_SIZE 64
#include "binom.cuh"
#include "gen_rand.cuh"
#include <algorithm>
#include <cuda_profiler_api.h>
#include <numeric>

// #include "cuList.cuh"
// #include "rmm.hpp"
#include "tools.hpp"
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <cuda_runtime_api.h>

void CUDART_CB myStreamCallback(cudaStream_t stream, cudaError_t status,
                                void *data) {
    if (status) {
        std::cout << "Error: " << cudaGetErrorString(status) << "-->";
    }
}

int showGPUsInfo(int dn, char *gpuuid, int *streamCount) {
    int nDevices, i, n_Devices;
    checkCudaErrors(cudaGetDeviceCount(&nDevices));
    if (dn >= 0) {
        n_Devices = dn + 1;
        i = dn;
    } else {
        i = 0;
        n_Devices = nDevices;
    }
    if (dn < nDevices) {
        for (; i < n_Devices; i++) {
            cudaDeviceProp prop;
            checkCudaErrors(cudaGetDeviceProperties(&prop, i));
            printf("Device Number: %d\n", i);
            if (gpuuid) {
#if (CUDART_VERSION < 10000)
                boost::uuids::uuid a_uuid = boost::uuids::random_generator()();
                memcpy(gpuuid, &a_uuid, 16);
#else
                memcpy(gpuuid, &(prop.uuid.bytes[0]), 16);
#endif
            }
#if (CUDART_VERSION >= 10000)
            printf("  Device UUID: ");
            for (int i = 0; i < 16; i++) {
                printf("%hhx", prop.uuid.bytes[i]);
            }
#endif
            printf("\n  Concurrent copy and kernel execution:          %s with "
                   "%d copy "
                   "engine(s)\n",
                   (prop.deviceOverlap ? "Yes" : "No"), prop.asyncEngineCount);
            if (streamCount) {
                if (!prop.deviceOverlap)
                    *streamCount = 1;
                else
                    *streamCount = prop.asyncEngineCount * 2;
            }
            printf("  Device name: %s\n", prop.name);
            /*
            printf("  Memory Clock Rate (KHz): %d\n",
                    prop.memoryClockRate);
            printf("  Memory Bus Width (bits): %d\n",
                    prop.memoryBusWidth);
            printf("  Peak Memory Bandwidth (GB/s): %f\n",
                    2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
            */
            printf("  GPU global memory = %lu GBytes\n",
                   (prop.totalGlobalMem >> 30) + 1);
        }
    }
    return nDevices;
}
#define CUDAstream_CHECK_LAST_ERROR                                                 \
    cudaStreamAddCallback(streams[sid], myStreamCallback, nullptr, 0)

//__forceinline__
// template <typename T>
// __device__ void binTimeHist(arrF* hist, arrI64& x,
//          cuList<T> bins ){
//     int binlen=bins.len;
//     hist->resize(1,binlen-1);
//     hist->setZero();
//     int datalen=x.cols();
//     for (int i=0;i<datalen;i++){
//         if(x(i)==0)
//             continue;
//         int idxbin=1;
//         do{
//             T v=*(bins.at(idxbin));
//             if (x(i)<v){
//                 ((*hist)(idxbin-1))+=1;
//                 break;
//             }
//             idxbin++;
//         }while(idxbin<binlen);
//     }
// }
/*
input:x 时间序列, bin0/1开始和结束时间点，x0起始搜索idx
output:hist 保存hist值，x0保存下次的起始位置
*/
// template <typename T>
// __device__ void p2TimeHist(float* hist, arrI64& x,
//          T bin0, T bin1 , int64_t* x0){
//     *hist=0;
//     int datalen=x.cols();
//     for (int i=*x0;i<datalen;i++){
//         if(x(i)==0)
//             continue;
//         if (x(i)<bin0)
//             continue;
//         else if(x(i)<bin1){
//             *hist=(*hist)+1;
//         }
//         else{
//             break;
//             *x0=i;
//         }
//     }
// }
__global__ void
mc_kernel(int64_t *start, int64_t *stop, int64_t *g_burst_ad, int64_t *g_burst_dd,
          int64_t *g_istart, int *g_phCount, float *T, /*float* SgDivSr,*/
          float clk_p, float bg_ad_rate, float bg_dd_rate, float *gpe, float *gpv,
          float *gpk, float *gpp, float *P_i2j, int N, int s_n,
          curandStateScrambledSobol64 *devQStates, rk_state *devStates, retype *mcE,
          int reSampleTimes, unsigned char debug = 0, float gamma = 0.34,
          float beta = 1.42, float DexDirAem = 0.08, float Dch2Ach = 0.07,
          float r0 = 52) {
    int NN = N * reSampleTimes;
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx < NN) {
        // idx is burst id
        // int idx=tidx;//%reSampleTimes;
        // int idx = __fdiv_rd(tidx,reSampleTimes);
        // int sampleTime=tidx%reSampleTimes;
        // If n is a power of 2, ( i / n ) is equivalent to ( i ≫ log2 n ) and (
        // i % n ) is equivalent to ( i & n - 1 ).
        int idx = tidx >> ((int)log2(reSampleTimes));
        // int sampleTime = tidx & reSampleTimes - 1;
        int phCount = g_phCount[idx];
        int64_t *burst_ad = g_burst_ad + g_istart[idx];
        int64_t *burst_dd = g_burst_dd + g_istart[idx];
        float F = 0;
        mcE[tidx] = 0;
        // for (int sampleTime=0;sampleTime<reSampleTimes;sampleTime++){
        // // int sampleTime=tidx/reSampleTimes;
        int si = drawDisIdx(s_n, gpp, devQStates + tidx);
        float mcSpendTime = 0;
        matXfMapper matKmp(gpk, s_n, s_n);
        int count = 0;
        int64_t bin0clk_t = start[idx];
        int64_t bin1clk_t = start[idx];
        int sj = 0, binIdxStart = 0;
        while (T[idx] > mcSpendTime) {
            sj = drawJ_Si2Sj(P_i2j + tidx * s_n, gpk, s_n, si, devQStates + tidx);
            // if(si==0 && sj==2&& tidx<100)
            //     printf("s_n=%d,sj=%d,gpp=%f, %f, %f,
            //     tidx=%d\n",s_n,sj,gpp[0],gpp[1],gpp[2],tidx);
            float st = drawTau(matKmp(si, sj), devQStates + tidx, 0);
            // if(si==0 && sj==2&& tidx<100)
            //     printf("drawTau=%f\n",st);
            mcSpendTime = mcSpendTime + st;
            // si=sj;
            if (mcSpendTime >= T[idx]) {
                //     bins.append(stop[idx]);
                bin1clk_t = stop[idx];
            } else {
                //     bins.append(*(bins.at(0))+mcSpendTime/clk_p);
                bin1clk_t = bin0clk_t + mcSpendTime / clk_p;
            }
            // [bin0clk_t bin1clk_t) is the clk timing range,
            // Then try to get the ad and dd count in this range.
            bool sdd = false, sad = false, bdd = false, bad = false;
            int f_id = 0, f_ia = 0;
            int64_t ddx, adx;
            long ai = 0;
            for (int iinb = binIdxStart; iinb < phCount; iinb++) {
                ddx = burst_dd[iinb];
                adx = burst_ad[iinb];
                // if(debug)
                // if(idx==200)
                //     printf("ddx= %ld, adx= %ld\n",ddx,adx);

                if (ddx >= bin1clk_t || adx >= bin1clk_t || iinb == phCount - 1) {
                    binIdxStart = iinb;
                    // calac F
                    float f_if = (gamma - Dch2Ach) * f_id + (1 - DexDirAem) * f_ia;
                    float f_i = 0;
                    F += f_if;
                    if (bg_dd_rate < 1e-4) {
                        f_i = floorf(f_if + 0.5);
                    } else {
                        float t_diff = (bin1clk_t - bin0clk_t) * clk_p;
                        float rf_ia = (1 - DexDirAem) * f_ia;
                        float bg_a;
                        draw_P_B_Tr(&bg_a, &rf_ia, 1, &t_diff, bg_ad_rate,
                                    devQStates + tidx);
                        float rf_id = (gamma - Dch2Ach) * f_id;
                        float bg_d;
                        draw_P_B_Tr(&bg_d, &rf_id, 1, &t_diff, bg_dd_rate,
                                    devQStates + tidx);
                        f_i = floorf(f_if - bg_d - bg_a + 0.5);
                    }
                    float de = drawE(gpe[si], r0, gpv[si], devQStates + tidx);
                    ai = drawA_fi_e(devStates + tidx, f_i, de);
                    break;
                }
                if (ddx > 0)
                    sdd = true;
                if (adx > 0)
                    sad = true;
                if (sad && sdd)
                    continue;
                if (sad && !bad)
                    f_ia++;
                if (sdd && !bdd)
                    f_id++;
            }
            mcE[tidx] += ai;
            count++;
            bin0clk_t = bin1clk_t;
            si = sj;
#define __count__ 3
            if (debug)
                if (count > __count__)
                    printf("burst id %d trans %d > %d. clk_p= %g, mcE[%d]= %g\n",
                           idx, count, __count__, clk_p, tidx, mcE[tidx]);
        }
        if (F > 0)
            mcE[tidx] = mcE[tidx] / F;
        else
            mcE[tidx] = 0;
    }
}
/**
 * @brief  Actual Setup of GPUID, if you have multi gpu. 
 * According to [nvidia blog](https://developer.nvidia.com/blog/cuda-pro-tip-always-set-current-device-avoid-multithreading-bugs/) this fuction must *Always* be called when a new host threads need to call cu kernel.
 * 
 */
void mc::set_gpuid() {
    cout<<"Thread id: "<<std::this_thread::get_id()<<" cudaSetDevice: "<<devid<<endl;
    CUDA_CHECK_RETURN(cudaSetDevice(devid));
    if (profiler) {
        std::cout << "cudaProfilerStart" << std::endl;
        checkCudaErrors(cudaProfilerStart());
    }
}
mc::mc(int id, int _streamNum, unsigned char de, std::uintmax_t hdf5size,
       bool _profiler) {
    debug = de;
    profiler = _profiler;
    devid = id;
    cudaGetDeviceCount(&nDevices);
    if (_streamNum == 0) {
        nDevices = showGPUsInfo(devid, gpuuuid, &streamNum);
        printf("streamCount is automatically determined as %d\n", streamNum);
    } else {
        nDevices = showGPUsInfo(devid, gpuuuid);
        streamNum = _streamNum;
    }
    workerNum.store(streamNum);
    if (devid >= nDevices || devid < 0) {
        std::cout << "gpu id set error!" << std::endl;
        return;
    }
    set_gpuid();
    mr = new mrImp(hdf5size, 0.85, devid,false,0);
    // setAllocator("rmmDefaultPool");
    streams = (cudaStream_t *)malloc(sizeof(cudaStream_t) * streamNum);
    for (int sid = 0; sid < streamNum; sid++) {
        CUDA_CHECK_RETURN(
            cudaStreamCreateWithFlags(&(streams[sid]), cudaStreamNonBlocking));
        // CUDA_CHECK_RETURN(cudaStreamCreateWithFlags (
        // &(streams[sid]),cudaStreamDefault) );
        streamFIFO.push(sid);
    }
    matK = NULL;
    matP = NULL;
    // hpv=(float **)malloc(sizeof(float*)*streamNum);
    // hpk=(float **)malloc(sizeof(float*)*streamNum);
    // hpp=(float **)malloc(sizeof(float*)*streamNum);
    // hpe=(float **)malloc(sizeof(float*)*streamNum);
    gpv = (float **)malloc(sizeof(float *) * streamNum);
    gpk = (float **)malloc(sizeof(float *) * streamNum);
    gpp = (float **)malloc(sizeof(float *) * streamNum);
    gpe = (float **)malloc(sizeof(float *) * streamNum);
    g_P_i2j = (float **)malloc(sizeof(float *) * streamNum);
    matK=(arrFF**)malloc(sizeof(arrFF *) * streamNum);
    matP=(arrF**)malloc(sizeof(arrF *) * streamNum);
    oldN = new int[streamNum];
    std::fill_n(oldN, streamNum, 0);
    // std::memset(hpv, 0, sizeof(float*)*streamNum);
    // std::memset(hpk, 0, sizeof(float*)*streamNum);
    // std::memset(hpp, 0, sizeof(float*)*streamNum);
    // std::memset(hpe, 0, sizeof(float*)*streamNum);
    std::memset(matP, 0, sizeof(arrF *) * streamNum);
    std::memset(matK, 0, sizeof(arrFF *) * streamNum);
    std::memset(gpv, 0, sizeof(float *) * streamNum);
    std::memset(gpk, 0, sizeof(float *) * streamNum);
    std::memset(gpp, 0, sizeof(float *) * streamNum);
    std::memset(gpe, 0, sizeof(float *) * streamNum);
    s_n = new int[streamNum];
    gridSize = new int[streamNum];
    begin_burst = new int[streamNum];
    end_burst = new int[streamNum];
    std::fill_n(s_n, streamNum, 0);
    std::fill_n(gridSize, streamNum, 0);
    std::fill_n(begin_burst, streamNum, 0);
    std::fill_n(end_burst, streamNum, 0);
    mcE = (retype **)malloc(sizeof(retype *) * streamNum);
    hmcE = (retype **)malloc(sizeof(retype *) * streamNum);
    std::memset(mcE, 0, sizeof(retype *) * streamNum);
    std::memset(hmcE, 0, sizeof(retype *) * streamNum);
    // hmcE=mcE=NULL;
    devStates = (rk_state **)malloc(sizeof(rk_state *) * streamNum);
    devQStates = (curandStateScrambledSobol64 **)malloc(
        sizeof(curandStateScrambledSobol64 *) * streamNum);
    // hostVectors64=(curandDirectionVectors64_t**)malloc(sizeof(curandDirectionVectors64_t*)*streamNum);
    // hostScrambleConstants64=(unsigned long long int**)malloc(sizeof(unsigned
    // long long int*)*streamNum);
    devDirectionVectors64 = (unsigned long long int **)malloc(
        sizeof(unsigned long long int *) * streamNum);
    devScrambleConstants64 = (unsigned long long int **)malloc(
        sizeof(unsigned long long int *) * streamNum);
    std::memset(devStates, 0, sizeof(rk_state *) * streamNum);
    std::memset(devQStates, 0, sizeof(curandStateScrambledSobol64 *) * streamNum);
    // std::memset(hostVectors64, 0,
    // sizeof(curandDirectionVectors64_t*)*streamNum);
    std::memset(devDirectionVectors64, 0,
                sizeof(unsigned long long int *) * streamNum);
    // std::memset(hostScrambleConstants64, 0, sizeof(unsigned long long
    // int*)*streamNum);
    std::memset(devScrambleConstants64, 0,
                sizeof(unsigned long long int *) * streamNum);
    reSampleTimes = 4;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, mc_kernel, 0, 0);
    // blockSize=128;
    printf("blockSize = %d\n", blockSize);
    CURAND_CALL(curandGetDirectionVectors64(
        &hostVectors64, CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6));
    CURAND_CALL(curandGetScrambleConstants64(&hostScrambleConstants64));
}
void mc::set_reSampleTimes(int t) { reSampleTimes = pow(2, (int)log2(t)); }
int mc::getStream() {
    int r = -1;
    if (!streamFIFO.empty()) {
        int r = streamFIFO.front();
        streamFIFO.pop();
    }
    return r;
}

void mc::givebackStream(int i) { streamFIFO.push(i); }

void mc::init_randstate(int N, int sid) {
    int NN;
    int oldNN = oldN[sid] * reSampleTimes;
    if (N == -1) {
        NN = sz_burst * reSampleTimes;
    } else {
        NN = N * reSampleTimes;
    }
    gridSize[sid] = (NN + blockSize - 1) / blockSize;
    // for (int sid=0;sid<streamNum;sid++){

    // cudaFree ( devStates[sid]) returned an illegal memory access was
    // encountered(700) at /home/liuk/data/proj/gSMFRETda/src/mc.cu:303
    // CUDA_CHECK_RETURN(cudaFree ( devStates[sid]));
    // CUDA_CHECK_RETURN(cudaFree ( devQStates[sid]));

    // printf("%d init_randstate, oldNN=%ld, devStates[sid]=%p\n", sid,
        //    oldNN * sizeof(rk_state), devStates[sid]);
    mr->free(devStates[sid], oldNN * sizeof(rk_state), streams[sid]);

    mr->free(devQStates[sid], oldNN * sizeof(curandStateScrambledSobol64),
             streams[sid]);
    devStates[sid] = (rk_state *)(mr->malloc(NN * sizeof(rk_state), streams[sid]));
    devQStates[sid] = (curandStateScrambledSobol64 *)(mr->malloc(
        NN * sizeof(curandStateScrambledSobol64), streams[sid]));
    // CUDA_CHECK_RETURN(cudaMalloc ( (void **)&(devStates[sid]), NN*sizeof
    // (rk_state ) )); CUDA_CHECK_RETURN(cudaMalloc ( (void
    // **)&(devQStates[sid]), NN*sizeof( curandStateScrambledSobol64)  ));

    // CUDA_CHECK_RETURN(cudaFree (devDirectionVectors64[sid]));
    // CUDA_CHECK_RETURN(cudaFree (devScrambleConstants64[sid]));
    mr->free(devDirectionVectors64[sid], oldNN * sizeof(curandDirectionVectors64_t),
             streams[sid]);
    mr->free(devScrambleConstants64[sid], oldNN * sizeof(unsigned long long int),
             streams[sid]);
    devDirectionVectors64[sid] = (unsigned long long int *)(mr->malloc(
        NN * VECTOR_SIZE * sizeof(long long int), streams[sid]));
    devScrambleConstants64[sid] = (unsigned long long int *)(mr->malloc(
        NN * sizeof(long long int), streams[sid]));
    // CUDA_CHECK_RETURN(cudaMalloc((void **)&(devDirectionVectors64[sid]),
    // NN * VECTOR_SIZE * sizeof(long long int)));
    // CUDA_CHECK_RETURN(cudaMalloc((void **)&(devScrambleConstants64[sid]),
    // NN * sizeof(long long int) ));
    /*
    curandStatus_t curandResult =curandGetDirectionVectors64(
    &(hostVectors64[sid]), CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6); if
    (curandResult != CURAND_STATUS_SUCCESS)
    {
        string msg("Could not get direction vectors for quasi-random number
    generator: "); msg += curandResult; throw std::runtime_error(msg);
    }
    curandResult=curandGetScrambleConstants64( &(hostScrambleConstants64[sid]));
    if (curandResult != CURAND_STATUS_SUCCESS)
    {
        string msg("Could not get direction vectors for quasi-random number
    generator: "); msg += curandResult; throw std::runtime_error(msg);
    }
    */
    // std::cout << sid << " devDirectionVectors64ing, size ="
    //           << NN * VECTOR_SIZE * sizeof(long long int) << std::endl;
    int n = 0;
    int tNN = NN;
    while (tNN > 0) {
        int size = (tNN > 20000) ? 20000 : tNN;
        unsigned long long int *buf = devScrambleConstants64[sid];
        CUDA_CHECK_RETURN(cudaMemcpyAsync(buf + n * 20000, hostScrambleConstants64,
                                          size * sizeof(unsigned long long int),
                                          cudaMemcpyHostToDevice, streams[sid]));
        // std::cout << "n = " << n << ", size = " << size << std::endl;
        buf = devDirectionVectors64[sid];
        CUDA_CHECK_RETURN(
            cudaMemcpyAsync(buf + n * 20000 * sizeof(curandDirectionVectors64_t) /
                                      sizeof(unsigned long long int),
                            hostVectors64, size * sizeof(curandDirectionVectors64_t),
                            cudaMemcpyHostToDevice, streams[sid]));

        tNN -= size;
        n++;
    }
    // std::cout << sid << " devDirectionVectors64ed \n";

    setup_kernel<<<gridSize[sid], blockSize, 0, streams[sid]>>>(
        devStates[sid], 0, /*time(NULL)*/ NN, devDirectionVectors64[sid],
        devScrambleConstants64[sid], devQStates[sid]);
    // CUDAstream_CHECK_LAST_ERROR;
    CUDA_CHECK_RETURN(cudaStreamSynchronize(streams[sid]));
    // }
    oldN[sid] = N;
}

void mc::init_data_gpu(vector<int64_t> &istart, vector<int64_t> &start,
                       vector<int64_t> &stop, std::vector<int> &phCount,
                       long _sz_tag, int64_t *burst_ad, int64_t *burst_dd,
                       vector<float> &T_burst_duration, vector<float> &SgDivSr,
                       float &iclk_p, float &ibg_ad_rate, float &ibg_dd_rate) {
    clk_p = iclk_p;
    bg_ad_rate = ibg_ad_rate;
    bg_dd_rate = ibg_dd_rate;
    sz_tag = _sz_tag;
    sz_burst = start.size();
    int sidx = 0;
    g_phCount =
        (int *)mr->malloc(sizeof(int) * sz_burst, streams[(sidx) % streamNum]);
    CUDA_CHECK_RETURN(cudaMemcpyAsync(g_phCount, phCount.data(),
                                      sizeof(int) * sz_burst, cudaMemcpyHostToDevice,
                                      streams[(sidx++) % streamNum]));
    g_burst_ad =
        (int64_t *)mr->malloc(sizeof(int64_t) * sz_tag, streams[(sidx) % streamNum]);
    CUDA_CHECK_RETURN(cudaMemcpyAsync(g_burst_ad, burst_ad, sizeof(int64_t) * sz_tag,
                                      cudaMemcpyHostToDevice,
                                      streams[(sidx++) % streamNum]));
    g_burst_dd =
        (int64_t *)mr->malloc(sizeof(int64_t) * sz_tag, streams[(sidx) % streamNum]);
    CUDA_CHECK_RETURN(cudaMemcpyAsync(g_burst_dd, burst_dd, sizeof(int64_t) * sz_tag,
                                      cudaMemcpyHostToDevice,
                                      streams[(sidx++) % streamNum]));

    g_istart = (int64_t *)mr->malloc(sizeof(int64_t) * sz_burst,
                                     streams[(sidx) % streamNum]);
    CUDA_CHECK_RETURN(
        cudaMemcpyAsync(g_istart, istart.data(), sizeof(int64_t) * sz_burst,
                        cudaMemcpyHostToDevice, streams[(sidx++) % streamNum]));
    // CUDA_CHECK_RETURN(cudaMalloc((void **)&g_start,
    // sizeof(int64_t)*sz_burst));
    g_start = (int64_t *)mr->malloc(sizeof(int64_t) * sz_burst,
                                    streams[(sidx) % streamNum]);
    CUDA_CHECK_RETURN(
        cudaMemcpyAsync(g_start, start.data(), sizeof(int64_t) * sz_burst,
                        cudaMemcpyHostToDevice, streams[(sidx++) % streamNum]));
    // CUDA_CHECK_RETURN(cudaMalloc((void **)&g_stop,
    // sizeof(int64_t)*sz_burst));
    g_stop = (int64_t *)mr->malloc(sizeof(int64_t) * sz_burst,
                                   streams[(sidx) % streamNum]);
    CUDA_CHECK_RETURN(
        cudaMemcpyAsync(g_stop, stop.data(), sizeof(int64_t) * sz_burst,
                        cudaMemcpyHostToDevice, streams[(sidx++) % streamNum]));

    // CUDA_CHECK_RETURN(cudaMalloc((void **)&g_burst_duration,
    // sizeof(float)*sz_burst));
    g_burst_duration =
        (float *)mr->malloc(sizeof(float) * sz_burst, streams[(sidx) % streamNum]);
    CUDA_CHECK_RETURN(cudaMemcpyAsync(
        g_burst_duration, T_burst_duration.data(), sizeof(float) * sz_burst,
        cudaMemcpyHostToDevice, streams[(sidx++) % streamNum]));
    // CUDA_CHECK_RETURN(cudaMalloc((void **)&g_SgDivSr,
    // sizeof(float)*sz_burst));
    g_SgDivSr =
        (float *)mr->malloc(sizeof(float) * sz_burst, streams[(sidx) % streamNum]);
    CUDA_CHECK_RETURN(
        cudaMemcpyAsync(g_SgDivSr, SgDivSr.data(), sizeof(float) * sz_burst,
                        cudaMemcpyHostToDevice, streams[(sidx++) % streamNum]));
    // CUDA_CHECK_RETURN(cudaMalloc((void **)&gchi2, sizeof(float)));
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
}

int mc::setBurstBd(int cstart, int cstop, int sid) {
    int rcstart = cstart;
    int rcstop = cstop;
    if (cstop >= sz_burst || cstop < 0)
        rcstop = sz_burst;
    if (cstart >= rcstop) {
        rcstart = 0;
    }
    int N = rcstop - rcstart;
    if (cstart == -1) {
        gridSize[sid] = -blockSize;
        N = -1;
    }
    if (end_burst[sid] - begin_burst[sid] != N) {
        begin_burst[sid] = rcstart;
        end_burst[sid] = rcstop;
        // int dimension=256;
        // dim3 threads = dim3(dimension, 1);
        // int blocksCount = ceil(float(N)/dimension);
        // dim3 blocks  = dim3(blocksCount, 1);
        gridSize[sid] = (N + blockSize - 1) / blockSize;
        // if (debug)
        //     cout << gridSize[sid] << " g " << N * reSampleTimes << " tN "
        //          << blockSize << " bS " << mcE[sid] << endl;

        // CUDA_CHECK_RETURN(cudaFree((void*)mcE[sid]));
        mr->free(mcE[sid], oldN[sid] * reSampleTimes * sizeof(retype), streams[sid]);
        // if (debug)
        //     cout << "mcE[sid]" << sid << ":" << mcE[sid] << endl;
        // CUDA_CHECK_RETURN(cudaMalloc((void **)&(mcE[sid]), N *reSampleTimes*
        // sizeof(retype)));
        mcE[sid] =
            (retype *)(mr->malloc(N * reSampleTimes * sizeof(retype), streams[sid]));

        // if (debug)
        //     cout << "mcE[sid]" << sid << ":" << mcE[sid] << endl;
        CUDA_CHECK_RETURN(cudaFreeHost((void *)hmcE[sid]));
        CUDA_CHECK_RETURN(cudaHostAlloc((void **)&(hmcE[sid]),
                                        N * reSampleTimes * sizeof(retype),
                                        cudaHostAllocDefault));
    }
    // if(nDevices>1)
    //     CUDA_CHECK_RETURN(cudaMemset(mcE[sid], 0, N *reSampleTimes*
    //     sizeof(retype)));
    // else
    // CUDA_CHECK_RETURN(cudaStreamSynchronize(streams[sid]));
    // CUDA_CHECK_RETURN(cudaMemsetAsync(mcE[sid], 0, N *reSampleTimes*
    // sizeof(retype),streams[sid]));
    // CUDA_CHECK_RETURN(cudaStreamSynchronize(streams[sid]));
    return N;
}

void mc::run_kernel(int N, int sid) {
    AtomicWriter(debug, debugLevel::gpu)
        << "mcE[sid]" << sid << ":" << mcE[sid] << "\n";

    mc_kernel<<<gridSize[sid], blockSize, 0, streams[sid]>>>(
        g_start, g_stop, g_burst_ad, g_burst_dd, g_istart, g_phCount,
        g_burst_duration, /*g_SgDivSr,*/
        clk_p, bg_ad_rate, bg_dd_rate, gpe[sid], gpv[sid], gpk[sid], gpp[sid],
        g_P_i2j[sid], N, s_n[sid], devQStates[sid], devStates[sid], mcE[sid],
        reSampleTimes, debug & debugLevel::kernel);
    // CUDAstream_CHECK_LAST_ERROR;
    // CUDA_CHECK_RETURN(cudaStreamSynchronize(streams[sid]));
    if (debug & debugLevel::gpu) {
        cout << "sid:" << sid << endl;
        cout << "streams[" << sid << "]:" << streams[sid] << endl;
        cout << "hmcE[sid]" << sid << ":" << hmcE[sid] << endl;
    }
    CUDA_CHECK_RETURN(cudaMemcpyAsync(hmcE[sid], mcE[sid],
                                      N * reSampleTimes * sizeof(retype),
                                      cudaMemcpyDeviceToHost, streams[sid]));
}
/**
 * @brief  Query if the stream finished
 *
 * @param sid
 * @return true
 * @return false
 */
bool mc::streamQuery(int sid) {
    if (sid < 0 || sid >= streamNum)
        return false;
    if (cudaStreamQuery(streams[sid]) == cudaSuccess)
        return true;
    return false;
}
void mc::get_res(int sid, int N) {
    // CUDA_CHECK_RETURN(cudaStreamSynchronize(streams[sid]));
    if (debug) {
        std::vector<retype> my_vector(hmcE[sid], hmcE[sid] + N * reSampleTimes);
        auto maxPosition = max_element(std::begin(my_vector), std::end(my_vector));
        for (int ip = 0; ip < 10; ip++)
            printf("%2.4f \t", *(hmcE[sid] + ip));
        cout << endl << sid << "======" << my_vector.size() << "\n";
        cout << *maxPosition << ","
             << accumulate(my_vector.begin(), my_vector.end(), 0.0) /
                    my_vector.size()
             << endl;
        // savehdf5("r.hdf5", "/r",my_vector);
    }
}

mc::~mc() {
    free_data_gpu();
    free (matK);
    free (matP);
    delete (mr);
    for (int sid = 0; sid < streamNum; sid++) {
        checkCudaErrors(cudaStreamSynchronize(streams[sid]));
        checkCudaErrors(cudaStreamDestroy(streams[sid]));
    }
    free(streams);
    delete[](s_n);
    delete[](oldN);
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
    free(g_P_i2j);
    // free(hostVectors64);
    // free(hostScrambleConstants64);

    free(devStates);
    free(devQStates);
    free(devDirectionVectors64);
    free(devScrambleConstants64);
    if (profiler) {
        checkCudaErrors(cudaProfilerStop());
        std::cout << "cudaProfilerStop" << std::endl;
    }
    cudaDeviceReset();
}

void mc::free_data_gpu() {
    // cudaDeviceSynchronize();

    // // CUDA_CHECK_RETURN(cudaFree(g_mask_ad));
    // // CUDA_CHECK_RETURN(cudaFree(g_mask_dd));
    // CUDA_CHECK_RETURN(cudaFree(g_start));
    // CUDA_CHECK_RETURN(cudaFree(g_stop));
    // // CUDA_CHECK_RETURN(cudaFree(g_istart));
    // // CUDA_CHECK_RETURN(cudaFree(g_istop));
    // // CUDA_CHECK_RETURN(cudaFree(g_times_ms));
    // CUDA_CHECK_RETURN(cudaFree(g_SgDivSr));
    // CUDA_CHECK_RETURN(cudaFree(g_burst_duration));
    int sidx = 0;
    checkCudaErrors(cudaDeviceSynchronize());
    mr->free(g_phCount, sizeof(int) * sz_burst, streams[(sidx++) % streamNum]);
    mr->free(g_burst_ad, sizeof(int64_t) * sz_tag, streams[(sidx++) % streamNum]);
    mr->free(g_burst_dd, sizeof(int64_t) * sz_tag, streams[(sidx++) % streamNum]);
    mr->free(g_start, sizeof(int64_t) * sz_burst, streams[(sidx++) % streamNum]);
    mr->free(g_istart, sizeof(int64_t) * sz_burst, streams[(sidx++) % streamNum]);
    mr->free(g_stop, sizeof(int64_t) * sz_burst, streams[(sidx++) % streamNum]);
    mr->free(g_burst_duration, sizeof(float) * sz_burst,
             streams[(sidx++) % streamNum]);
    mr->free(g_SgDivSr, sizeof(float) * sz_burst, streams[(sidx++) % streamNum]);
    checkCudaErrors(cudaDeviceSynchronize());

    for (int sid = 0; sid < streamNum; sid++) {
       delete(matK[sid]);
       delete(matP[sid]);
        // cudaStreamSynchronize(streams[sid]);
        // CUDA_CHECK_RETURN(cudaFree(gpe[sid]));
        // CUDA_CHECK_RETURN(cudaFree(gpv[sid]));
        // CUDA_CHECK_RETURN(cudaFree(gpp[sid]));
        // CUDA_CHECK_RETURN(cudaFree(gpk[sid]));
        mr->free(gpe[sid], s_n[sid] * sizeof(float), streams[sid]);
        mr->free(gpv[sid], s_n[sid] * sizeof(float), streams[sid]);
        mr->free(gpp[sid], s_n[sid] * sizeof(float), streams[sid]);
        mr->free(gpk[sid], s_n[sid] * sizeof(float), streams[sid]);

        // CUDA_CHECK_RETURN(cudaFree(mcE[sid]));
        int oldNN = oldN[sid] * reSampleTimes;
        mr->free(mcE[sid], oldNN * sizeof(retype), streams[sid]);
        mr->free(g_P_i2j[sid], s_n[sid] * sizeof(float) * oldNN, streams[sid]);
        // CUDA_CHECK_RETURN(cudaFree(devStates[sid]));
        // CUDA_CHECK_RETURN(cudaFree(devQStates[sid]));
        // CUDA_CHECK_RETURN(cudaFree(devDirectionVectors64[sid]));
        // CUDA_CHECK_RETURN(cudaFree(devScrambleConstants64[sid]));
        mr->free(devStates[sid], oldNN * sizeof(rk_state), streams[sid]);
        mr->free(devQStates[sid], oldNN * sizeof(curandStateScrambledSobol64),
                 streams[sid]);
        mr->free(devDirectionVectors64[sid],
                 oldNN * VECTOR_SIZE * sizeof(long long int), streams[sid]);
        mr->free(devScrambleConstants64[sid], oldNN * sizeof(long long int),
                 streams[sid]);

        cout << "free sid" << sid << endl;
        CUDA_CHECK_RETURN(cudaFreeHost(hmcE[sid]));
    }
}
/**
 * @brief               Setup the number of protien's states in simulation.
 *
 * @param n         The number of protien's states
 * @param sid      The gpu stream idx
 * @return int       The number of protien's states setuped in the gpu stream
 * idx.
 */
int mc::set_nstates(int n, int sid) {
    int r = n;
    if (s_n[sid] != n) {
        r = s_n[sid];
        checkCudaErrors(cudaStreamSynchronize(streams[sid]));
        // std::cout << "checkCudaErrors( cudaStreamSynchronize(streams[" << sid
        //           << "])\n";
        // CUDA_CHECK_RETURN(cudaFreeHost(hpe[sid]));
        // CUDA_CHECK_RETURN(cudaFreeHost(hpv[sid]));
        // CUDA_CHECK_RETURN(cudaFreeHost(hpp[sid]));
        // CUDA_CHECK_RETURN(cudaFreeHost(hpk[sid]));
        // // CUDA_CHECK_RETURN(cudaFree(gpe[sid]));
        // // CUDA_CHECK_RETURN(cudaFree(gpv[sid]));
        // // CUDA_CHECK_RETURN(cudaFree(gpp[sid]));
        // // CUDA_CHECK_RETURN(cudaFree(gpk[sid]));
        // CUDA_CHECK_RETURN(cudaHostAlloc((void **)&(hpe[sid]),
        // n*sizeof(float),cudaHostAllocDefault));
        // CUDA_CHECK_RETURN(cudaHostAlloc((void **)&(hpv[sid]),
        // n*sizeof(float),cudaHostAllocDefault));
        // CUDA_CHECK_RETURN(cudaHostAlloc((void **)&(hpp[sid]),
        // n*sizeof(float),cudaHostAllocDefault));
        // CUDA_CHECK_RETURN(cudaHostAlloc((void **)&(hpk[sid]),
        // n*n*sizeof(float),cudaHostAllocDefault)); if (gpe[sid]!=0)
        // CUDA_CHECK_RETURN(cudaFree(gpe[sid])); if (gpv[sid]!=0)
        // CUDA_CHECK_RETURN(cudaFree(gpv[sid])); if (gpp[sid]!=0)
        // CUDA_CHECK_RETURN(cudaFree(gpp[sid])); if (gpk[sid]!=0)
        // CUDA_CHECK_RETURN(cudaFree(gpk[sid]));
        mr->free(gpe[sid], s_n[sid] * sizeof(float), streams[sid]);
        mr->free(gpv[sid], s_n[sid] * sizeof(float), streams[sid]);
        mr->free(gpp[sid], s_n[sid] * sizeof(float), streams[sid]);
        mr->free(gpk[sid], s_n[sid] * sizeof(float), streams[sid]);
        // std::cout << "&(gpe[" << sid << "]):" << (void **)&(gpe[sid]) << "\tn:" << n
        //           << std::endl;
        gpe[sid] = (float *)(mr->malloc(n * sizeof(float), streams[sid]));
        gpv[sid] = (float *)(mr->malloc(n * sizeof(float), streams[sid]));
        gpp[sid] = (float *)(mr->malloc(n * sizeof(float), streams[sid]));
        gpk[sid] = (float *)(mr->malloc(n * n* sizeof(float), streams[sid]));
        // CUDA_CHECK_RETURN(cudaMalloc((void **)&(gpe[sid]), n*sizeof(float)));
        // CUDA_CHECK_RETURN(cudaMalloc((void **)&(gpv[sid]), n*sizeof(float)));
        // CUDA_CHECK_RETURN(cudaMalloc((void **)&(gpp[sid]), n*sizeof(float)));
        // CUDA_CHECK_RETURN(cudaMalloc((void **)&(gpk[sid]),
        // n*n*sizeof(float)));

        // CUDA_CHECK_RETURN(_rmmReAlloc((void **)&(gpe[sid]),
        // n*sizeof(float),streams[sid])); CUDA_CHECK_RETURN(_rmmReAlloc((void
        // **)&(gpv[sid]), n*sizeof(float),streams[sid]));
        // CUDA_CHECK_RETURN(_rmmReAlloc((void **)&(gpp[sid]),
        // n*sizeof(float),streams[sid])); CUDA_CHECK_RETURN(_rmmReAlloc((void
        // **)&(gpk[sid]), n*n*sizeof(float),streams[sid]));
        s_n[sid] = n;
    }
    return r;
}

void mc::set_params_buff(int oldS_n, int N_sid, int sid) {
    mr->free(g_P_i2j[sid], oldS_n * sizeof(float) * oldN[sid] * reSampleTimes,
             streams[sid]);
    g_P_i2j[sid] = (float *)mr->malloc(
        s_n[sid] * sizeof(float) * N_sid * reSampleTimes, streams[sid]);
}

bool mc::set_params(int n, int sid, vector<float> &args) {
    bool r;
    vecFloatMapper evargs(args.data(), n * n + n);
    // RowVectorXf eargs=evargs(seqN(0,n));
    RowVectorXf eargs = evargs.block(0, 0, 1, n);
    float *peargs = eargs.data();
    // RowVectorXf kargs=evargs(seqN(n,n*n-n));
    RowVectorXf kargs = evargs.block(0, n, 1, n * n - n);
    // RowVectorXf vargs=evargs(seqN(n*n,n));
    RowVectorXf vargs = evargs.block(0, n * n, 1, n);
    float *pvargs = vargs.data();
    r = genMatK(&matK[sid], n, kargs);
    //&matK不可修改，但是matK的值可以修改
    r = r && genMatP(&matP[sid], matK[sid]);
    // cout<<"[K]:\n"<<*matK<<endl;
    // memcpy(hpe[sid], peargs, sizeof(float)*n);
    // memcpy(hpv[sid], pvargs, sizeof(float)*n);
    // memcpy(hpk[sid], matK->data(), sizeof(float)*n*n);
    // memcpy(hpp[sid], matP->data(), sizeof(float)*n);
    // todo
    CUDA_CHECK_RETURN(cudaMemcpyAsync(gpe[sid], peargs, sizeof(float) * n,
                                      cudaMemcpyHostToDevice, streams[sid]));
    CUDA_CHECK_RETURN(cudaMemcpyAsync(gpv[sid], pvargs, sizeof(float) * n,
                                      cudaMemcpyHostToDevice, streams[sid]));
    CUDA_CHECK_RETURN(cudaMemcpyAsync(gpk[sid], matK[sid]->data(), sizeof(float) * n * n,
                                      cudaMemcpyHostToDevice, streams[sid]));
    CUDA_CHECK_RETURN(cudaMemcpyAsync(gpp[sid], matP[sid]->data(), sizeof(float) * n,
                                      cudaMemcpyHostToDevice, streams[sid]));
    CUDA_CHECK_RETURN(cudaStreamSynchronize(streams[sid]));
    return r;
}
