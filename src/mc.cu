#include "mc.hpp"
#include "eigenhelper.hpp"
#include "cuda_tools.hpp"

using namespace std;
__global__ void mc_kernel(unsigned char *g_mask_ad,long size_mask,long* rsize){
    arrUcharMapper mask_adA(g_mask_ad,size_mask);
    *rsize=mask_adA.cols();
}

void mc_gpu(std::vector<unsigned char> mask_ad,std::vector<unsigned char> mask_dd){
    arrUchar *mask_adAp,*mask_ddAp;
    long size_mask=mask_ad.size();
    arrUcharMapper mask_adA(mask_ad.data(),size_mask);      
    arrUcharMapper mask_ddA(mask_dd.data(),size_mask);  
    unsigned char *g_mask_ad;
    long *r_size,*rc=new long[1];
    CUDA_CHECK_RETURN(cudaMalloc((void **)&g_mask_ad, sizeof(unsigned char)*size_mask));
    CUDA_CHECK_RETURN(cudaMemcpy(g_mask_ad, mask_ad.data(), sizeof(unsigned char)*size_mask, 
        cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&r_size, sizeof(long)));

    mc_kernel<<<1,1>>>(g_mask_ad,size_mask,r_size);
    

    CUDA_CHECK_RETURN(cudaMemcpy(rc, r_size, sizeof(long), cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaFree(g_mask_ad));
    CUDA_CHECK_RETURN(cudaFree(r_size));
    cudaDeviceSynchronize();
    cout<<"rsize:"<<*rc<<endl;
}