#include "mrImp.hpp"
#include "cuda_tools.hpp"
#include "helper_cuda.h"
mrImp::mrImp(std::size_t init_size,float maxe, int gpuid, bool sync,int type){
    _sync=sync;
    _gpuid=gpuid;
    _type=type;
}
mrImp::~mrImp(){

}
void* mrImp::malloc(std::size_t size,cudaStream_t stream ){
        void *p;
        checkCudaErrors(cudaMalloc(&p,size));
        if (_sync)
            checkCudaErrors(cudaStreamSynchronize(stream));
        return p;
}
void mrImp::free(void *p,std::size_t size,cudaStream_t stream ){
        checkCudaErrors(cudaFree(p));
        if(_sync)
            checkCudaErrors(cudaStreamSynchronize(stream));
}




