#include "mrImp.hpp"
#include "cuda_tools.hpp"

mrImp::mrImp(std::size_t init_size,float maxe, int gpuid, bool sync,int type){
    _sync=sync;
    _gpuid=gpuid;
    _type=type;
}
mrImp::~mrImp(){

}
void* mrImp::malloc(std::size_t size,cudaStream_t stream ){
        void *p;
        CUDA_CHECK_RETURN(cudaMalloc(&p,size));
        if (_sync)
            CUDA_CHECK_RETURN(cudaStreamSynchronize(stream));
        return p;
}
void mrImp::free(void *p,std::size_t size,cudaStream_t stream ){
        CUDA_CHECK_RETURN(cudaFree(p));
        if(_sync)
            CUDA_CHECK_RETURN(cudaStreamSynchronize(stream));
}




