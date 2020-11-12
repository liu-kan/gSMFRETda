#pragma once
#include "cuda_tools.hpp"

class mrImp
{
    protected:
        bool _sync;
        int _gpuid;
        int _type;
    public:
        mrImp(std::size_t init_size,float maxe,int gpuid=0, bool sync=false,int type=0);
        void* malloc(std::size_t size,cudaStream_t stream=cudaStreamDefault);
        void free(void *p,std::size_t size,cudaStream_t stream=cudaStreamDefault);
        ~mrImp();

};