#pragma once

#include <rmm/mr/device/pool_memory_resource.hpp>
#include "rmm/detail/error.hpp"
#include "rmm/mr/device/cuda_memory_resource.hpp"
#include "rmm/mr/device/default_memory_resource.hpp"


class mrImp
{
    using Pool = rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>;
    protected:
        Pool *mr;
        bool _sync;
        int _gpuid;
        rmm::mr::cuda_memory_resource *cuda_mr;
        rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> *pool_mr;
        int _type;
    public:
        mrImp(std::size_t init_size,float maxe,int gpuid=0, bool sync=false,int type=0);
        void* malloc(std::size_t size,cudaStream_t stream=cudaStreamDefault);
        void free(void *p,std::size_t size,cudaStream_t stream=cudaStreamDefault);
        ~mrImp();

};