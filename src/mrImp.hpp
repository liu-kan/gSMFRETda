#pragma once

#include <rmm/mr/device/pool_memory_resource.hpp>
#include "rmm/detail/error.hpp"
#include "rmm/mr/device/cuda_memory_resource.hpp"
#include "rmm/mr/device/default_memory_resource.hpp"
#include "rmm/mr/device/device_memory_resource.hpp"

class mrImp
{
    using Pool = rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>;
    protected:
        Pool *mr;
        bool _sync;
    public:
        mrImp(std::size_t init_size,float maxe,bool sync=true);
        void* malloc(std::size_t size,cudaStream_t stream);
        void free(void *p,std::size_t size,cudaStream_t stream);
        ~mrImp();

};