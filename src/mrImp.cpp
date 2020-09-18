#include "mrImp.hpp"
#include "helper_cuda.h"
mrImp::mrImp(std::size_t init_size,float maxe,bool sync){
    _sync=sync;
    std::size_t free{}, total{};
    std::tie(free, total) = rmm::mr::detail::available_device_memory();
    auto const max_pool = static_cast<std::size_t>(free) ;
    std::cout<<"init_size: "<<init_size<<" max_pool: "<<max_pool<<std::endl;
    std::size_t initpool=0;
    float _maxe=maxe;
    if(init_size>=max_pool){
        std::cout<<"Don't have enough GPU MEM! quit!\n";
        return;
    }
    else if (max_pool<init_size*2&& max_pool*maxe>init_size)
        initpool=init_size;
    else if(max_pool*maxe>init_size*2)
        initpool=init_size*2;
    else if(max_pool>init_size*2){
        initpool=init_size*2;
        _maxe=1;
    }
    else if(max_pool>init_size){
        _maxe=1;initpool=init_size;
    }
    std::cout<<"initpool:"<<initpool<<std::endl;
    mr=new Pool{rmm::mr::get_current_device_resource(),initpool,static_cast<std::size_t>(max_pool*_maxe)};
}
mrImp::~mrImp(){
    // mr->release();
    delete(mr);
}
void* mrImp::malloc(std::size_t size,cudaStream_t stream ){
    return mr->allocate(size, stream);
    if(_sync)
        checkCudaErrors(cudaStreamSynchronize(stream));
}
void mrImp::free(void *p,std::size_t size,cudaStream_t stream ){
    mr->deallocate(p,size, stream);
    if(_sync)
        checkCudaErrors(cudaStreamSynchronize(stream));
}




