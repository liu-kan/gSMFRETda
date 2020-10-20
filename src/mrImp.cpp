#include "mrImp.hpp"
#include "cuda_tools.hpp"
#include "helper_cuda.h"
mrImp::mrImp(std::size_t init_size,float maxe, int gpuid, bool sync,int type){
    _sync=sync;
    _gpuid=gpuid;
    _type=type;
    if(_type==1){
        std::size_t free{}, total{};
        std::tie(free, total) = rmm::mr::detail::available_device_memory();
        std::cout<< "Tot gpu Mem size: "<<total<<std::endl;
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
        // cudaSetDevice(_gpuid);
        cuda_mr=new rmm::mr::cuda_memory_resource();
        // Construct a resource that uses a coalescing best-fit pool allocator
        pool_mr=new rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>{cuda_mr};
        // rmm::mr::set_current_device_resource(pool_mr); // Updates the current device resource pointer to `pool_mr`
        rmm::mr::set_per_device_resource(rmm::cuda_device_id{_gpuid}, pool_mr);
        mr=new rmm::mr::pool_memory_resource{rmm::mr::get_per_device_resource(rmm::cuda_device_id{_gpuid}),initpool,static_cast<std::size_t>(max_pool*_maxe)};
    }
}
mrImp::~mrImp(){
    if(_type==1){
        delete(mr);
        delete(pool_mr);
        delete(cuda_mr);
    }
}
void* mrImp::malloc(std::size_t size,cudaStream_t stream ){
    if(_type==1){
        void* p= mr->allocate(size, stream);
        if(_sync)
            checkCudaErrors(cudaStreamSynchronize(stream));
        return p;
    }
    else {        
        void *p;
        checkCudaErrors(cudaMalloc(&p,size));
        checkCudaErrors(cudaStreamSynchronize(stream));
        return p;
    }
}
void mrImp::free(void *p,std::size_t size,cudaStream_t stream ){
    if(_type==1){
        mr->deallocate(p,size, stream);
        if(_sync)
            checkCudaErrors(cudaStreamSynchronize(stream));
    }
    else{
        checkCudaErrors(cudaFree(p));
        checkCudaErrors(cudaStreamSynchronize(stream));
    }
}




