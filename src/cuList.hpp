#ifndef culist_INCLUDED
#define culist_INCLUDED
#ifdef __CUDACC__
#define CU_CALL_MEMBER __host__ __device__
#else
#define CU_CALL_MEMBER
#endif 
#define _culist_ele_sz 2
template <typename T>
class cuList
{
public:
    T x*;
    // T x2;
    cuList* nx;
    int sz;
    CU_CALL_MEMBER cuList();
    CU_CALL_MEMBER T* at(int i);
    CU_CALL_MEMBER void set(int i, T& v);
};

#endif