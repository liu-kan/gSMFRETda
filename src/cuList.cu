#include "cuList.hpp"
#define _sz _culist_ele_sz
template<typename T>
CU_CALL_MEMBER cuList<T>::cuList()
{
    nx=NULL;
    x=(T*)alloc(sizeof(T)*_sz);
    sz=0;
}
template<typename T>
CU_CALL_MEMBER T* cuList<T>::at(int i)
{
    if (i<sz)
    return x1;
}
template<typename T>
CU_CALL_MEMBER void cuList<T>::set(int i, T& v){

}