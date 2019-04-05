#ifndef culist_INCLUDED
#define culist_INCLUDED
#ifdef __CUDACC__
#define CU_CALL_MEMBER __host__ __device__
#else
#define CU_CALL_MEMBER
#endif 
#define _culist_ele_sz 2
#define _sz _culist_ele_sz
#include "eigenhelper.hpp"
template <typename T>
struct clist{
    T *x;
    struct clist<T>* nx;
};
template <typename T>
class cuList
{
public:
    clist<T> *l;
    int sz;
    int len;   
    int errno; 
    CU_CALL_MEMBER cuList(){
        l=new clist<T>();
        l->x=new T[_sz];
        sz=_sz;
        len=0;
        errno=0;
    };
    CU_CALL_MEMBER void freeList(){
        int len1=sz/_sz;
        clist<T> *r=l;
        if (len1>0){
            for (int i=0;i<len1;i++){
                delete[](r->x);
                clist<T> *t=r;
                r=r->nx;
                delete(t); 
            }
        }        
    };
    CU_CALL_MEMBER T* at(int i){
        if (i>=sz){
            int newblk=floorf(((float)i-sz)/_sz)+1;
            new_node(newblk);
            return at(i);
        }
        else{
            int y=i%_sz;
            int x=i/_sz;
            clist<T> *tl=goth(x);
            return (tl->x)+y;
        }
    };
    CU_CALL_MEMBER void diff(arrF* vDiff){
        if(len>1){         
            for(int i=len-1;i>0;i--){
                (*vDiff)(i-1)=*(at(i))-*(at(i-1));
            }            
        }
    };
    CU_CALL_MEMBER void set(int i, const T &v){
        T* item=at(i);
        *item=v;
    };
    CU_CALL_MEMBER void append(const T& v){
        set(len,v);
        ++len;
    };
    CU_CALL_MEMBER void new_node(int i){
        clist<T> *r=goth(-1);
        for (int ii=0;ii<i;ii++){
            r->nx=new clist<T>();
            r=r->nx;
            r->x=new T[_sz];
            if (r->nx==NULL||r->x==NULL)                
                printf("cuList malloc NULL #\n");
            // else
            //     printf("cuList malloc fine!\n");
            sz+=_sz;
        }
    };
    CU_CALL_MEMBER clist<T>* goth(int i){
        clist<T> *r=l;
        if (i>=0 && i<sz/_sz){
            for (int ii=0;ii<i;ii++){
                r=r->nx;
            }
            return r;
        }
        else if(i<0){
            int ri=sz/_sz-1;
            for (int ii=0;ii<ri;ii++){
                r=r->nx;
            }
            return r;
        }
        else
            return NULL;
    };
};

#endif