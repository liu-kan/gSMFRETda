#ifndef genrand_INCLUDED
#define genrand_INCLUDED

__forceinline__ __device__ int drawDisIdx(int n,float* p,curandStateScrambledSobol64* state){
    curandStateScrambledSobol64 s=*state;
    float pv=curand_uniform(&s);
    float a=0;
    int i=0;    
    for (;i<n;i++){
        a+=p[i];
        if (a>=pv){
            *state=s;
            return i;
        }
    }
    *state=s;
    return n-1;
}
__forceinline__ __device__ float drawTau(float k,curandStateScrambledSobol64* state){
    curandStateScrambledSobol64 s=*state;
    float pv=curand_uniform(&s);
    *state=s;
    return logf(1-pv)/(-k);
}
__forceinline__ __device__ float drawE(float e,float v,curandStateScrambledSobol64* state){
    curandStateScrambledSobol64 s=*state;
    float pv=curand_normal(&s)*v+e;
    *state=s;
    return pv;
}

__forceinline__ __device__ int drawJ_Si2Sj(float *matP,int n_sates,int i,curandStateScrambledSobol64* state){
    /*    
    P_i2j=copy.deepcopy(matP)
    P_i2j[i]=0
    P_i2j=P_i2j/sum(P_i2j)
    j=drawDisIdx(np.arange(n_states),P_i2j)
    return j
    */

}

//malloc/free *bc
__forceinline__ __device__ bool draw_P_B_Tr(int *bc,float totPhoton,int timebin,float* timesp,
        float bg_rate, curandStateScrambledSobol64* state){
    curandStateScrambledSobol64 s=*state;
    bool r=true;
    for(int i=0;i<timebin;i++){
        int sv=curand_poisson (&s, *(timesp+i)*bg_rate);
        if (sv<=totPhoton)
            bc[i]=sv;
        else{
            bc[i]=floorf(curand_uniform(&s) * totPhoton);
            r=false;
        }
    }
    *state=s;
    return r;
}

#endif