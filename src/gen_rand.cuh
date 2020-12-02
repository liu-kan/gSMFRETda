#pragma once

//#include <Eigen/Eigen>

#define VECTOR_SIZE 64

__global__ void setup_kernel  (rk_state * state, unsigned long seed , int N,
    unsigned long long *sobolDirectionVectors, 
    unsigned long long *sobolScrambleConstants, 
    curandStateScrambledSobol64* stateQ)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<N){
        // curand_init ( seed, idx, 0, &state[idx] );        
        curand_init(sobolDirectionVectors + VECTOR_SIZE*idx, 
            sobolScrambleConstants[idx], 
            1234, 
            &stateQ[idx]);
        unsigned long long llseed=curand(stateQ+idx);  
        // printf("llseed = %llu\n",llseed);  
        rk_seed(llseed,state+idx);
    }
    // else
    // printf("idx = %d, N=%d \n",idx,N); 
} 

__device__ int drawDisIdx(int n,float* p,curandStateScrambledSobol64* state){
    curandStateScrambledSobol64 s=*state;
    float pv=curand_uniform(&s);
    float a=0;
    int i=0;    
    // printf("pv= %f\n",pv);
    for (;i<n;i++){
        a+=p[i];
        if (a>pv){
            *state=s;
            return i;
        }
    }
    *state=s;
    return n-1;
}
__device__ float drawTau(float k,curandStateScrambledSobol64* state,int randN=10, float precision=1e-6){
    curandStateScrambledSobol64 s=*state;
    float pv=curand_uniform(&s);
    *state=s;
    float r=logf(1-pv)/(-k);
    if (r<precision && randN>0){
        float rd=curand_uniform(&s);
        r+=ceilf(randN*rd)*precision;
    }
    return r; //precision is max precision
}
__device__ float drawE(float e,float r0,float v,curandStateScrambledSobol64* state){
    curandStateScrambledSobol64 s=*state;
    float rd=curand_normal(&s)*v+r0*powf(1/e-1,1.0/6);
    *state=s;
    return 1/(1+powf(rd/r0,6));
}
// typedef Eigen::Map<Eigen::MatrixXf> matFlMapper;
__device__ int drawJ_Si2Sj(float *P_i2j, float *matK,int n_sates,int i,curandStateScrambledSobol64* state){
    /*    
    P_i2j=copy.deepcopy(matP)
    P_i2j[i]=0
    P_i2j=P_i2j/sum(P_i2j)
    j=drawDisIdx(np.arange(n_states),P_i2j)
    return j
    */
    memcpy(P_i2j,matK+i*n_sates,sizeof(float)*n_sates);   //Eigen::ColMajor
    P_i2j[i]=0.0;
    for( int ii=0;ii<n_sates-1;ii++){
        if(ii>=i)
            P_i2j[ii]=P_i2j[ii+1];
    }
    float sum=0.0;
    for( int ii=0;ii<n_sates-1;ii++){
        sum+=P_i2j[ii];
    }
    for( int ii=0;ii<n_sates-1;ii++){
        P_i2j[ii]=P_i2j[ii]/sum;
    }
    int j = drawDisIdx(n_sates-1,P_i2j,state);
    if (j>=i)
        ++j;
    // free(P_i2j);
    return j;
}

//malloc/free *bc
__device__ bool draw_P_B_Tr(float *bc,float *totPhoton,int timebin,float* timesp,
        float bg_rate, curandStateScrambledSobol64* state){
    curandStateScrambledSobol64 s=*state;
    bool r=true;
    for(int i=0;i<timebin;i++){
        int sv=curand_poisson (&s, *(timesp+i)*bg_rate);
        if (sv<=*(totPhoton+i))
            bc[i]=float(sv);
        else{
            bc[i]=floorf(*(totPhoton+i)*curand_uniform(&s) );
            r=false;
        }
    }
    *state=s;
    return r;
}

