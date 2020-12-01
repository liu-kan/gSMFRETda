#include "eigenhelper.hpp"
#include <iostream>

bool genMatK(float* matK,int n, RowVectorXf& args){
    if (args.size()<1)
        return false;
    if (args.size()!=n*n-n)
        return false;
    for (int i=0;i<n;i++)
        for (int j=0;j<n;j++)
            if (i<j)
                *(matK+i+j*n)=args[i*(n-1)+j-1];
            else if(i>j)
                *(matK+i+j*n)=args[i*(n-1)+j];
            else
                *(matK+i+j*n)=0;
    for(int j=0;j<n;j++)
        for (int ii=0;ii<n;ii++)
            if(ii!=j)
                *(matK+j+j*n)-=*(matK+ii+j*n);
    return true;
}

bool genMatP(float* matP,float* matK,int n){
    // int n=matK->rows();
    if (n<1)
        return false;
    float ap=0;
    for (int i=0;i<n;i++){
        *(matP+i)=0;
        for(int j=0;j<n;j++)
                *(matP+i)+=*(matK+i+j*n);
        *(matP+i)-=*(matK+i+i*n);
        ap+=*(matP+i);
    }
    // for (int i=0;i<n;i++) 
    //     *(*matP+i)=*(*matP+i)/ap;
    return true;
}