#include "eigenhelper.hpp"
#include <iostream>
using namespace std;
using namespace Eigen;
bool genMatK(arrFF** matK,int n, RowVectorXf& args){
    if (args.size()<1)
        return false;
    if (args.size()!=n*n-n)
        return false;
    // if (*matK!=NULL)
        delete(*matK);
    *matK=new arrFF(n,n);
    for (int i=0;i<n;i++)
        for (int j=0;j<n;j++)
            if (i<j)
                (**matK)(i,j)=args[i*(n-1)+j-1];
            else if(i>j)
                (**matK)(i,j)=args[i*(n-1)+j];
            else
                (**matK)(i,j)=0;
    for (int i=0;i<n;i++)
        for (int j=0;j<n;j++)
            if (i==j)
                (**matK)(i,j)=-(*matK)->col(j).sum();                   
    return true;
}

bool genMatP(arrF** matP,arrFF* matK){
    int n=matK->rows();
    if (n<1)
        return false;
    // if (*matP!=NULL)
        delete(*matP);        
    *matP=new arrF(n);
    float ap=0;
    for (int i=0;i<n;i++) 
                (**matP)(i)=(*matK).row(i).sum()-(*matK)(i,i);
    ap=(**matP).sum();
    for (int i=0;i<n;i++) 
        (**matP)(i)=(**matP)(i)/ap;
    return true;
}