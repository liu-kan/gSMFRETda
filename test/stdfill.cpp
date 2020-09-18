#include <iostream>
// #include <Eigen/Core>
// #include <Eigen/Eigen>
#include <string>
#include <vector>
#include <algorithm>
#include <iterator>
#include <numeric>
using namespace std;

void tfuncp(int p){
    while (p>0)
    {
        p=p-(p/2)-1;
        printf("%d\n",p);
    }
    
}
int main(){
    int p=10;
    tfuncp(p);
    printf("main p:%d\n",p);
    int sidx=0;
    int streamNum=3;
    for(int i=0;i<10;i++){
        printf("malloc\t=%d\n",(sidx)%streamNum);
        printf("copy\t=%d\n",(sidx++)%streamNum);
    }
    return 1;
}