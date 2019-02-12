#include <iostream>
#include <Eigen/Dense>
#include <string>
#include <vector>
#include <algorithm>
#include <iterator>
#include <numeric>
using namespace std;
int main(){
// bool bits[8];
// std::fill(bits, bits+8,1);
// std::fill(bits, bits+7,0);
// // bits[1]=1;bits[4]=1;

// for (auto i: bits)
//   std::cout << i << ' ';


// std::vector<double> temperatures {65, 75, 56, 48, 31, 28, 32, 44,29, 40, 41,  50};
// std::copy(std::begin(temperatures), std::end(temperatures), //List the values
// std::ostream_iterator<double>{std::cout, " "});
// std::cout << std::endl;
// auto average = std::accumulate(std::begin(temperatures),std::end(temperatures), 0.0)/temperatures.size();
// std::cout << "Average temperature: "<< average << std::endl;
// std::stable_partition(std::begin(temperatures), std::end(temperatures),[average](double t) { return t < average; });
// std::copy(std::begin(temperatures), std::end(temperatures),std::ostream_iterator<double>{std::cout, " "});
// std::cout << std::endl;


int s=7;
Eigen::Array<int,1,7,Eigen::RowMajor> m;
m <<  1, 2, 3,4,5,6,7;
Eigen::Array<bool,1,Eigen::Dynamic,Eigen::RowMajor> mm,*mp;
mm.resize(1,s);
mm << 0, 1, 0,1,1,0,1;
mp=&mm;
m=(mm).select(m,0) ;
cout << m << endl;
cout <<(int)(*mp)(0,0) << endl;
typedef Eigen::Map<Eigen::Array<int,1,Eigen::Dynamic,Eigen::RowMajor>> Mapper;
std::vector<int> v1={1,2,3,4,5,6,7};
Mapper m3(v1.data(),s);

cout <<m3 << endl;
return 1;
}