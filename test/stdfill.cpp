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



Eigen::Array<int,1,7,Eigen::RowMajor> m;
m <<  1, 2, 3,4,5,6,7;
Eigen::Array<int,1,7,Eigen::RowMajor> mm;
mm << 0, 1, 0,1,1,0,1;
mm=(mm<1).select(0, m) ;
cout <<mm << endl;

std::vector<int> v1={1,2,3,4,5,6,7};
Eigen::Array<int,1,7,Eigen::RowMajor> m3(v1.data()); 
cout <<m3 << endl;
return 1;
}