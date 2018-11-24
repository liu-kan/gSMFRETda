#include <iostream>
#include <Eigen/Dense>
#include <string>
#include <vector>
#include<algorithm>
#include<iterator>
#include <numeric>
using namespace std;
int main(){
bool bits[8];
std::fill(bits, bits+8,1);
std::fill(bits, bits+7,0);
// bits[1]=1;bits[4]=1;

for (auto i: bits)
  std::cout << i << ' ';


std::vector<double> temperatures {65, 75, 56, 48, 31, 28, 32, 44,29, 40, 41,  50};
std::copy(std::begin(temperatures), std::end(temperatures), //List the values
std::ostream_iterator<double>{std::cout, " "});
std::cout << std::endl;
auto average = std::accumulate(std::begin(temperatures),std::end(temperatures), 0.0)/temperatures.size();
std::cout << "Average temperature: "<< average << std::endl;
std::stable_partition(std::begin(temperatures), std::end(temperatures),[average](double t) { return t < average; });
std::copy(std::begin(temperatures), std::end(temperatures),std::ostream_iterator<double>{std::cout, " "});
std::cout << std::endl;



Eigen::MatrixXi m(1,7);
m << 1, 2, 3,4,5,6,7;
Eigen::MatrixXi mm(1,10);
mm << 0, 1, 0,1,1,0,1,0,0,0;
Eigen::MatrixXi mmm=mm.block(0,0,1,7);
Eigen::MatrixXi z(1,7);
z << 0, 0, 0,0,0,0,0;
m = (mmm.array() == 1).select(m, z);
cout << m << endl;

return 1;
}