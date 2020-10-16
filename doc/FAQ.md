# FAQ
## RMM build
* If you use Ubuntu 18.04, download new stable version of cmake, and install it by hand first.
* If you use Archlinux, etc., whose version of gcc is too high, set CC and CXX to suitable version of gcc copy.
```bash
export CC=/opt/cuda/bin/gcc
export CXX=/opt/cuda/bin/g++
mkdir build            # make a build directory
cd build               # enter the build directory
cmake .. # configure cmake ... use $CONDA_PREFIX if you're using Anaconda
make 
make install  
```
* Make sure you build rmm lib with the **cuda version same as** you build gSMFREda.