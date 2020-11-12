# Installtion of gSMFRETda

## Compile from source

### Clone the code
```bash
git clone https://github.com/liu-kan/gSMFRETda.git
```
Submodules are not necessary to pull, them are just listed for FOSSA to analyse dependencies.

### Building prerequest
* CUDA version >= 10 
* libhdf5-dev 1.10 and newer 
* CMake >= 3.11

I'm trying to let the program can be compiled both by Linux and Windows natively. But now if you want to use it under windows, just compile it in [WSL2 with CUDA](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).

### Build the code
For deb systems, like Debian or Ubuntu
```bash
sudo apt install build-essential libhdf5-dev pkg-config protobuf-compiler libprotobuf-dev libnanomsg-dev libboost-dev doxygen libboost-system-dev libboost-serialization-dev cmake gengetopt libboost-filesystem-dev
```
For rpm systems, like Fedora, Centos or Redhat
```bash
sudo yum groupinstall "Development Tools" 
#if centos or redhat
sudo yum install https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm # or epel-release-latest-<your_version>.noarch.rpm  
#endif
#if centos 8
sudo dnf config-manager --set-enabled PowerTools 
#endif
sudo yum install pkg-config python3-protobuf openssl-devel python3-devel 
sudo dnf install protobuf-devel texinfo hdf5-devel 
#If centos or redhat
# Download [nanomsg](https://nanomsg.org/), [gengetopt](http://www.gnu.org/software/gengetopt/), [cmake >=3.14](https://github.com/Kitware/CMake/releases/download/v3.17.4/cmake-3.17.4.tar.gz) and install them.
#endif
#if fedora
sudo dnf install nanomsg-devel gengetopt cmake
#endif
# Download [rmm](https://github.com/rapidsai/rmm), and install it.
```

Then
```bash
mkdir gSMFRETda/build
cd gSMFRETda/build
cmake ..
make -j8
sudo ldconfig
```
If you get error like "No CMAKE_CUDA_COMPILER could be found", try
```bash
export CUDA_HOME=/usr/local/cuda # Your cuda install path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64
export PATH=$PATH:$CUDA_HOME/bin
```
before cmake. Or add them into ~/.bashrc

### Notice
<!-- If you encounter cuda memory access issues, check if your GPU has enough memory first!  -->
