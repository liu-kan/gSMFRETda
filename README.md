# gSMFRETda

## Clone the code
```bash
git clone --recurse-submodules -j8 https://git.liukan.org/liuk/gSMFRETda.git
```
or
```bash
git clone https://git.liukan.org/liuk/gSMFRETda.git
git submodule init
git submodule update
```

## Get dataset & 3rd Party libs
gSMFRETda use git annex instead of git lfs to handle large files, because it can use local filesystem repository to save large files. To get dataset and 3rd party libs, you need install GitAnnex first, and follow the instructions.
```bash
sudo apt-get install git-annex
git remote add datasrc file:///home/liuk/sync/coding/smfret/gSMFRETda.git
git fetch datasrc # getting files
git annex get . # retrieve everything under the current directory
```

## Building prerequest
* CUDA version >= 10 
* libhdf5-dev 1.10 and newer 
* CMake >= 3.11
I'm trying to let the program can be compiled both by Linux and Windows natively. If you compile it in Windows, compile with VS 2019. Of course, you can just compile in [WSL2 with CUDA](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) supported.

## Build the code
```bash
sudo apt install build-essential libhdf5-dev pkg-config protobuf-compiler libprotobuf-dev libnanomsg-dev libboost-dev doxygen libboost-system-dev libboost-serialization-dev cmake gengetopt
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
before cmake.

## Notice
If you encounter cuda memory access issues, check if your GPU has enough memory first! 
If your cmake version is less than 3.11, try add line in /etc/apt/sources.list like.
```bash
deb [arch=amd64] https://apt.kitware.com/ubuntu/ bionic main
```
