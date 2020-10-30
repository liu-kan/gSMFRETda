# Installtion gSMFRETda

## Compile from source

### Clone the code
```bash
git clone --recurse-submodules -j8 https://git.liukan.org/liuk/gSMFRETda.git
```
or
```bash
git clone https://git.liukan.org/liuk/gSMFRETda.git
git submodule init
git submodule update
```

### Get dataset & 3rd Party libs
gSMFRETda use git annex instead of git lfs to handle large files, because it can use local filesystem repository to save large files. To get dataset and 3rd party libs, you need install GitAnnex first, and follow the instructions.
```bash
sudo apt-get install git-annex
git remote add datasrc file:///home/liuk/sync/coding/smfret/gSMFRETda.git
git fetch datasrc # getting files
git annex get . # retrieve everything under the current directory
```

### Building prerequest
* CUDA version >= 10 
* libhdf5-dev 1.10 and newer 
* CMake >= 3.11
* librmm 0.15 https://anaconda.org/rapidsai/librmm/files

I'm trying to let the program can be compiled both by Linux and Windows natively. I recommend you run the program under Linux now. If you compile it in Windows, [compile with VS 2019](https://docs.microsoft.com/en-us/cpp/build/cmake-projects-in-visual-studio?view=vs-2019). Of course, you can just compile in [WSL2 with CUDA](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) supported.

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
