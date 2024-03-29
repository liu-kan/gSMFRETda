# Installtion of gSMFRETda

Although gSMFRETda is coded under Linux firstly, but it designed as a cross-platform application. The core program than runs on GPU nodes, can be compiled and executed under Linux and Windows. The [parameter server demo](https://github.com/liu-kan/pySMFRETda) written with python can run on Linux/Windows/MacOS boxes.

If your program runs on the same CUDA_ARCHITECTURES machines as your compiling box, you can add ```-DCMAKE_CUDA_ARCHITECTURES=""``` to the cmake configure command. i.e. ```cmake .. -DCMAKE_CUDA_ARCHITECTURES=""``` for Linux, or ```cmake %src_dir% -DCMAKE_CUDA_ARCHITECTURES="" -G "Visual Studio 16"``` in buildWin.bat for Windows.

## Linux
### Compile from source

### Clone the code
```bash
git clone https://github.com/liu-kan/gSMFRETda.git
```
Submodules are not necessary to pull, them are just listed for FOSSA to analyse dependencies.

### Building prerequest
* CUDA version >= 10 
* libhdf5-dev 1.10 and newer 
* CMake >= 3.11

### Install CUDA
Follow the documentation at https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

### Build the code
For deb systems, like Debian or Ubuntu
```bash
sudo apt install build-essential libhdf5-dev pkg-config protobuf-compiler libprotobuf-dev libnanomsg-dev libboost-dev doxygen libboost-system-dev libboost-serialization-dev cmake gengetopt libboost-filesystem-dev libdlib-dev libgtest-dev libblas-dev liblapack-dev libsqlite3-dev libgsl-dev
```
For rpm systems, like Fedora, Centos or Redhat
```bash
#if centos 8
sudo dnf config-manager --set-enabled PowerTools 
#elseif redhat 8
sudo subscription-manager repos --enable codeready-builder-for-rhel-8-x86_64-rpms
#endif
sudo yum groupinstall "Development Tools" 
#if centos or redhat
sudo yum install https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm # or epel-release-latest-<your_version>.noarch.rpm  
#endif
sudo yum install pkg-config python3-protobuf openssl-devel python3-devel 
sudo dnf install protobuf-devel texinfo hdf5-devel gengetopt 
#If centos or redhat
# Download [nanomsg](https://nanomsg.org/), [gengetopt](http://www.gnu.org/software/gengetopt/), [cmake >=3.14](https://github.com/Kitware/CMake/releases/download/v3.17.4/cmake-3.17.4.tar.gz) and install them.
# If build with ctest, install [remi-release-<version>.rpm](https://rpms.remirepo.net/), then 
sudo dnf install libXext-devel libpng-devel libjpeg-turbo-devel openblas-devel gtest-devel sqlite-devel lapack-devel dlib-devel gsl-devel
# if you don't build with ctest (add -DBUILD_TESTS=OFF when cmake), you can ignore them.
#endif
#if fedora
sudo dnf install nanomsg-devel cmake gtest-devel sqlite-devel lapack-devel dlib-devel gsl-devel
#endif
```

Then, [install Conan](https://conan.io/downloads.html). Finally
```bash
mkdir gSMFRETda/build
cd gSMFRETda/build
conan install --build=missing ../conanfile.posix
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
Now, gSMFRETda depend on boost version EXACTly above 1.70.0. If your Linux distribution doesn't ship this version (in some cases), you need install [CONAN](http://conan.io/downloads.html) first, and run 
```bash
conan install --build=missing ../conanfile.posix 
```
in build directory before you run ```cmake ..```  .

## Windows
The program can be compiled on Windows natively. And if you wish, it can also run on [WSL2 with CUDA](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) with compiling method of Linux mentioned above.

### Compiling natively from source
Compiling under Windows depends on [conan](http://conan.io/downloads.html), in sequence, install [Visual Studio 2019](https://visualstudio.microsoft.com/downloads/) (with **C++, CMake for Windows, CMake for Linux & English language pack Component** installation. It's highly recommended that installing VS to default path.), CUDA for Windows, [vcpkg](https://vcpkg.io/), conan (Add conan to user path), [Git for Windows](https://git-scm.com/download/win). Open a "x64 Native Tools Command Prompt for VS 2019" console form Windows Start Menu.

Assume your vcpkg is installed in c:\vcpkg

```bash
cd \path\to\build\dir
git clone https://github.com/liu-kan/gSMFRETda.git
cd gSMFRETda
vcpkg install --triplet x64-windows "@.vcpkg_deps.txt"
mkdir build
cd build
..\gSMFRETda\buildWin.bat
```

The exe files will sit on .\bin\

If the install_dir of Git for Windows is not default sit at "C:/Program Files/Git". Set environment variables patch to full path of patch.exe file,
```bash
set patch=F:/Program Files/Git/usr/bin/patch.exe
```
before run batch file of ```buildWin.bat```/```buildWinDebug.bat``` .
