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

## Build the code
```bash
sudo apt install libhdf5-dev pkg-config protobuf-compiler libprotobuf-dev libnanomsg-dev libboost-dev doxygen libboost-system-dev libboost-serialization-dev cmake gengetopt
mkdir gSMFRETda/build
cd gSMFRETda/build
cmake ..
make -j8
sudo ldconfig
```
