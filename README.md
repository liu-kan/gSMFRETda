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
## Building prerequest
* CUDA version 10 
* libhdf5-dev 1.10 and newer 

## Build the code
```bash
sudo apt install libhdf5-dev pkg-config
cd gSMFRETda
make main -j8
```