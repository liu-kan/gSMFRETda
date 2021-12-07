# gSMFRETda

[![build status](https://github.com/liu-kan/gSMFRETda/actions/workflows/cpu_test.yml/badge.svg)](https://github.com/liu-kan/gSMFRETda/actions/workflows/cpu_test.yml)
[![gpu test status](https://github.com/liu-kan/gSMFRETda/actions/workflows/gpu_test.yml/badge.svg)](https://github.com/liu-kan/gSMFRETda/actions/workflows/gpu_test.yml)
[![windows build status](https://github.com/liu-kan/gSMFRETda/actions/workflows/windows.yml/badge.svg)](https://github.com/liu-kan/gSMFRETda/actions/workflows/windows.yml)
<!-- [![codecov](https://codecov.io/gh/liu-kan/gSMFRETda/branch/master/graph/badge.svg?token=T6XKP99802)](https://codecov.io/gh/liu-kan/gSMFRETda) -->

gSMFRETda is a single molecule fluorescence resonance energy transfer (or smFRET) probability distribution analysis (PDA) program written under C++/CUDA. It can use GPUs to accelerate Monte Carlo simulations of PDA. And because it drastically reduces the calculation time, people can sample dwell time and other parameters more densely, it enable PDA to analyse very fast dynamic interconversion of the system or some other complex TCSPC setup forms requesting lots of PDA calculation.

The program is fine designed. It's not only allows you use multi streams multi GPUs on multi nodes, it also implements a parameter server protocol to allow simulations decouple with the optimization process. In fact, I implemented a Python evolutionary optimization server to provide parameters to gSMFRETda nodes, users can implement their own algorithm easily base on this repo's [opt.py](https://github.com/liu-kan/pySMFRETda/blob/main/serv_pdaga/opt.py).

## Installtion

Although the program can calculate FRET efficiency (E) interconversion matrix (K) now, it is still rapidly developed, the binary package isn't provided, you can compile it under instructions of file [INSTALL.md](INSTALL.md)

## Usage

After compiling code under instructions of file [INSTALL.md](INSTALL.md), you will have a program file gSMFRETda and a directory called pySMFRETda in building directory. Before running code, install prerequest of [pySMFRETda](https://github.com/liu-kan/pySMFRETda/blob/main/README.md). Then, use [screen](https://gist.github.com/liu-kan/9ab154d91c3bc8659a2979fcca74406d) or [tmux](https://gist.github.com/liu-kan/59ed943b149447aa34dff87d49c8dc96) to run programs.

```bash
# Prepare data for gSMFRETda
# open one terminal use screen
python3 pySMFRETda/untils/ptu2hdf.py -i /path/to/tcspc.ptu -o /path/to/phconvertHDF5file.h5
python3 pySMFRETda/untils/arrivalTimePDAdata.py -i /path/to/phconvertHDF5file.h5 -o /path/to/gSMFRETdaHDF5file.hdf5
# start parameter server on localhost:7777
python3 pySMFRETda/serv_pdaga/pdaServ.py
# open another terminal use ctrl_a + c
./gSMFRETda /path/to/gSMFRETdaHDF5file.hdf5
# if you have more than one GPU, open more terminal use ctrl_a + c
./gSMFRETda /path/to/gSMFRETdaHDF5file.hdf5 -g1 # -g1 means gpu id 1, gpu id start from 0
```
More help can be obtained by 
```bash 
./gSMFRETda -h
python3 pySMFRETda/serv_pdaga/pdaServ.py -h
```
Or refering [doc/USAGE.md](doc/USAGE.md).

## Trouble shooting
Do not run this GPU computing program for very long time on machines with poor heat dissipation, especially on non-gaming notebooks like ultra-thin laptops. Otherwise, the program may encounter runtime errors or even bring irreversible damages to your computer hardware.

If you encounter any problems, feel free to open a [new issue here](https://github.com/liu-kan/gSMFRETda/issues).