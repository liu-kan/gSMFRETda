# gSMFRETda

gSMFRETda is a single molecule fluorescence resonance energy transfer (or smFRET) probability distribution analysis (PDA) program written under C++/CUDA. It can use GPUs to accelerate Monte Carlo simulations of PDA. And because it drastically reduces the calculation time, people can sample dwell time and other parameters more densely, it enable PDA to analyse very fast dynamic interconversion of the system or some other complex TCSPC setup forms requesting lots of PDA calculation.

The program is fine designed. It's not only allows you use multi streams multi GPUs on multi nodes, it also implements a parameter server protocol to allow simulations decouple with the optimization process. In fact, I implemented a Python evolutionary optimization server to provide parameters to gSMFRETda nodes, users can implement their own algorithm easily base on this repo's [opt.py](https://github.com/liu-kan/smFRETLikelihood/blob/gSMFRETda/serv_pdaga/opt.py).

## Installtion

Although the program can calculate FRET efficiency (E) interconversion matrix (K) now, it is still rapidly developed, the binary package isn't provided, you can compile it under instructions of file [INSTALL.md](INSTALL.md)

## Useage

After compiling code under instructions of file [INSTALL.md](INSTALL.md), you will have a program file gSMFRETda and a directory called smFRETLikelihood in building directory. Before running code, install prerequest of [smFRETLikelihood](https://github.com/liu-kan/smFRETLikelihood/blob/gSMFRETda/README.md). Then, use screen or tmux to run programs.

```bash
# Prepare data for gSMFRETda
# open one terminal use screen
python3 smFRETLikelihood/untils/ptu2hdf.py -i /path/to/tcspc.ptu -o /path/to/phconvertHDF5file.h5
python3 smFRETLikelihood/untils/arrivalTimePDAdata.py -i /path/to/phconvertHDF5file.h5 -o /path/to/gSMFRETdaHDF5file.hdf5
# start parameter server on localhost:7777
python3 smFRETLikelihood/serv_pdaga/pdaServ.py
# open another terminal use ctrl_a + c
./gSMFRETda /path/to/gSMFRETdaHDF5file.hdf5
# if you have more than one GPU, open more terminal use ctrl_a + c
./gSMFRETda /path/to/gSMFRETdaHDF5file.hdf5 -g1 # -g1 means gpu id 1, gpu id start from 0
```
More help can be obtained by 
```bash 
./gSMFRETda -h
python3 smFRETLikelihood/serv_pdaga/pdaServ.py -h
```