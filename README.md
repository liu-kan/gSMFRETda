# gSMFRETda

gSMFRETda is a single molecule fluorescence resonance energy transfer (or smFRET) probability distribution analysis (PDA) calculation program written under C++/CUDA. It can use GPUs to accelerate Monte Carlo simulations. And because it drastically reduces the calculation time, people can sample dwell time more densely, it enable PDA to analyse very fast dynamic interconversion of the system or some other complex TCSPC setup forms requesting lots of PDA calculation.

The program is fine designed. It's not only allows you use multi stream multi GPUs on multi nodes, it also implements a parameter server protocol to allow simulations decouple with the optimization process. In fact, I implemented a Python evolutionary optimization server to provide parameters to gSMFRETda nodes, users can implement their own algorithm easily base on code in [the repo](https://github.com/liu-kan/smFRETLikelihood/blob/gSMFRETda/serv_pdaga/opt.py).

## Installtion

Although the program can calculate FRET efficiency (E) interconversion matrix (K) now, it is still rapidly developed, the binary package isn't provided, you can compile it under instructions of file [INSTALL.md](INSTALL.md)

## Useage