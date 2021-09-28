# Usage of gSMFRETda

## Configuration of parpmeters

The gSMFRETda can analysis multi-state dynamic systems. It uses transition rate matrix , which is shown in figure below, to represent systems' multi-state dynamic properties.
The element <!-- $K_{i,j}$ --> <img style="transform: translateY(0.1em); background: white;" src="svg/NEmAFy9DBm.svg"> in the matrix represent the interconversion rate constant from the state j to the state i. 
The cumulative distribution of dwell times from a given state, asserting exponential decay, gives <!-- $\tau$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\nNfxWYQ5VJ.svg"> as the inverse of the rate constant. 
When i=j, the element equal to negative numbers of sum of other elements in the column.

<div align="center">[<img src="mat.jpg" width="800"/>](mat.jpg)</div>

The matrix of below figure represent the dynamic system in the left subfigure above. In case you want to specify the kinetic model to be used in the calculation as shown above, you can add the argument "-k" to the pdaServ program to define the transition rate matrix  as 
<!-- $$
\begin{bmatrix}K \end{bmatrix} = \begin{bmatrix}-k_f & k_{sf} & 0 \\k_f & -k_{sf}-k_{su} & k_u \\0 & k_{su} & -k_u \end{bmatrix} .
$$ --> 

<div align="center"><img style="background: white;" src="svg/VHFSqsdsSB.svg"></div>

Specifically, you need to add "-k 3 7" to set ke_zero=[3,7], in this case, to setup which element is zero in the matrix. Index starts from 1, and the matrix is RowMajor.

## The Monte Carlo approach for Probability Distribution Analysis (mcPDA)
When use the Monto Carlo approach to compute PDA, the target is to minimize the probabilistic objective function F of form
<!-- $$ 
\mathcal{F} (\theta ):=\int p(x;\theta )f(x;\phi )dx=\mathbb{E}_{p(x;\theta)}[f(x;\phi)]
$$ --> 

<div align="center"><img style="background: white;" src="svg\Uf8ikKw1ol.svg"></div>
in which a function f of an input variable x with structural parameters <!-- $\phi$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\uMb0GOU5Y2.svg"> is evaluated on average with respect to an input distribution <!-- $p(x; \theta)$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\ecxAAeUAvY.svg"> with distributional
parameters <!-- $\theta$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\6eoE4oEkFJ.svg">. In PDA, f is refer to the reduced chi-squared statistic to measure the goodness of of fit of PDA models.
High-dimensional parameters <!-- $\theta$ --> <img style="transform: translateY(0.1em); background: white;" src="svg\WWHcdpVT13.svg"> of distribution p include the kinetic parameters <!-- $K_{i,j}$ --> <img style="transform: translateY(0.1em); background: white;" src="svg/NEmAFy9DBm.svg">