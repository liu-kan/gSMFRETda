# Usage of gSMFRETda

The gSMFRETda can analysis multi-state dynamic systems. It uses conversion rate matrix, which is shown in figure below, to represent systems' multi-state dynamic properties.
The element [![K_{i,j}](https://quicklatex.com/cache3/8f/ql_e5a7eae102c94f7a1c0ca24935c97b8f_l3.png)](https://quicklatex.com/cache3/8f/ql_e5a7eae102c94f7a1c0ca24935c97b8f_l3.png) in the matrix represent the conversion rate from the state j to the state i. When i=j, the element equal to negative numbers of sum of other elements in the column.

[<img src="doc/mat.jpg" width="600"/>](doc/mat.jpg)

The matrix of below figure represent the dynamic system in the left subfigure above. In case you want to specify the kinetic model to be used in the calculation as shown above, you can add the argument "-k" to the pdaServ program to define the conversion rate matrix as [![\begin{bmatrix}K \end{bmatrix} = \begin{bmatrix}-k_f & k_{sf} & 0 \\k_f & -k_{sf}-k_{su} & k_u \\0 & k_{su} & -k_u \end{bmatrix}](https://quicklatex.com/cache3/ac/ql_cd9bf1e7cf8744223c5adb0a78259eac_l3.png)](https://quicklatex.com/cache3/ac/ql_cd9bf1e7cf8744223c5adb0a78259eac_l3.png). 
Specifically, you need to add "-k 3 7" to set ke_zero=[3,7], in this case, to setup which element is zero in the matrix. Index starts from 1, and the matrix is RowMajor.