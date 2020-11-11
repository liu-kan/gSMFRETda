# Usage of gSMFRETda

The gSMFRETda can analysis multi-state dynamic systems. It uses conversion rate matrix, which is shown in figure below, to represent systems' multi-state dynamic properties.
The element <!--![K_{i,j}]-->[<img src="http://www.sciweavers.org/tex2img.php?eq=%20K_%7Bi%2Cj%7D%20&bc=Transparent&fc=Black&im=png&fs=13&ff=fourier&edit=0" align="top"/>](http://www.sciweavers.org/tex2img.php?eq=%20K_%7Bi%2Cj%7D%20&bc=Transparent&fc=Black&im=png&fs=13&ff=fourier&edit=0) in the matrix represent the conversion rate from the state j to the state i. When i=j, the element equal to negative numbers of sum of other elements in the column.

[<img src="doc/mat.jpg" width="600"/>](doc/mat.jpg)

The matrix of below figure represent the dynamic system in the left subfigure above. In case you want to specify the kinetic model to be used in the calculation as shown above, you can add the argument "-k" to the pdaServ program to define the conversion rate matrix as ![ \begin{bmatrix}K \end{bmatrix} = \begin{bmatrix}-k_f & k_{sf} & 0 \\k_f & -k_{sf}-k_{su} & k_u \\0 & k_{su} & -k_u \end{bmatrix} ](http://www.sciweavers.org/tex2img.php?eq=%20%5Cbegin%7Bbmatrix%7DK%20%5Cend%7Bbmatrix%7D%20%3D%20%5Cbegin%7Bbmatrix%7D-k_f%20%26%20k_%7Bsf%7D%20%26%200%20%5C%5Ck_f%20%26%20-k_%7Bsf%7D-k_%7Bsu%7D%20%26%20k_u%20%5C%5C0%20%26%20k_%7Bsu%7D%20%26%20-k_u%20%5Cend%7Bbmatrix%7D%20&bc=Transparent&fc=Black&im=png&fs=13&ff=fourier&edit=0). 
Specifically, you need to add "-k 3 7" to set ke_zero=[3,7] in this case to setup which element is zero. Index start from 1, matrix is setted RowMajor.