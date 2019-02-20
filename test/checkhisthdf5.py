import h5py
f=h5py.File("r.hdf5","r")
dset=f['/r']
print(dset.dtype)

import numpy as np
import matplotlib.pyplot as plt
data=dset[:]
print(len(data))
print(np.sum(data))

fig=plt.figure()
# the histogram of the data
n, bins, patches = plt.hist(data, 100,  facecolor='g', alpha=0.75)
plt.xlabel('X')
plt.ylabel('C')
plt.title('Histogram of CUDA data')
plt.grid(True)

fig2=plt.figure()
from scipy import stats
# b=stats.binom.rvs(5,0.7,size=2048)
b=stats.expon.rvs(scale=1.0/25,size=2048)
n, bins, patches = plt.hist(b, 100,  facecolor='g', alpha=0.75)
plt.xlabel('X')
plt.ylabel('C')
plt.title('Histogram of python cpu data')
plt.grid(True)

print(b)

plt.show()