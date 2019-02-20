import h5py
f=h5py.File("r.hdf5","r")
dset=f['/r']
print(dset.dtype)

import numpy as np
import matplotlib.pyplot as plt
data=dset[:]
print(len(data))
print(np.sum(data))

# the histogram of the data
n, bins, patches = plt.hist(data, 100,  facecolor='g', alpha=0.75)
plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
plt.grid(True)


# from scipy import stats
# b=stats.poisson.rvs(0.6,size=2048)
# n, bins, patches = plt.hist(b, 100,  facecolor='g', alpha=0.75)
# plt.xlabel('Smarts')
# plt.ylabel('Probability')
# plt.title('Histogram of IQ')
# plt.grid(True)



plt.show()