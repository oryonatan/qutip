# Finds the minimal angle between two subsapces
# I think

import numpy as np
from qutip import *

print("hey")

A  = (tensor([basis(2,0)]*3),tensor([basis(2,1),basis(2,0),basis(2,1)]))
B  = (tensor([basis(2,1)]*3),tensor([basis(2,1),basis(2,0),basis(2,0)]))

mat = np.zeros([len(A),len(B)])
for i in range(len(A)):
    for j in range(len(B)):
        mat[i][j] = A[i].overlap(B[j].trans().conj())

print (np.arccos(np.amax(mat)))
