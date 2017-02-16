import warnings

import pyximport

warnings.filterwarnings('ignore')

import sys
import os


#sys.path.insert(0, os.path.join(os.getcwd(), os.pardir))
import numpy as np

#pyximport.install(setup_args={"include_dirs": np.get_include()})
import XXZZham as XXZZham
from XXZZham import rotate_to_00_base
import random
import time
from scipy import linalg
PRECISION = 2 ** -30

for n in range(2,7):
    alist = []
    terms = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            a = random.uniform(-10, 10)
            alist.append(a)
            terms.append(XXZZham.XXZZ_term(i, j, a))

    H = XXZZham.XXZZham(terms)
    H_com = H.get_commuting_term_ham()
    h_com_rot = rotate_to_00_base(H_com)

    start_time = time.time()

    HCR_en = h_com_rot.eigenstates(eigvals=10)[0]
    # HCR_en = linalg.eigvals(h_com_rot.data.todense())
    duration = time.time() - start_time

    HCR_degeneracy = sum(abs(HCR_en - HCR_en.min()) < PRECISION)

    print ("%d-qubits %d degen; %f seconds to compute\n" % (n,HCR_degeneracy,duration))
