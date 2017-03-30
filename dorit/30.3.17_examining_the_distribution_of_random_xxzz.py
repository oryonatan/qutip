import warnings
import matplotlib.pyplot as plt
import pyximport
from tqdm import tnrange, tqdm_notebook,tqdm

warnings.filterwarnings('ignore')

import sys
import os

sys.path.insert(0, os.path.join(os.getcwd(), os.pardir))
from qutip import *
import numpy as np
import matplotlib.pyplot as plt

pyximport.install(setup_args={"include_dirs": np.get_include()})
import XXZZham as XXZZham
from XXZZham import add_high_energies, rotate_to_00_base
import random
import adiabatic_sim as asim
import time

import multiprocessing
import ctypes

mkl_rt = ctypes.CDLL('libmkl_rt.so')
mkl_get_max_threads = mkl_rt.mkl_get_max_threads
mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(multiprocessing.cpu_count())))
import os
import LH_tools as LHT

PRECISION = 2 ** -40

# Actual code
random_projections = []
for i in tqdm(range(100)):
    n = 4
    H1 = XXZZham.gen_random_XXZZham(n, 1)
    H2 = XXZZham.gen_random_XXZZham(n, 1)

    H1_he = add_high_energies(rotate_to_00_base(H1.get_commuting_term_ham()), 30)
    H2_cr = rotate_to_00_base(H2.get_commuting_term_ham())

    H2_cr_energies, H2_cr_ev = H2_cr.eigenstates(eigvals=4)
    H2_cr_groundspace = H2_cr_ev[0:4]

    H1_he_gs = H1_he.eigenstates(eigvals=1)[1][0]
    rand_proj = LHT.get_total_projection_size(H2_cr_groundspace, H1_he_gs)[0]
    random_projections.append(rand_proj)

print("Median:", 1/np.median(random_projections))
print("Mean:", 1/np.mean(random_projections))
plt.hist(random_projections,250)
plt.show()
