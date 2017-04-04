import warnings
import matplotlib.pyplot as plt
import pyximport
from tqdm import tnrange, tqdm_notebook, tqdm

warnings.filterwarnings('ignore')

import sys
import os
import time

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

number_of_hams_to_try = 250

random_projections4 = []
for i in tqdm(range(number_of_hams_to_try)):
    n = 4
    zer = tensor([qutip.basis(2, 0)] * n)
    IDn = LHT.gen_ID_n(n)

    # Generate some random vector
    randH = qutip.rand_unitary(2 ** n, dims=IDn.dims)
    psi = randH * zer

    # Generate some random 4 subspace
    subspace = []
    for i in range(n):
        randH = qutip.rand_unitary(2 ** n, dims=IDn.dims)
        subspace.append(randH * zer)
    rand_proj = LHT.get_total_projection_size(subspace, psi)[0]
    random_projections4.append(rand_proj)

random_projections8 = []
for i in tqdm(range(number_of_hams_to_try)):
    n = 8
    zer = tensor([qutip.basis(2, 0)] * n)
    IDn = LHT.gen_ID_n(n)

    # Generate some random vector
    randH = qutip.rand_unitary(2 ** n, dims=IDn.dims)
    psi = randH * zer

    # Generate some random 4 subspace
    subspace = []
    for i in range(n):
        randH = qutip.rand_unitary(2 ** n, dims=IDn.dims)
        subspace.append(randH * zer)
    rand_proj = LHT.get_total_projection_size(subspace, psi)[0]
    random_projections8.append(rand_proj)

print("Random hams")
print("Median:", 1 / np.median(random_projections4))
print("Mean:", 1 / np.mean(random_projections4))
print("Same hamiltonian ")
print("Median:", 1 / np.median(random_projections8))
print("Mean:", 1 / np.mean(random_projections8))

LHT.plot_two_histograms("Random hams",
                        random_projections4,
                        "Same hamiltonian",
                        random_projections8)
savefile_name = "figs/" + time.strftime("%d.%m.%y:%H:%M:%S", time.gmtime()) + ".png"
plt.savefig(savefile_name)
try:
    plt.show()
except Exception as e:
    print("Failed to draw", e)
