import warnings

import pyximport

warnings.filterwarnings('ignore')

import sys
import os

sys.path.insert(0, os.path.join(os.getcwd(), os.pardir))
from qutip import *
import numpy as np

pyximport.install(setup_args={"include_dirs": np.get_include()})
import XXZZham as XXZZham
from XXZZham import add_high_energies, rotate_to_00_base
import random
import adiabatic_sim as asim
import time

import ctypes

mkl_rt = ctypes.CDLL('libmkl_rt.so')
mkl_get_max_threads = mkl_rt.mkl_get_max_threads
mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(4)))
import os

n = 4

while True:
    def create_vector_from_string(vec_to_build: str) -> Qobj:
        """ Creates a vector from 0/1 string - indicator in the computational basis
            i.e. the string 010 will create the vector |010>

        """
        to_tensor = []
        for digit in vec_to_build:
            if digit == '0':
                to_tensor.append(basis(2, 0))
            elif digit == '1':
                to_tensor.append(basis(2, 1))
            else:
                raise ValueError("String should consist only of 0 and 1")
        return tensor(to_tensor)


    def create_all_even_vectors(n):
        ret = []
        for i in range(2 ** n):
            binformat = bin(i)[2:].zfill(n)
            if 0 == binformat.count('0') % 2:
                ret.append(create_vector_from_string(binformat))
        return ret


    evens = create_all_even_vectors(n)
    IDeven = sum(v * v.trans() for v in evens)
    PRECISION = 2 ** -40

    alist = []
    terms = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            a = random.uniform(-10, 10)
            alist.append(a)
            terms.append(XXZZham.XXZZ_term(i, j, a))

    H = XXZZham.XXZZham(terms)
    H_com = H.get_commuting_term_ham()
    H_highE = add_high_energies(rotate_to_00_base(H_com), 30)
    h_com_rot = rotate_to_00_base(H_com)

    ID2n = tensor([qeye(2)] * n * 2)
    HE_ev = H_highE.eigenstates(eigvals=1)[1][0]
    P_HE = ID2n - HE_ev * HE_ev.trans()
    HCR_en, HCR_ev = h_com_rot.eigenstates(eigvals=10)
    HCR_degeneracy = sum(abs(HCR_en - HCR_en.min()) < PRECISION)
    HCR_groundspace = HCR_ev[0:HCR_degeneracy]
    # _, HCR_groundspace = h_com_rot.eigenstates(eigvals=HCR_degeneracy)
    P_CR = ID2n - sum([gs * gs.trans() for gs in HCR_groundspace])

    # prepare the first state for CLH-HIGHE evolution
    tlist = np.linspace(0, 10, 10)
    P99 = P_HE * 0.01 + P_CR * 0.99
    psi0_99 = P99.eigenstates(eigvals=1)[1][0]

    P_mat, _, psis = asim.sim_degenerate_adiabatic(tlist, P99, P_CR, psi0_99, 10)
    psifinal_from_99 = psis[-1]  # should be the GS we get from evolution to P_CR starting in 99
    psi0_small = H.get_ham().eigenstates(eigvals=1)[1][0]

    uniform_over_left_0000 = sum(create_all_even_vectors(n))
    uniform_over_left_0000/=uniform_over_left_0000.norm()


    left_extended_to_even_psi0 = tensor([uniform_over_left_0000, psi0_small])

    print(abs(left_extended_to_even_psi0.overlap(psifinal_from_99)))
