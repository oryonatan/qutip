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
import LH_tools as LHT
import adiabatic_sim as asim
import time

import ctypes
mkl_rt = ctypes.CDLL('libmkl_rt.so')
mkl_get_max_threads = mkl_rt.mkl_get_max_threads
mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(48)))
import os

<<<<<<< HEAD
n = 6
=======

<<<<<<< HEAD
n = 5 
=======
n = 6 
>>>>>>> afa916d359e34cf6f69e65ae2b46863faf95d254
>>>>>>> 89ec7dfcdb533a734a72b6e96ca5ab491848b5df
OUTPUT_PATH = "/cs/labs/doria/oryonatan/qutip/simulation_outputs/"
OUTPUT_FILENAME = os.path.join(OUTPUT_PATH, time.ctime().replace(' ', '_') + "n_%d" % n)


evens = LHT.create_all_even_vectors(n)
IDeven = sum(v * v.trans() for v in evens)
PRECISION = 2 ** -40

for _ in range(1000):
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
    HCR_en,HCR_ev = h_com_rot.eigenstates(eigvals=10)
    HCR_degeneracy = sum(abs(HCR_en - HCR_en.min()) < PRECISION)
    HCR_groundspace = HCR_ev[0:HCR_degeneracy]
    #_, HCR_groundspace = h_com_rot.eigenstates(eigvals=HCR_degeneracy)
    P_CR = ID2n - sum([gs * gs.trans() for gs in HCR_groundspace])

    # prepare the first state for CLH-HIGHE evolution
    tlist = np.linspace(0, 10, 10)
    P99 = P_HE * 0.01 + P_CR * 0.99
    psi0_99 = P99.eigenstates(eigvals=1)[1][0]

    P_mat, _, psis = asim.sim_degenerate_adiabatic(tlist, P99, P_CR, psi0_99, 10)
    print(P_mat[-1][0], "this should be close to one")
    psifinal_from_99 = psis[-1]  # should be the GS we get from evolution to P_CR starting in 99

    # DEBUG prints:
    # print("PCR ev0 %f" % P_CR.eigenenergies(eigvals=1))
    # print("\t actual energy %f" % expect(P_CR, psifinal_from_99))
    # print("CR ev0 %f" % h_com_rot.eigenenergies(eigvals=1))
    # print("\t actual energy %f " % expect(h_com_rot, psifinal_from_99))


    """
     I want to compute the prjection over the even-right-half-space , i.e. vectors of the form |0000....> ,|0011....>
     |0110....> , |1100....>, |1001....> , |1010....> , |1100....> |1111....>
     """
    psi0_small = H.get_ham().eigenstates(eigvals=1)[1][0]
    Proj_gLSH = tensor([tensor([qeye(2)] * n) - psi0_small * psi0_small.trans()])
    # |g><g| where |g> is the GS of P1 - Proj over the HighE GS where we choose only the GS that we can evolve to
    gamma = psifinal_from_99 * psifinal_from_99.trans()

    unnormalized_even_PgsLH = abs((tensor([Proj_gLSH, IDeven]) * gamma).tr())
    even_P_normalization = abs((tensor([tensor([qeye(2)] * n), IDeven]) * gamma).tr())
    with open(OUTPUT_FILENAME, 'a') as outfile:
        outfile.write("Unormalized projection (GLSH+EVEN)\t\t\t%f" % unnormalized_even_PgsLH)
        outfile.write("Even projection \t\t\t\t\t\t\t%f" % even_P_normalization)
        outfile.write("Normalized projection (GLSH+EVEN)/EVEN \t%f" % (unnormalized_even_PgsLH / even_P_normalization))
        outfile.write("==========================================================================")

    print(alist)
    print("Unormalized projection (GLSH+EVEN\t\t\t%f" % unnormalized_even_PgsLH)
    print("Even projection \t\t\t\t\t\t\t%f" % even_P_normalization)
    print("Normalized projection (GLSH+EVEN)/EVEN \t%f" % (unnormalized_even_PgsLH / even_P_normalization))
    print("==========================================================================")
    if unnormalized_even_PgsLH / even_P_normalization < 0.3:
        print("low ")
