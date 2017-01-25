import warnings

import pyximport

warnings.filterwarnings('ignore')

import sys
import os

sys.path.insert(0, os.path.join(os.getcwd(), os.pardir))
from qutip import *
import numpy as np

pyximport.install(setup_args={"include_dirs": np.get_include()})
import dorit.XXZZham as XXZZham
from dorit.XXZZham import add_high_energies, rotate_to_00_base
import random
import adiabatic_sim as asim


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


n = 5
evens = create_all_even_vectors(n)
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
    HCR_en = h_com_rot.eigenenergies()
    HCR_degeneracy = sum(abs(HCR_en - HCR_en.min()) < PRECISION)
    _, HCR_groundspace = h_com_rot.eigenstates(eigvals=HCR_degeneracy)
    P_CR = ID2n - sum([gs * gs.trans() for gs in HCR_groundspace])

    # prepare the first state for CLH-HIGHE evolution
    tlist = np.linspace(0, 10, 10)
    P99 = P_HE * 0.01 + P_CR * 0.99
    psi0_99 = P99.eigenstates(eigvals=1)[1][0]

    P_mat, _, psis = asim.sim_degenerate_adiabatic(tlist, P99, P_CR, psi0_99)
    print(P_mat[-1][0], "this should be close to one")
    psifinal_from_99 = psis[-1]  # should be the GS we get from evolution to P_CR starting in 99

    #DEBUG prints:
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
    print (alist)
    print("Unormalized projection (GLSH+EVEN\t\t\t%f" % unnormalized_even_PgsLH)
    print("Even projection \t\t\t\t\t\t\t%f" % even_P_normalization)
    print("Nonrmalized projection (GLSH+EVEN)/EVEN \t%f" % (unnormalized_even_PgsLH/even_P_normalization))
    print("==========================================================================")
    if unnormalized_even_PgsLH/even_P_normalization < 0.3 :
        print ("low ")