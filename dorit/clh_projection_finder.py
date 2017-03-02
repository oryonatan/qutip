from qutip import *
import numpy as np
import LH_tools as LHT
from numpy import linalg as LA


def find_projections_maximal_on_even(gs: [Qobj], gamma: Qobj, lh_gs: Qobj, proximity: float):
    """
    Find the vector that maximizes abs(tr(I_{even}\otimes \Pi))
    :param gs:
    :param gamma:

    :return:
    """
    n = int(np.log2(lh_gs.shape[0]))
    Proj_gLSH = tensor([tensor([qeye(2)] * n) - lh_gs * lh_gs.trans()])
    evens = LHT.create_all_even_vectors(n)
    IDeven = sum(v * v.trans() for v in evens)

    # Randomly select a vecrtor in C4 and normalize it
    coefficients = np.random.rand(8)
    coefficients /= LA.norm(coefficients)
    v = (coefficients[0] + coefficients[1] * 1j * gs[0] +
         coefficients[2] + coefficients[3] * 1j * gs[1] +
         coefficients[4] + coefficients[5] * 1j * gs[2] +
         coefficients[6] + coefficients[7] * 1j * gs[3])
    # TODO:  implement
    D = generate_diffrential(v,coefficients,)
    # TODO: is this actually what I should find ?
    unnormalized_even_PgsLH = abs((tensor([Proj_gLSH, IDeven]) * v).tr())
    even_P_normalization = abs((tensor([tensor([qeye(2)] * n), IDeven]) * v).tr())
    even_PgsLH = unnormalized_even_PgsLH / even_P_normalization
