import sys
import os

sys.path.insert(0, os.path.join(os.getcwd(), os.pardir))
from qutip import *
import numpy as np

import multiprocessing
import ctypes

mkl_rt = ctypes.CDLL('libmkl_rt.so')
mkl_get_max_threads = mkl_rt.mkl_get_max_threads
mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(multiprocessing.cpu_count())))
import os
import LH_tools as LHT


def prepare_groverlike_system(n: int) -> [Qobj, Qobj, Qobj]:
    """
    creates a 2x2 system that simulates grover dynamic with search state of |0^n> and start state which is the uniform
    superposition state
    @params n: number of qubits
    returns H0, H1 two 2x2 hamiltonians that represents the evolution in the of the two basis vectors that will rotate
    under a quench
    returns psi0 - eigenstate of H0
    """
    c = np.sqrt((2 ** n - 1) / 2 ** n)
    offdiag = (-1 / np.sqrt(2 ** n) + 1 / (2 ** (n + 1))) / c
    H0 = Qobj([[1 - 1 / 2 ** n, offdiag],
               [offdiag, 1 / 2 ** n]])
    psi0 = Qobj(np.array([1 / np.sqrt(2 ** n), (1 - 1 / 2 ** n) / c]))
    H1 = Qobj([[0, 0],
               [0, 1]])
    return H0, H1, psi0


def create_back_and_forward_props(tlist, H_0, H_1):
    """

    Creates a back and forwared propagators
    :param tlist:
    :param H_0:
    :param H_1:
    :return:
    """
    n = len(H_0.dims[0])
    prop = tensor([qeye(2)] * n)
    tmax = max(tlist)
    last_time = 0
    for time in tlist:
        dt = abs(time - last_time)
        s = time / tmax
        Hs = H_0 * (1 - s) + H_1 * s
        U = (-1j * Hs * dt).expm()
        prop = U * prop
        last_time = time
    last_time = 0
    backprop = tensor([qeye(2)] * n)
    for time in tlist[1:]:
        dt = abs(time - last_time)
        s = time / tmax
        Hs = H_0 * (s) + H_1 * (1 - s)
        U = (-1j * Hs * dt).expm()
        backprop = U * backprop
        last_time = time
    return backprop, prop


def create_back_and_forward_props_sfunction(s, tlist, H_0, H_1):
    """
    Creates a back and forwared propagators
    :param tlist:
    :param H_0:
    :param H_1:
    :return:
    """
    n = len(H_0.dims[0])
    prop = tensor([qeye(2)] * n)
    slist = s(tlist)
    tmax = max(tlist)
    last_time = 0
    for time, s_t in zip(tlist, slist):
        dt = abs(time - last_time)
        Hs = H_0 * (1 - s_t) + H_1 * s_t
        U = (-1j * Hs * dt).expm()
        prop = U * prop
        last_time = time
    last_time = 0
    backprop = tensor([qeye(2)] * n)
    for time, s_t in zip(tlist[1::-1], slist[1::-1]):
        dt = abs(time - last_time)
        Hs = H_0 * (1 - s_t) + H_1 * s_t
        U = (-1j * Hs * dt).expm()
        backprop = U * backprop
        last_time = time
    return backprop, prop


def _simulate_time_p_and_pfab(T: float, steps: int, psi0: Qobj, psi1: Qobj, H0: Qobj, H1: Qobj):
    """
        Callback for simulation of fowrard and backword propagation
    :param T:
    :param steps:
    :param psi0:
    :param psi1:
    :param H0:
    :param H1:
    :return:
    """
    tlist = np.linspace(0, T, steps)
    backprop_t, prop_t = create_back_and_forward_props(tlist, H0, H1)
    psi_forward = prop_t * psi0
    psi_fab = backprop_t * psi_forward
    Pf0 = abs(psi1.overlap(psi_forward)) ** 2
    Pfab = abs(psi0.overlap(psi_fab)) ** 2
    return T, Pf0, Pfab
