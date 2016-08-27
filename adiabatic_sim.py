from qutip import *
from scipy.linalg import expm

def sim_simple_adiabatic(tlist, H0, H1, s='linear'):
    """

    :param tlist: Time list
    :param H0: first hamiltonian
    :param H1: second hamiltonian
    :param s: function - relates time to coupling, default is linear dependecy
    :return:
    """
    if s == 'linear':
        s = lambda t: (t - tmin) / (tmax - tmin)
    duration = len(tlist)
    tmin = min(tlist)
    tmax = max(tlist)
    # start at H0 ground state
    eigvals, eigvecs = H0.eigenstates(eigvals=2)
    P_mat = []
    eigvals_mat = []
    psi = eigvecs[0]
    psis = [psi]
    eigvals_mat.append(eigvals)

    P_mat.append(
        [abs(eigvecs[0].overlap(psi)) ** 2,
         abs(eigvecs[1].overlap(psi)) ** 2])
    oldt = tmin
    for t in tlist[1:]:
        dt = oldt - t
        Ht = H0 * (1 - s(t)) + H1 * (s(t))
        eigvals, eigvecs = Ht.eigenstates(eigvals=2)
        U = expm(-1j * Ht.data * dt)
        psi = Qobj(U * psi.data, dims=psi.dims)
        psis.append(psi)
        eigvals_mat.append(eigvals)
        P_mat.append(
            [abs(eigvecs[0].overlap(psi)) ** 2,
             abs(eigvecs[1].overlap(psi)) ** 2])
        oldt = t
    return P_mat, eigvals_mat, psis
