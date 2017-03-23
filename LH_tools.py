from numpy import pi

from qutip import *
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import smtplib
import scipy.linalg
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart

from qutip import Qobj
from typing import Tuple
import collections


def s_function(t, N=1024, epsilon=0.1):
    """Computes the rate function s(t)"""
    nomerator = np.sqrt(N - 1) * np.tan((2 * t * epsilon * np.sqrt(N - 1) - N * np.arctan(np.sqrt(N - 1))) / N)
    denominator = 1 - N
    return 1 / 2 * (1 - nomerator / denominator)


def find_s_one_numeric(s_function, N, epsilon):
    # We use hinge function to prevent negative solutions
    s_zero = lambda x: max(1 - s_function(x, N, epsilon), 2 - x)
    # Find closest solution to 0 for 1-s(t)=0
    return fsolve(s_zero, 0)[0]


# Or
def find_s_one(N, epsilon):
    return 1 / (2 * epsilon) * N / np.sqrt((N - 1)) * 2 * np.arctan(np.sqrt(N - 1))


def SendMail(subject, text, figure=False):
    From = 'rashbam.mailer@gmail.com'
    To = ['or.yonatan@gmail.com']
    if figure:
        figure.savefig("fig.png")
        img_data = open("fig.png", 'rb').read()
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = 'or.yonatan@gmail.com'
    msg['To'] = 'or.yonatan@gmail.com'

    text = MIMEText(text)
    msg.attach(text)
    if figure:
        image = MIMEImage(img_data, name=os.path.basename("fig.png"))
        msg.attach(image)
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login("rashbam.mailer@gmail.com", "nv8ZqTi1oSUYSfN5DS$r")
    s.sendmail(From, To, msg.as_string())
    s.quit()


# Speed test

def benchmark(h_t, in_state, steps=100, T=100):
    """Run simulation for the given number of steps and returns P_mat, Evals_mat and time to compute
    """
    start_time = time.time()
    tlist = np.linspace(0, T, steps)
    args = {'t_max': max(tlist)}
    P_mat, evals_mat, psis = simulate_adiabatic_process2(tlist, h_t, args, in_state, False)
    plot_PandEV(P_mat, evals_mat, tlist)
    print("Computation time %s seconds :" % (time.time() - start_time))
    return P_mat, evals_mat, psis, time.time() - start_time


def plot_PandEV(P_mat, EV_mat, tlist, figsize=(15, 5)):
    fig = plt.figure(figsize=figsize)
    P_plt = fig.add_subplot(1, 2, 1)
    P_plt.set_title("Occupation probabilities")
    P_plt.plot(tlist, P_mat)
    P_plt.set_ylim(0, 1.1)
    ev_plt = fig.add_subplot(1, 2, 2)
    ev_plt.set_title("Eigenvalues")
    ev_plt.plot(tlist, EV_mat)
    return fig


"""

def plot_PandEV(P_mat, EV_mat, tlist, figsize=(15,5)):
    f,(P_plt,ev_plt) = plt.subplots(1, 2, figsize=figsize)
    P_plt.set_title("Occupation probabilities")
    P_plt.plot(tlist, P_mat)
    ev_plt.set_title("Eigenvalues")
    ev_plt.plot(tlist, EV_mat)
    
"""


def find_gap(evals_mat):
    """Finds the minimal gap in a list of eiganvalue pairs"""
    min_gap = float("inf")
    for pair in evals_mat:
        dist = abs(pair[0] - pair[1])
        min_gap = min(dist, min_gap)
    return min_gap


def gen_SAT_LH():
    zeromat = basis(2, 0) * basis(2, 0).trans()
    oneomat = basis(2, 1) * basis(2, 1).trans()
    zero_one = tensor(basis(2, 0), basis(2, 1))  # |01>
    zero_onemat = zero_one * zero_one.trans()
    hamiltonian_terms = []
    term1_hamiltonian = tensor(zero_onemat, qeye(2))
    hamiltonian_terms.append(term1_hamiltonian)
    hamiltonian_terms.append(tensor(qeye(2), zero_onemat))
    hamiltonian_terms.append(tensor(oneomat, qeye(2), zeromat))
    hamiltonian_terms.append(tensor(zeromat, qeye(2), qeye(2)))
    return sum(hamiltonian_terms)


def gen_simple_ham(n):
    zero3 = tensor([basis(2, 0)] * n)
    zero3_mat = zero3 * zero3.trans()
    id3 = tensor([qeye(2)] * n)
    simple_ham = id3 - zero3_mat
    return simple_ham, simple_ham.groundstate()[1]


def rotate_by_had(in_ham, groundstate):
    n = len(in_ham.dims[0])
    rot_ham = tensor([hadamard_transform()] * n) * in_ham * tensor([hadamard_transform()] * n)
    rot_gs = tensor([hadamard_transform()] * n) * groundstate
    return rot_ham, rot_gs


def simulate_adiabatic_process2(tlist, h_t, args, rho0, draw, options=Options()):
    #
    # callback function for each time-step
    #
    N = 2
    M = 2
    evals_mat = np.zeros((len(tlist), M))
    P_mat = np.zeros((len(tlist), M))
    psis = []
    idx = [0]

    def process_rho(tau, psi):

        # evaluate the Hamiltonian with gradually switched on interaction
        H = Qobj.evaluate(h_t, tau, args)

        evals, ekets = H.eigenstates(eigvals=M)
        evals_mat[idx[0], :] = np.real(evals)
        # print(psi)
        # find the overlap between the eigenstates and psi
        for n, eket in enumerate(ekets):
            P_mat[idx[0], n] = abs((eket.dag().data * psi.data)[0, 0]) ** 2
        psis.append(psi)
        idx[0] += 1

    output = qutip.sesolve(H=h_t,
                           rho0=rho0,
                           tlist=tlist,
                           e_ops=process_rho,
                           args=args,
                           options=options)
    # rc('font', family='serif')
    # rc('font', size='10')
    if draw:
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        #
        # plot the energy eigenvalues
        #

        # first draw thin lines outlining the energy spectrum
        for n in range(len(evals_mat[0, :])):
            ls, lw = ('b', 1) if n == 0 else ('k', 0.25)
            axes[0].plot(tlist / max(tlist), evals_mat[:, n] / (2 * pi), ls, lw=lw)

        # second, draw line that encode the occupation probability of each state in
        # its linewidth. thicker line => high occupation probability.
        for idx in range(len(tlist) - 1):
            for n in range(len(P_mat[0, :])):
                lw = 0.5 + 4 * P_mat[idx, n]
                if lw > 0.55:
                    axes[0].plot(np.array([tlist[idx], tlist[idx + 1]]) / max(tlist),
                                 np.array([evals_mat[idx, n], evals_mat[idx + 1, n]]) / (2 * pi),
                                 'r', linewidth=lw)
        axes[0].set_xlabel(r'$\tau$')
        axes[0].set_ylabel('Eigenenergies')
        axes[0].set_title("Energyspectrum (%d lowest values) of a chain of %d spins.\n " % (M, N)
                          + "The occupation probabilities are encoded in the red line widths.")

        #
        # plot the occupation probabilities for the few lowest eigenstates
        #
        for n in range(len(P_mat[0, :])):
            if n == 0:
                axes[1].plot(tlist / max(tlist), 0 + P_mat[:, n], 'r', linewidth=2)
            else:
                axes[1].plot(tlist / max(tlist), 0 + P_mat[:, n])

        axes[1].set_xlabel(r'$\tau$')
        axes[1].set_ylabel('Occupation probability')
        axes[1].set_title("Occupation probability of the %d lowest " % M +
                          "eigenstates for a chain of %d spins" % N)
        axes[1].legend(("Ground state",))

    return P_mat, evals_mat, psis


def plot_operator(operator, vmin='not set', vmax='not set'):
    if vmin == 'not set':
        vmin = np.amin(np.real(operator.data.toarray()))
    if vmax == 'not set':
        vmax = np.amax(np.real(operator.data.toarray()))

    data = operator.data.toarray()
    data = np.ma.masked_where(abs(data) < 0.00000001, data)
    cmap = plt.cm.nipy_spectral
    cmap.set_bad(color='whitesmoke')
    plt.imshow(np.real(data),
               interpolation='nearest', vmin=vmin, vmax=vmax, cmap=cmap)


def plot_commutations(op1, op2, figsize=(15, 5)):
    com = op1 * op2 - op2 * op1
    vmin = min(np.amin(np.real(op1.data.toarray())),
               np.amin(np.real(op2.data.toarray())),
               np.amin(np.real(com.data.toarray())))
    vmax = max(np.amax(np.real(op1.data.toarray())),
               np.amax(np.real(op2.data.toarray())),
               np.amax(np.real(com.data.toarray())))
    fig = plt.figure(figsize=figsize)
    fig.suptitle("Real part only")
    op1_plt = fig.add_subplot(1, 3, 1)
    op1_plt.set_title("First Op")
    plot_operator(op1, vmin=vmin, vmax=vmax)
    op2_plt = fig.add_subplot(1, 3, 2)
    op2_plt.set_title("Second Op")
    plot_operator(op2, vmin=vmin, vmax=vmax)
    com_plt = fig.add_subplot(1, 3, 3)
    com_plt.set_title("Commutation relation")
    plot_operator(op1 * op2 - op2 * op1, vmin=vmin, vmax=vmax)


class LocalOperator:
    def __init__(self, dict_of_ops):
        # list of 1 qubit ops and indexes of the shape (op, i)
        # for example {1:sigmaz(), 5: sigmax(), 10:hadamard()}
        self.dict_of_ops = dict_of_ops
        self.update_d()

    def update_d(self):
        if len(self.dict_of_ops) == 0:
            self.d = 0
        else:
            self.d = max(self.dict_of_ops.keys())

    def full_form(self, n=None):
        if n == None: n = self.d

        full_list = []
        for i in range(1, n + 1):
            if i in self.dict_of_ops.keys():
                full_list.append(self.dict_of_ops[i])
            else:
                full_list.append(qeye(2))
        return tensor(full_list)

    def tensor(self, other):
        # new is copy of old
        new = LocalOperator(self.dict_of_ops.copy())
        for index, op in other.dict_of_ops.items():
            if index in new.dict_of_ops.keys():
                new.dict_of_ops[index] *= other.dict_of_ops[index]
            else:
                new.dict_of_ops[index] = other.dict_of_ops[index]
        new.update_d()
        return new

    def __mul__(self, other):
        return self.tensor(other)

    def force_d(self, degree):
        new = LocalOperator(self.dict_of_ops.copy())
        if degree not in new.dict_of_ops.keys():
            new.dict_of_ops[degree] = qeye(2)
            new.update_d()
        return new

    def __sub__(self, other):
        deg = max(self.d, other.d)
        return self.full_form(deg) - other.full_form(deg)
        # new = LocalOperator(self.dict_of_ops.copy())
        # if (sorted(self.dict_of_ops.keys()) !=
        #         sorted(other.dict_of_ops.keys())):
        #     raise TypeError('Local operator operate on different qubits')
        #
        # for index, op in other.dict_of_ops.items():
        #     new.dict_of_ops[index] -= other.dict_of_ops[index]
        #
        # new.dict_of_ops = {x: y for (x, y) in new.dict_of_ops.items() if y.norm() != 0}
        # new.update_d()
        # return new

    def __add__(self, other):
        deg = max(self.d, other.d)
        return self.full_form(deg) + other.full_form(deg)

        # if (sorted(self.dict_of_ops.keys()) !=
        #         sorted(other.dict_of_ops.keys())):
        #     raise TypeError('Local operator operate on different qubits')
        #
        # for index, op in other.dict_of_ops.items():
        #     new.dict_of_ops[index] += other.dict_of_ops[index]
        #
        # new.dict_of_ops = {x: y for (x, y) in new.dict_of_ops.items() if y.norm() != 0}
        # new.update_d()
        # return new

    def norm(self):
        if self.d == 0: return 0
        ret = 1
        for op in self.dict_of_ops.values():
            ret *= op.norm()
        return ret

    def __repr__(self):
        return str(self.dict_of_ops)


def make_pair_orthonormal(v1: qobj, v2: qobj) -> (qobj, qobj):
    """
    Uses gramm shmidt like process to take two vectors and return two orthonormal vectors
    :param v1:
    :param v2:
    :return: (a,b) two orthonormal vectors
    """
    a = v1 / v1.norm()
    aproj = a.overlap(v2) * a
    b_non_norm = (v2 - aproj)
    b = b_non_norm / b_non_norm.norm()
    return a, b


def get_total_projection_size(subspace: Tuple[Qobj], psi: Qobj) -> float:
    """
    Finds the total projection size over a subspace defined by a list of vectors (not necessarily orthogonal)
    :param subspace:
    :param psi:
    :return:
    """
    subspace_mat = np.concatenate([vector.data.toarray() for vector in subspace], axis=1)
    # use gram shmidt QR factorization
    Q, _ = np.linalg.qr(subspace_mat)

    projection_vector = np.abs(
        Q.transpose().dot(psi.data.toarray())
    ) ** 2
    return sum(projection_vector)


# Finds the minimal angle between two subsapces
# I think

def subspace_angle(A: Tuple[Qobj], B: Tuple[Qobj]) -> (float, str):
    """
    Computes angle between subspace A and B
    :param A:
    :param B:
    :return: angle in rads, string - angle as parts of pi
    """
    inner_product_mat = np.zeros([len(A), len(B)])
    for i in range(len(A)):
        for j in range(len(B)):
            inner_product_mat[i][j] = A[i].overlap(B[j].trans().conj())

    rads = (np.arccos(np.amax(inner_product_mat)))
    deg_as_str = "π/%s" % (np.pi / rads)
    return rads, deg_as_str


def find_degeneracy(H, precision=10 ** -10) -> int:
    """
    Return the degree of the gorundsaoce of H
    :param H: hamitonian either Qobj or hermitian scipy.sparse.csr.csr_matrix
    :param precision: precision parameter, all energies < min(energies)+ precision will be considered ground energies
    :return: dimension of groundspace.
    """
    if type(H) == qutip.qobj.Qobj:
        energies = scipy.linalg.eigh(H.data.toarray())[0]
    elif type(H) == scipy.sparse.csr.csr_matrix:
        energies = scipy.linalg.eigh(H.toarray())[0]
    elif type(H) == numpy.ndarray:
        energies = scipy.linalg.eigh(H)[0]
    return sum(abs(energies - energies.min()) < precision)


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


def create_all_even_vectors(n) -> [Qobj]:
    """
    Creates an array of all the vectors in the computational basis with an even number of 1's
    :param n: basis size
    :return: array
    """
    ret = []
    for i in range(2 ** n):
        binformat = bin(i)[2:].zfill(n)
        if 0 == binformat.count('0') % 2:
            ret.append(create_vector_from_string(binformat))
    return ret


def create_all_odd_vectors(n) -> [Qobj]:
    """
    Creates an array of all the vectors in the computational basis with an odd number of 1's
    :param n: basis size
    :return: array
    """
    ret = []
    for i in range(2 ** n):
        binformat = bin(i)[2:].zfill(n)
        if 1 == binformat.count('0') % 2:
            ret.append(create_vector_from_string(binformat))
    return ret


def proj_on(subspace) -> Qobj:
    """
    Creates a projector on the space given
    :param subspace: if a list of vectors - should find a basis and project onto
            if a single vector - creates a projector on the it
    :return: a projection operator on the subspace
    """
    if isinstance(subspace, collections.Sequence):
        # use QR factorization to get an orthogonal basis
        subspace_mat = np.concatenate([vector.data.toarray() for vector in subspace], axis=1)
        Q, _ = np.linalg.qr(subspace_mat)
        proj_arr = Q.dot(Q.T.conj())
        proj = Qobj(proj_arr,dims=[subspace[0].dims[0]]*2)
    else:
        proj = subspace * subspace.trans().conj()
    return proj


def proj_orth(subspace) -> Qobj:
    """
    Creates a projector subspace orthogonal to the given subspace in the same Hilbert space
    :param subspace: if a list of vectors - should find a basis and project onto
            if a single vector - creates a projector on the it
    :return: a projection operator on the subspace
    """
    proj = proj_on(subspace)
    Idn = tensor([qeye(2)] * len(proj.dims[0]))
    orthproj = Idn - proj
    return orthproj
