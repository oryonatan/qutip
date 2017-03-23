import sys
import os
import random

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from LH_tools import LocalOperator as LO
import numpy as np
import qutip

_sx = qutip.sigmax()
_sz = qutip.sigmaz()
_ID = qutip.qeye(2)


class XXZZ_term:
    def __init__(self, i, j, a):
        """
        Creates a a_ij*(X_i * X_j+Z_i * Z_j) hamiltonian where i < j
        :param i: first index
        :param j: second index (must be j>i)
        :param a: coefficient
        """

        if i >= j:
            raise ValueError("i must be strictly smaller than j")
        self.i = i
        self.j = j
        self.a = a

    def get_oper(self, degree=0):
        if degree == 0:
            return self.a * \
                   (LO({self.i: _sx, self.j: _sx}) +
                    LO({self.i: _sz, self.j: _sz}))
        else:
            return self.a * \
                   (LO({self.i: _sx, self.j: _sx}).force_d(degree) +
                    LO({self.i: _sz, self.j: _sz}))

    def get_commuting_form(self, max_degree):
        ihat = self.i + max_degree
        jhat = self.j + max_degree
        new_space_degree = 2 * max_degree
        X_i_hat = LO({self.i: _sx, ihat: _ID}).force_d(new_space_degree) + LO({self.i: _sz, ihat: _sx})
        X_j_hat = LO({self.j: _sx, jhat: _ID}).force_d(new_space_degree) + LO({self.j: _sz, jhat: _sx})
        Z_i_hat = LO({self.i: _sz, ihat: _ID}).force_d(new_space_degree) + LO({self.i: _sx, ihat: _sx})
        Z_j_hat = LO({self.j: _sz, jhat: _ID}).force_d(new_space_degree) + LO({self.j: _sx, jhat: _sx})
        return self.a * (X_i_hat * X_j_hat + Z_i_hat * Z_j_hat)


class XXZZham:
    def __init__(self, local_terms, degree=0):
        self.local_terms = local_terms
        if degree == 0:
            self.degree = max([term.j for term in local_terms])
        else:
            self.degree = degree

    def get_ham(self) -> qutip.Qobj:
        return qutip.Qobj(sum([term.get_oper(self.degree) for term in self.local_terms]))

    def get_commuting_term_ham(self) -> qutip.Qobj:
        return sum([term.get_commuting_form(self.degree) for term in self.local_terms])




def rotate_to_00_base(oper):
    """
    Assumes matrix of size 2^(2n) where
    rotates to the a base where the first 2^n vectors are
    {0,1}^n \tensor {0}^n
    the second is 2^n vectors areN
    {0,1}^n \tensor {0}^(n-1)\tensor 1
    ans so on
    untill the laste 2^n vectors which are
    {0,1}^n \tensor {1}^(n)
    :param oper: operator to rotate
    :return: rotated operator
    """
    space_size = oper.data.shape[0]
    oper_ar = oper.data.toarray()
    rotmat = np.zeros_like(oper_ar)
    for i in range(space_size):
        j = int((i % np.sqrt(space_size)) * np.sqrt(space_size) \
                + np.floor(i / np.sqrt(space_size)))
        rotmat[j][i] = 1
    rotated = rotmat.conj().dot(oper_ar.dot(rotmat))
    return qutip.Qobj(rotated, dims=oper.dims)


def rotate_to_evil_base(oper):
    """
    Assumes matrix of size 2^(2n) where
    rotates to the a base where the first 2^n vectors are
    00x00
    01x00
    10x00
    11x00
    00x11
    01x11
    10x11
    11x11
    00x01
    01x01
    10x01
    11x01
    00x10
    01x10
    10x10
    11x10
    until the last vector with odd number
    :param oper: operator to rotate
    :return: rotated operator
    """
    import pydevd
    pydevd.settrace('localhost', port=4000, stdoutToServer=True, stderrToServer=True)
    space_size = oper.data.shape[0]
    oper_ar = oper.data.toarray()
    rotmat = np.zeros_like(oper_ar)
    for j,i in enumerate(evil_base_get_next_number(int(np.log2(space_size)))):
        rotmat[j][i] = 1
    rotated = rotmat.conj().dot(oper_ar.dot(rotmat))
    return qutip.Qobj(rotated, dims=oper.dims)


def evil_base_get_next_number(base_len):
    """
    Returns the next number in the evil base
    :param base_len:
    :return:
    """
    part_len = int(base_len / 2)
    right_part = "0" * part_len
    left_part = "0" * part_len
    for i in range(2 ** base_len):
        if 0 == right_part.count("1") % 2 :
            print(left_part + right_part)
            yield int(left_part + right_part, base=2)
            left_part, overflow = inc_binary_string(left_part)
            if overflow :
                right_part,end = inc_binary_string(right_part)
                if end:
                    break
        else:
            right_part, end = inc_binary_string(right_part)
    for i in range(2 ** base_len):
        if 1 == right_part.count("1") % 2 :
            print(left_part + right_part)
            yield int(left_part + right_part, base=2)
            left_part, overflow = inc_binary_string(left_part)
            if overflow :
                right_part,_ = inc_binary_string(right_part)
                if not "0" in right_part:
                    break
        else:
            right_part, _ = inc_binary_string(right_part)


def inc_binary_string(bin_str):
    """
    Increases a a value of number represented as binary string
    :param bin_str:
    :return:
    """
    # if str is 11111
    if "0" not in bin_str:
        overflow = True
        return bin_str.replace("1", "0"), True
    overflow = False
    newstr = \
        str(
            bin(
                int(bin_str, base=2) + 1
            ))[2:].zfill(len(bin_str))
    return newstr, overflow


def add_high_energies(oper, big_value) -> qutip.Qobj:
    """
    Assuming an operator is rotated to a "nice" base, I will add values to its diagonal in outside the small
    top-leftsided box
    :param oper:
    :return:
    """
    space_size = oper.data.shape[0]
    oper_ar = oper.data.toarray()
    id_outside = np.identity(space_size)

    small_box_size = int(np.sqrt(space_size))
    id_outside[0:small_box_size, 0:small_box_size] = np.zeros((small_box_size, small_box_size))
    oper_ar += id_outside * big_value
    return qutip.Qobj(oper_ar, dims=oper.dims)


def gen_random_XXZZham(n:int,rand_range=1)->XXZZham:
    """
    Generates a random XXZZ ham on n qubits with parameter -rand_range < a_{ij} < rand_range 
    :param n: number of qubits in space
    :param rand_range: bounds for a_{ij}
    :return: a XXZZ ham
    """
    terms = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            a = random.uniform(-rand_range, rand_range)
            terms.append(XXZZ_term(i, j, a))
    return XXZZham(terms)


