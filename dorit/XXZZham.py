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

    def get_ham(self):
        return qutip.Qobj(sum([term.get_oper(self.degree) for term in self.local_terms]))

    def get_commuting_term_ham(self):
        return sum([term.get_commuting_form(self.degree) for term in self.local_terms])


def rotate_to_00_base(oper):
    """
    Assumes matrix of size 2^(2n) where
    rotatets to the a base where the first 2^n vectors are
    {0,1}^n \tensor {0}^n
    the second is 2^n vectors are
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
