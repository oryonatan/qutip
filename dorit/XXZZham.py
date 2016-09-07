from LocalOperator import LocalOperator as LO
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
        return qutip.Qobj(sum([term.get_oper(degree) for term in self.local_terms]))

    def get_commuting_term_ham(self):
        return sum([term.get_commuting_form(self.degree) for term in self.local_terms])
