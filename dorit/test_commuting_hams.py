import random

import matplotlib.pyplot as plt

import dorit.XXZZham as XXZZham
from dorit.XXZZham import rotate_to_00_base, add_high_energies
from LH_tools import plot_operator

if __name__ == '__main__':
    terms = []
    for i in range(1, 5):
        for j in range(i + 1, 5):
            a = random.uniform(-10, 10)
            terms.append(XXZZham.XXZZ_term(i, j, a))

    H = XXZZham.XXZZham(terms)
    plot_operator(H.get_ham())
    # plt.figure()
    H_com = H.get_commuting_term_ham()
    # plot_operator(H_com)
    # plt.figure()
    # plot_operator(rotate_to_00_base(H_com))
    plt.figure()
    high_energies = add_high_energies(rotate_to_00_base(H_com))
    # plot_operator(high_energies)
    plt.show()
    pass
