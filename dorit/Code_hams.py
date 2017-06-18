import scipy
import random
import numpy as np
from qutip import Qobj
import LH_tools as LHT


class BinaryCode:
    """
    Code from z_2^n to z_2^m 
    """

    def __init__(self, code_matrix):
        self.m, self.n = code_matrix.shape
        self.code_matrix = code_matrix

    def encode(self, plaintext: np.ndarray) -> np.ndarray:
        if plaintext.shape != (self.n,):
            raise TypeError("expected plaintext dims: (%i,) got: %s instead" % (self.n, plaintext.shape))
        return self.code_matrix.dot(plaintext) % 2

    def __repr__(self):
        return repr(self.code_matrix)


def generate_random_code(n, m) -> BinaryCode:
    """
    Generates a random linear code matrix from z_2^n to z_2^m
    :param n: plaintext space dimension
    :param m: code space dimension
    :return: a random BinaryCode  
    """
    rank = 0
    while rank < n:
        code_matrix = np.random.randint(2, size=[m, n])
        # make sure this is injective(one to one)
        rank = np.linalg.matrix_rank(code_matrix)
    return BinaryCode(code_matrix)


def generate_random_code_hamiltonian(n, m) -> (Qobj, BinaryCode, int):
    """
    Generates a random linear code n->m, build it's weight hamiltonian and return the 
     minimal codeword weight
    :param n: plaintext space dimension
    :param m: code space dimension
    :return: ham, rand_code, min_hamming
    """
    rand_code = generate_random_code(n, m)
    ham, min_hamming = generate_code_hamiltonian(rand_code)
    return ham, rand_code, min_hamming


def generate_code_hamiltonian(from_code: BinaryCode):
    """
    Generates hamitonian from binary code hamiltonian and return the minimal codeword weight
    :param from_code: 
    :return:  hamiltonian, minimal hamming weight
    """
    n, m = from_code.n, from_code.m
    ham = scipy.sparse.identity(2 ** m).tolil() * 2 ** m
    min_hamming = 2 ** m * 2
    for word_int in range(1, 2 ** n):
        word_bin_array = np.array(list(bin(word_int)[2:].zfill(n)), dtype=int)
        code_word = from_code.encode(word_bin_array)
        hamming_weight = sum(code_word)
        min_hamming = min(min_hamming, hamming_weight)
        ham[word_int, word_int] = hamming_weight
    return Qobj(ham, dims=LHT.n_qubit_oper_dims(m)), min_hamming


def generate_noised_hamiltonian(from_code: BinaryCode, noise_magnitude: float):
    """
    Generates hamitonian from binary code hamiltonian and return the minimal codeword weight
    :param from_code: 
    :return:  hamiltonian, minimal hamming weight
    """
    n, m = from_code.n, from_code.m
    ham = scipy.sparse.identity(2 ** m).tolil() * 2 ** m
    min_hamming = 2 ** m * 2
    for word_int in range(1, 2 ** n):
        word_bin_array = np.array(list(bin(word_int)[2:].zfill(n)), dtype=int)
        code_word = from_code.encode(word_bin_array)
        hamming_weight = sum(code_word)
        min_hamming = min(min_hamming, hamming_weight)
        ham[word_int, word_int] = hamming_weight
    # noise the fast lil object (should be parlelized)
    for i in range(1, 2 ** m):
        noise = noise_magnitude *random.random() * random.randrange(-1, 2, 2)
        ham[i, i] = hamming_weight + noise
    return Qobj(ham, dims=LHT.n_qubit_oper_dims(m)), min_hamming


