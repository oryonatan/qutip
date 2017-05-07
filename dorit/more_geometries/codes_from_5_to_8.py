import warnings

import pyximport

warnings.filterwarnings('ignore')

import sys
import os
sys.path.insert(0,os.path.join(os.getcwd(),os.pardir))
from qutip import *
import numpy as np
pyximport.install(setup_args={"include_dirs":np.get_include()})
import adiabatic_sim as asim
import Code_hams as CH
import multiprocessing
import ctypes

mkl_rt = ctypes.CDLL('libmkl_rt.so')
mkl_get_max_threads = mkl_rt.mkl_get_max_threads
mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(multiprocessing.cpu_count())))
import LH_tools as LHT


n, m = 5,8
# generate the uniform-perp projector
zero_m = LHT.create_vector_from_string("0" * m )
uniform_m = hadamard_transform(m) * zero_m
H0_U = LHT.proj_orth(uniform_m)

# Generate H0 as sigma |->_i
terms = []
plus = (basis(2,0)+basis(2,1))/2**(1/2)
minus = (basis(2,0)-basis(2,1))/2**(1/2)
proj_minus = LHT.proj_on(minus)
for i in range(1,m+1):
    terms.append(LHT.LocalOperator({i:proj_minus}).full_form(m))
H0_sum_minus = sum(terms)


print("Calulating minimum adiabatic time")
print("{0:20}{1:20}{2:20}{3:20}".format("Uniform to code", "Uniform to noise", "Sum to code", "Sum to noise") )

while(True):
    # generate our code
    Hcode,code,min_weight = CH.generate_random_code_hamiltonian(n, m)
    Hnoise,min_weight_noised = CH.generate_noised_hamiltonian(code,np.sqrt(m))

    mingap_U_noise = asim.find_global_adiabatic_rate( H0_U, Hnoise, 2500,adiabatic_steps=2000)
    mingap_U_code =asim.find_global_adiabatic_rate( H0_U, Hcode, 2500,adiabatic_steps=2000)
    mingap_sum_noise = asim.find_global_adiabatic_rate( H0_sum_minus, Hnoise, 2500,adiabatic_steps=2000)
    mingap_sum_code =asim.find_global_adiabatic_rate( H0_sum_minus, Hcode, 2500,adiabatic_steps=2000)

    print("{0:20}{1:20}{2:20}{3:20}".format(mingap_U_code, mingap_U_noise , mingap_sum_code, mingap_sum_noise))