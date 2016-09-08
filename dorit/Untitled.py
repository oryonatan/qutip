
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')
import pyximport

import sys
import os
sys.path.insert(0,os.path.join(os.getcwd(),os.pardir))
from qutip import *
import numpy as np
pyximport.install(setup_args={"include_dirs":np.get_include()})
import matplotlib.pyplot as plt
import LH_tools as LHT
import dorit.XXZZham as XXZZham
from dorit.XXZZham import add_high_energies, rotate_to_00_base
import random


# In[2]:

terms = []
for i in range(1, 5):
    for j in range(i + 1, 5):
        a = random.uniform(-10, 10)
        terms.append(XXZZham.XXZZ_term(i, j, a))

H = XXZZham.XXZZham(terms)
H_com = H.get_commuting_term_ham()
H_high_energies = add_high_energies(rotate_to_00_base(H_com))


# In[3]:

h_t= [[H_com,'(t_max-t)/t_max'],
      [H_high_energies, 't/t_max']]


# In[4]:

in_state = H_com.eigenstates(eigvals=1)[1][0]


# In[5]:

LHT.benchmark(h_t, in_state)


# In[ ]:



