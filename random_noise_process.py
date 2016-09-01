
# coding: utf-8

# In this work I'll try using RC01 function as an inverse function for "powering" a third random hamiltonian in a grover search problem.
# 
# 

# In[106]:

get_ipython().magic('matplotlib inline')

from IPython.display import Image,display

from numpy import pi

from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import LH_tools as LHT

import adiabatic_sim as adsim
from importlib import reload ; reload(adsim)

n = 6
N = 2**n
eps = 0.3
tmax = LHT.find_s_one(N,eps)
tlist = np.linspace(0, tmax, 25)
id_n = tensor([qeye(2)]*n)
psi0 = tensor([basis(2,0)]*n)
psi0= hadamard_transform(n)*psi0
H_0 = id_n-psi0*psi0.trans()
LHT.plot_operator(H_0)


# In[78]:

#look for some state
if (n < 4 ) :
    in_state = tensor([basis(2,1)]*n)
else:
    in_state = tensor(tensor([basis(2,1)]*(n-2)),basis(2,0),basis(2,1))
H_1 = id_n - in_state*in_state.trans()
LHT.plot_operator(H_1)


# In[107]:

no_noise = Qobj(dims=H_0.dims)
s = lambda x : 1
P_mat, eigvals_mat, _ = adsim.sim_noise_evolutoion(tlist, H_0, H_1 , no_noise,  s)
LHT.plot_PandEV(P_mat,eigvals_mat,tlist)
pass


# In[157]:

H_rand = rand_herm(H_0.shape[0], dims=H_0.dims)


# In[159]:

no_noise = Qobj(dims=H_0.dims)

for noise_power in np.logspace(1,4):
    s_func = lambda t : LHT.s_function(t, N, eps)
    s = lambda t : 1/scipy.misc.derivative(s_func, t)*min(scipy.misc.derivative(s_func,tlist)) / noise_power

    plt.plot(s(tlist))
    f = plt.figure()
    f.suptitle("Noise power %s" % noise_power)
    P_mat, eigvals_mat, _ = adsim.sim_noise_evolutoion(tlist, H_0, H_1 , H_rand,  s)
    LHT.plot_PandEV(P_mat,eigvals_mat,tlist)
pass


# In[104]:




# In[120]:

plt.plot(
    1/np.diff(LHT.s_function(tlist, N, eps) / min(np.diff(LHT.s_function(tlist, N, eps)))
             ))


# In[161]:

s_func = lambda t : LHT.s_function(t, N, eps)
s = lambda t : 1/scipy.misc.derivative(s_func, t)*min(scipy.misc.derivative(s_func,tlist)) 
plt.plot(s(tlist))


# In[156]:

np.logspace(2,4)


# In[167]:

H_rand


# In[ ]:



