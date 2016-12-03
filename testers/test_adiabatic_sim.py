
# coding: utf-8

# In[2]:

get_ipython().magic('matplotlib inline')

from IPython.display import Image,display

from numpy import pi

from qutip import *
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.insert(0,"/home/yonatan/PycharmProjects/qutip")
import LH_tools as LHT


# In[3]:

import dorit.XXZZham as XXZZham


# In[4]:

import adiabatic_sim 
n = 6
N = 2**n



id_n = tensor([qeye(2)]*n)
psi0 = tensor([basis(2,0)]*n)
psi0= hadamard_transform(n)*psi0
H_0 = id_n-psi0*psi0.trans()
rot_H0, rot_psi0 = LHT.rotate_by_had(H_0, psi0)
w = tensor([basis(2,0),
            basis(2,1),
            basis(2,0),
            basis(2,0),
            basis(2,1),
            basis(2,0)])
H_1 = id_n - w*w.trans()

##Using roland

eps = 0.3
s = lambda t : LHT.s_function(t,N,eps)
tmax = LHT.find_s_one(N,eps)
tlist = np.linspace(0, tmax , 10)

# P_mat, evals_mat,psis_mine = adiabatic_sim.sim_simple_adiabatic(tlist, H_0, H_1 ,s)

# LHT.plot_PandEV(P_mat, evals_mat, tlist)
# pass


# In[5]:

import importlib


# In[6]:

importlib.reload(adiabatic_sim)
tlist = adiabatic_sim.sim_dynamic_evolution_binsearch(H_0, H_1,0.9)
P_mat, evals_mat,psis_mine = adiabatic_sim.sim_simple_adiabatic(tlist, H_0, H_1)
LHT.plot_PandEV(P_mat, evals_mat, np.linspace(0,max(tlist),len(tlist)))
print(len(tlist))
pass


# In[56]:

importlib.reload(adiabatic_sim)
tlist = adiabatic_sim.sim_dynamic_evolution_binsearch(H_0, H_1,0.1)
P_mat, evals_mat,psis_mine = adiabatic_sim.sim_simple_adiabatic(tlist, H_0, H_1)
LHT.plot_PandEV(P_mat, evals_mat, np.linspace(0,max(tlist),len(tlist)))
print(len(tlist))
pass


# In[57]:

tlist


# In[49]:

linlist = np.linspace(0,np.amax(tlist),len(tlist))
P_mat, evals_mat,psis_mine = adiabatic_sim.sim_simple_adiabatic(linlist, H_0, H_1)
LHT.plot_PandEV(P_mat, evals_mat, linlist)
pass


# In[1]:

linlist = np.linspace(0,np.amax(tlist),len(tlist))
s = lambda t : LHT.s_function(t,N,eps)
P_mat, evals_mat,psis_mine = adiabatic_sim.sim_simple_adiabatic(linlist, H_0, H_1,s)
LHT.plot_PandEV(P_mat, evals_mat, linlist)
pass


# In[ ]:




# In[72]:

lin_space =  np.linspace(1,max(tlist),len(tlist))

plt.plot(lin_space, tlist)
plt.figure()
plt.plot(lin_space,LHT.s_function(lin_space,N,0.3))


# This seems a bit like roland function , but the slope in the middle is too steep, so let's try shrinking the min step

# In[59]:

# when ran, I had min step = max_time /5000 intead of the  max_time / 1000 used above
importlib.reload(adiabatic_sim)
tlist = adiabatic_sim.sim_dynamic_evolution_binsearch(H_0, H_1,0.1)
P_mat, evals_mat,psis_mine = adiabatic_sim.sim_simple_adiabatic(tlist, H_0, H_1)
LHT.plot_PandEV(P_mat, evals_mat, np.linspace(0,max(tlist),len(tlist)))
print(len(tlist))
pass


# In[60]:

plt.plot(np.linspace(1,max(tlist),len(tlist)), tlist)


# Okay that didn't help, mabe just use the s_function defined by the search with tlist?

# In[82]:

linlist = np.linspace(0,np.amax(tlist),len(tlist))
def s_from_tlist(x,count = []):
    return linlist[tlist.index(x)]
    
P_mat, evals_mat,psis_mine = adiabatic_sim.sim_simple_adiabatic(tlist, H_0, H_1,s_from_tlist)
LHT.plot_PandEV(P_mat, evals_mat, tlist)
pass


# In[4]:

from importlib import reload
reload(adiabatic_sim)


# In[ ]:

psis, slist, pr_list, ev_list = adiabatic_sim.sim_dynamic_evolution_binsearch2( H_0, H_1, 0.3, 39,1)


# In[115]:

LHT.plot_PandEV(pr_list,ev_list,np.linspace(0,len(pr_list),len(pr_list)))
pass


# In[107]:

slist


# In[7]:

P_mat


# In[ ]:



