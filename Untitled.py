
# coding: utf-8

# In[1]:



import numpy as np
from qutip import *
import numpy as np
import numpy.linalg as npla
import numba
from scipy.linalg import expm
import LH_tools
import matplotlib.pyplot as plt
import pdb


# In[2]:

A = np.random.random((3,3))


# In[3]:

@numba.jit(nogil=True)
def two_min_eigs(matrix):
    eigenValues,eigenVectors = npla.eig(matrix)
    print(eigenVectors)
    idx = eigenValues.argsort()[:2]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    return eigenValues,eigenVectors

a ,b = two_min_eigs(A)


# In[4]:

A* b[:,1]


# In[91]:


import numpy as np
from qutip import *
import numpy as np
import numba
from scipy.linalg import expm
import LH_tools
import matplotlib.pyplot as plt
import pdb

@numba.jit(nopython=True,nogil=True)
def sim_simple_adiabatic(tlist, H0, H1, rho0, draw):
    duration = len(tlist)
    tmin = min(tlist)
    tmax = max(tlist)
    eigvals_mat = []
    P_mat = []
    # start at H0 ground state 
    eigvals, eigvecs = H0.eigenstates(eigvals = 2)
    psi = eigvecs[0].data
    eigvals_mat.append(eigvals)
#     pdb.set_trace()
    P_mat.append(
        [abs(eigvecs[0].trans().data * psi).data[0] ** 2,
         abs(eigvecs[1].trans().data * psi).data[0] ** 2,])
    oldt = tmin    
    for t in tlist[1:]:
        dt = oldt - t
        Ht = H0 * (tmax-t)/(tmax-tmin) + H1* (t-tmin)/(tmax-tmin)
        eigvals , eigvecs = Ht.eigenstates(eigvals=2)
        U = expm(-1j * Ht.data * dt)
        psi = U * psi
        eigvals_mat.append(eigvals)
        P_mat.append(
            [ abs(eigvecs[0].trans().data * psi).data[0] ** 2,
              abs(eigvecs[1].trans().data * psi).data[0] ** 2 ] 
        )
        oldt = t
#         pdb.set_trace();
    return (P_mat, eigvals_mat)

                    
        
        


# In[210]:


import numpy as np
from qutip import *
import numpy as np
import numba
from scipy.linalg import expm
import LH_tools
import matplotlib.pyplot as plt
import pdb


    
def sim_simple_adiabatic(tlist, H0, H1, draw):
    duration = len(tlist)
    tmin = min(tlist)
    tmax = max(tlist)
    eigvals_mat = []
    P_mat = []
    # start at H0 ground state 
    eigvals, eigvecs = (H0+tmin*H1).eigenstates()
    psi = eigvecs[0]
    eigvals_mat.append(eigvals)
    P_mat.append(
        [abs(psi[0][0][0]) ** 2,
         abs(psi[1][0][0]) ** 2])
#     P_mat.append(
#         [abs(eigvecs[0].overlap(psi)) ** 2,
#          abs(eigvecs[1].overlap(psi)) ** 2])
    oldt = tmin    
    for t in tlist[1:]:
        dt = 0.1
        Ht = H0 + H1 * t
        eigvals , eigvecs = Ht.eigenstates(eigvals=2)
        U = Qobj(expm(-1j * Ht.data.tocsc() * dt))
        psi = U * psi
        eigvals_mat.append(eigvals)
        P_mat.append(
                [abs(psi[0][0][0]) ** 2,
                 abs(psi[1][0][0]) ** 2])
#         print(eigvecs[0], " " , psi,"\n" ,abs(eigvecs[0].overlap(psi)) ** 2)
#         print(eigvecs[1], " " , psi ,"\n", abs(eigvecs[0].overlap(psi)) ** 2)
#         P_mat.append(
#             [ abs(eigvecs[0].overlap(psi)) ** 2,
#               abs(eigvecs[1].overlap(psi)) ** 2 ] 
#         )
        oldt = t
    return (P_mat, eigvals_mat)


# In[211]:

n = 1
N = 2**n
v = np.sqrt(0.06)
H_0 = Qobj( ( (0,v),(v,0) ) )
H_1 = Qobj( ( (-1/2,0),(0,1/2) ) )


import time
start_time = time.time()

tlist = np.linspace(-12, 12 , 240)

P_mat, evals_mat = sim_simple_adiabatic(tlist, H_0, H_1, False)


print("--- %s seconds ---" % (time.time() - start_time))
LH_tools.plot_PandEV(P_mat, evals_mat, tlist)
pass


# In[238]:

from importlib import reload
reload(LH_tools)

# P_mat, evals_mat = sim_simple_adiabatic(tlist, H_0, H_1, psi0, False)
# LH_tools.plot_PandEV(P_mat, evals_mat, tlist)
_,psi0 = (H_0+H_1*min(tlist)).eigenstates(eigvals=1)
print(psi0)
h_t = [[H_0,'1'],
       [H_1,'t']]
P_mat, evals_mat = LH_tools.simulate_adiabatic_process3(tlist,h_t,{},psi0[0], False)
LH_tools.plot_PandEV(P_mat, evals_mat, tlist)
pass


# In[237]:

isket(psi0)


# In[235]:

psi0[0]


# In[15]:

n = 6
N = 2**n
id_n = tensor([qeye(2)]*n)
psi0 = tensor([basis(2,0)]*n)
psi0= hadamard_transform(n)*psi0
H_0 = id_n-psi0*psi0.trans()
rot_H0, rot_psi0 = LH_tools.rotate_by_had(H_0, psi0)
w = tensor([basis(2,0),
            basis(2,1),
            basis(2,0),
            basis(2,0),
            basis(2,1),
#             basis(2,0),
#             basis(2,0),
#             basis(2,1),
#             basis(2,0),
            basis(2,0)])
H_1 = id_n - w*w.trans()

import time
start_time = time.time()

tlist = np.linspace(0, N , 100)

P_mat, evals_mat = sim_simple_adiabatic(tlist, H_0, H_1, psi0, False)


print("--- %s seconds ---" % (time.time() - start_time))
LH_tools.plot_PandEV(P_mat, evals_mat, tlist)
pass


# In[134]:

q = basis(2)


# In[156]:

q.norm()


# In[195]:

q


# In[213]:

abs(q[0])**2


# In[197]:

q.data


# In[201]:

q.diag()


# In[ ]:



