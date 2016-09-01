
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')

from IPython.display import Image,display

from numpy import pi

from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import LH_tools
from importlib import reload 
from scipy.optimize import fsolve



# In[21]:

v = np.sqrt(0.06)
b2 = 0.5
b1 = -0.5


# In[22]:


H_0 =  Qobj([[0,v],[v,0]])
H_1 = Qobj([[b1,0],[0,b2]])
h_t = [[H_0,'1'],
      [H_1, 't']]


# In[23]:

tlist = np.linspace(-12, 12, 500)


# In[24]:

h_start = Qobj.evaluate(h_t, tlist[0], {})
psi0 = Qobj(h_start.eigenstates()[1][0])


# In[25]:

P_mat, EV_mat = LH_tools.simulate_adiabatic_process2(tlist, h_t, {}, psi0, False)


# In[26]:

LH_tools.plot_PandEV(P_mat, EV_mat, tlist)
plt.show()


# In[ ]:



