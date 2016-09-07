
# coding: utf-8

# In[20]:

get_ipython().magic('matplotlib inline')

from IPython.display import Image,display

from numpy import pi

import qutip
from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload 
from scipy.optimize import fsolve
import pdb

class LocalOperator:
    #qubits are one based (1..d)
    def __init__(self, dict_of_ops):
        # list of 1 qubit ops and indexes of the shape (op, i) 
        # for example {1:sigmaz(), 5: sigmax(), 10:hadamard()}
        self.dict_of_ops = dict_of_ops
        self.update_d()
        
    def update_d(self):
        if len(self.dict_of_ops) == 0:
            self.d = 0
        else:
            self.d = max(self.dict_of_ops.keys())
    
    def force_d(self, d):
        """ Returns a copy of the local operator, with dimension of d
            notice that running update_d on the returned object will not work 
        """
        new = LocalOperator(self.dict_of_ops.copy())
        if d not in new.dict_of_ops.keys():
            new.dict_of_ops[d] = qeye(2)
            new.update_d()
        return new
        
    def full_form(self, n = None):
        if n == None : n = self.d
            
        full_list = []
        for i in range(1,n+1):
            if i in self.dict_of_ops.keys():
                full_list.append(self.dict_of_ops[i])
            else:
                full_list.append(qeye(2))
        return qutip.tensor(full_list)
    
    def tensor(self, other):
        #new is copy of old
        new = LocalOperator(self.dict_of_ops.copy())
        for index,op in other.dict_of_ops.items():
            if index in new.dict_of_ops.keys():
                new.dict_of_ops[index] *=  other.dict_of_ops[index]
            else:
                new.dict_of_ops[index] = other.dict_of_ops[index]
        new.update_d()
        return new
    
    def __mul__(self, other):
        return self.tensor(other)
    
    def __sub__(self, other):
        """ Adding two local operators could yield a non local one
            return Qobj of maximal dimension
        """
        n = max(self.d, other.d)
        return self.full_form(n) - other.full_form(n)
        #new = LocalOperator(self.dict_of_ops.copy())
        #if (sorted( self.dict_of_ops.keys() ) !=
        #    sorted( other.dict_of_ops.keys() ) ) :
        #     raise TypeError('Local operator operate on different qubits') 
        #        
        #for index,op in other.dict_of_ops.items():
        #    new.dict_of_ops[index] -=  other.dict_of_ops[index]
        #    
        #new.dict_of_ops = {x:y for (x,y) in new.dict_of_ops.items() if y.norm() != 0  }
        #new.update_d()
        # return new
    
    def __add__(self, other):
        """ Adding two local operators could yield a non local one
            return Qobj of maximal dimension
        """
        n = max(self.d, other.d)
        return self.full_form(n) + other.full_form(n)
        #new = LocalOperator(self.dict_of_ops.copy())
        #if (sorted( self.dict_of_ops.keys() ) !=
        #    sorted( other.dict_of_ops.keys() ) ) :
        #     raise TypeError('Local operator operate on different qubits') 
                
        #for index,op in other.dict_of_ops.items():
        #    new.dict_of_ops[index] +=  other.dict_of_ops[index]
            
        #new.dict_of_ops = {x:y for (x,y) in new.dict_of_ops.items() if y.norm() != 0  }
        #new.update_d()
        #return new
    
    def norm(self):
        if self.d == 0 : return 0
        ret = 1
        for op in self.dict_of_ops.values():
            ret *= op.norm()
        return ret
    
    def __repr__(self):
        return str(self.dict_of_ops)


# In[17]:

# H1 = LocalOperator({0:sigmaz(), 1:sigmaz()})
# H2 = LocalOperator({1:sigmax(), 2:sigmax()})
# H3 = LocalOperator({2:sigmaz(), 3:sigmaz()})
# print ("Sanity check :\n(H1 and H2) and (H2 and H3) not commuting ",(H1*H2-H2*H1).norm(), (H2*H3-H3*H2).norm())
# print ("H1-H3 are commuting ", (H1*H3-H3*H1).norm())


# In[18]:

# (H1*H3 - H3*H1).norm()


# In[19]:

# a = {1:0,2:4}


# In[ ]:

# list(filter(None,a.values()))


# In[ ]:

# l1.dict_of_ops


# In[ ]:

# Qobj(((0,1),(1,0)))\


# In[3]:

# basis(2) - basis(4)


# In[ ]:



