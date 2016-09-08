
# coding: utf-8

# In[2]:

get_ipython().magic('matplotlib inline')

from IPython.display import Image,display

from numpy import pi

from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload 
from scipy.optimize import fsolve
import pdb

class LocalOperator:
    def __init__(self, dict_of_ops):
        # list of 1 qubit ops and indexes of the shape (op, i) 
        # for example {1:sigmaz(), 5: sigmax(), 10:hadamard()}
        self.dict_of_ops = dict_of_ops
        self.update_d()
        
    def update_d(self):
        if len(self.dict_of_ops) == 0:
            self.d = 0
        else:
            self.d = max(self.dict_of_ops.keys()) + 1 
        
    def full_form(self, n = None):
        if n == None : n = self.d
            
        full_list = []
        for i in range(n):
            
            if i in self.dict_of_ops.keys():
                full_list.append(self.dict_of_ops[i])
            else:
                full_list.append(qeye(2))
        return tensor(full_list)
    
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
        new = LocalOperator(self.dict_of_ops.copy())
        if (sorted( self.dict_of_ops.keys() ) !=
            sorted( other.dict_of_ops.keys() ) ) :
             raise TypeError('Local operator operate on different qubits') 
                
        for index,op in other.dict_of_ops.items():
            new.dict_of_ops[index] -=  other.dict_of_ops[index]
            
        new.dict_of_ops = {x:y for (x,y) in new.dict_of_ops.items() if y.norm() != 0  }
        new.update_d()
        return new
    
    def __add(self, other):
        new = LocalOperator(self.dict_of_ops.copy())
        if (sorted( self.dict_of_ops.keys() ) !=
            sorted( other.dict_of_ops.keys() ) ) :
             raise TypeError('Local operator operate on different qubits') 
                
        for index,op in other.dict_of_ops.items():
            new.dict_of_ops[index] +=  other.dict_of_ops[index]
            
        new.dict_of_ops = {x:y for (x,y) in new.dict_of_ops.items() if y.norm() != 0  }
        new.update_d()
        return new
    
    def norm(self):
        if self.d == 0 : return 0
        ret = 1
        for op in self.dict_of_ops.values():
            ret *= op.norm()
        return ret
    
    def __repr__(self):
        return str(self.dict_of_ops)


# In[ ]:



