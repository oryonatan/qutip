
# coding: utf-8

# In[1]:

# %matplotlib inline

# from IPython.display import Image,display

# from numpy import pi

# from qutip import *
# import numpy as np
# import matplotlib.pyplot as plt
# from importlib import reload 
# from scipy.optimize import fsolve

# class LocalOperator:
#     def __init__(self, list_of_ops):
#         # list of 1 qubit ops and indexes of the shape (op, i) 
#         # for example (sigmax(), 1), (sigmaz(),5), (hadamard(),9)
#         self.list_of_ops = sorted(list_of_ops, key=lambda pair: pair[1])
#         self.d = self.list_of_ops[-1][1]
        
#     def full_form(self, n = None):
#         if n == None : n = self.d
#         ops_copy = list(self.list_of_ops)
#         cur_op = ops_copy.pop()
#         full_list = []
#         for i in range(n):
#             if cur_op[1] == i:
#                 full_list.append(cur_op[0])
#                 try:
#                     cur_op = ops_copy.pop()
#                 except IndexError:
#                     cur_op = (0,-1)
#             else:
#                 full_list.append(qeye(2))
#         return tensor(full_list)
    
#     def tensor(self, other_op):
#         maxd = max(self.d, other_op.d)
#         new_ops = []
#         for i in range(maxd):
#             this_i = 
        


# In[27]:

get_ipython().magic('matplotlib inline')

from IPython.display import Image,display

from numpy import pi

from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload 
from scipy.optimize import fsolve

class LocalOperator:
    def __init__(self, dict_of_ops):
        # list of 1 qubit ops and indexes of the shape (op, i) 
        # for example {1:sigmaz(), 5: sigmax(), 10:hadamard()}
        self.dict_of_ops = dict_of_ops
        self.update_d()
        
    def update_d(self):
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
        new = LocalOperator(self.dict_of_ops)
        for index,op in other.dict_of_ops.items():
            if index in new.dict_of_ops.keys():
                new.dict_of_ops[index] *=  other.dict_of_ops[index]
            else:
                new.dict_of_ops[index] = other.dict_of_ops[index]
        new.update_d()
        return new


# In[42]:

a = {4:sigmax(),2:sigmax()}
b = {3:sigmax()}
# a = {0:sigmax()}
# b = {0:sigmax()}


# In[43]:

l1 = LocalOperator(a)
l2 = LocalOperator(b)


# In[44]:

c= l1.tensor(l2)


# In[45]:

c.full_form()


# In[ ]:

a[0]


# In[ ]:

pop(a)


# In[ ]:

a.pop()


# In[ ]:

a


# In[ ]:

b=list(a)


# In[ ]:

b.pop()


# In[ ]:

set ((1,2,3,4,4,4))


# In[ ]:

next(a.items()


# In[ ]:

c = 5
c*=4


# In[ ]:

c


# In[ ]:



