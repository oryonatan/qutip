
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')


from IPython.display import Image,display

from numpy import pi

from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload 
from scipy.optimize import fsolve
import LocalOperator as LO
reload(LO)
sx = sigmax()
sy = sigmay()
sz = sigmaz()
ID = qeye(2)

from Two_non_commutings import plot_commutations, plot_operator


# First lets make some non-commuting-terms LH to commuting-terms  ones !
# for starters I'll work on the 4 qubit chain.

# In[3]:

chain  = tensor([basis(2,0)] * 4)
# plot_qubism(chain, legend_iteration=2)
plt.show()


# now lets add some interaction!, we take the operators to be :
# $$H1 = \sigma_z^1\sigma_z^2 \\
# H2 = \sigma_x^2\sigma_x^3 \\
# H3 = \sigma_z^3\sigma_z^4 $$
# [H1,H2] and [H2,H3] are supposed zero.

# In[4]:

H1 = LO.LocalOperator({1:sz, 2:sz})
H2 = LO.LocalOperator({2:sx, 3:sx})
H3 = LO.LocalOperator({3:sz, 4:sz})
print ("Sanity check :\n(H1 and H2) and (H2 and H3) not commuting ",(H1*H2-H2*H1).norm(), (H2*H3-H3*H2).norm())
print ("H1-H3 are commuting ", (H1*H3-H3*H1).norm())
H1f = H1.full_form(3)
H2f = H2.full_form(3)
H3f = H3.full_form(3)
plot_commutations(H1f,H2f);
plot_commutations(H2f,H3f)
plot_commutations(H1f,H3f)


# Now since two operators operates on qubit number 2, and two operators operate on qubit number 3, we need to add additional 4 qubits to the system. they will be qubit 5 and 6 for the 2 qubit, and 7 and 8 for the third.
# 
# 
# <i><font color='teal'>Is it possible to algorithmically tell on which qubits an operator does not operate as identity ? it is probably possible because we know that each term operates only on constant number of qubits, however in it's full form (when expanded with identity to the rest of the space) it might not be possible in short time.</font></i>

# $$C_i = \sum_{j=1}^{k}C_j\otimes \sigma_x^j \otimes \sigma_x^i $$
# 

# $L1 = \sigma_z^1 \otimes C1^{2,5,6} \otimes I^{rest}$
# 
# $C1^{2,5,6} = \sigma_z^2\otimes \sigma_x^5 * \sigma_x^5 +\sigma_z^2\otimes \sigma_x^5 \otimes \sigma_x^6 $

# In[5]:

n = 8
C1 = ( LO.LocalOperator({2:sz}).force_d(n) + 
      LO.LocalOperator({2:sx, 5: sx ,6: sx }) )
L1 = LO.LocalOperator({1:sz}).full_form(n) + C1


# $L2 = C2^{2,5,6} \otimes C3^{3,7,8} \otimes I^{rest} $
# 
# $C2^{2,5,6} = \sigma_x^2\otimes \sigma_x^5 \otimes\sigma_x^6 +\sigma_x^2\otimes \sigma_x^6 *\sigma_x^6 $
# 
# $C3^{3,7,8} = \sigma_x^3\otimes \sigma_x^7 *\sigma_x^7 +\sigma_x^3\otimes \sigma_x^7 \otimes\sigma_x^8 $
# 

# In[6]:

C2 =(LO.LocalOperator( {2:sz,5:sx,6:sx} ).force_d(n) +
     LO.LocalOperator( {2:sx} ) )
C3 =(LO.LocalOperator( {3:sx} ).force_d(n) +
     LO.LocalOperator( {3:sz,7:sx,8:sx} ) )
L2 = C2+C3


# 
# $L3 =  C4^{3,7,8} \otimes \sigma_z^{4}\otimes I^{rest} $
# 
# $C4^{3,7,8} =  \sigma_z^3\otimes \sigma_x^7 \otimes \sigma_x^8 +\sigma_z^3\otimes\sigma_x^8*\sigma_x^8 $
# 

# In[7]:

C4 =(LO.LocalOperator( {3:sz} ).force_d(n) +
     LO.LocalOperator( {3:sx,7:sx,8:sx} ) )
L3 = LO.LocalOperator( {4:sz} ).full_form(n) + C4


# In[ ]:




# In[8]:

plot_commutations(L1,L2)
plot_commutations(L2,L3)
plot_commutations(L1,L3)


# In[27]:

plt.figure(figsize=(8,8))
plot_operator(L1+L2+L3)


# In[26]:

plt.figure(figsize=(8,8))
plot_operator(H1f+H2f+H3f)


# In[14]:

LH = L1+L2+L3


# In[20]:

plot_operator(Qobj(LH.data[0:8,0:8]))


# In[22]:

display(Qobj(LH.data[0:8,0:8]))
display(H1f+H2f+H3f)


# ## As we can see, this is not directly the top left ... what am I missing here?

# In[32]:

plt.figure(figsize=(8,8))

plot_operator(H1.full_form(n)+H2.full_form(n)+H3.full_form(n))


# In[4]:

H1 = LO.LocalOperator({1:sx,2:sx,3:ID})+LO.LocalOperator({1:sz,2:sz})
H2 = LO.LocalOperator({2:sx,3:sx})+LO.LocalOperator({2:sz,3:sz})


# In[6]:

plot_commutations(H1,H2)


# In[8]:

H2


# In[34]:

H1*H2 -H2 * H1


# In[59]:

O = -2j*(LO.LocalOperator({1:sx,2:sy,3:sz})-LO.LocalOperator({1:sz,2:sy,3:sx}))
plot_operator(O)
plot_commutations(H1,H2)


# In[39]:

sx*sz - sz*sx


# In[41]:

sy


# In[ ]:



