import sys
sys.path.append('/home/oryonatan/pycharm-debug-py3k.egg')

import pydevd

from numpy import pi
from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import LH_tools
from importlib import reload
from scipy.optimize import fsolve

n = 6
N = 2**n
id_n = tensor([qeye(2)]*n)
psi0 = tensor([basis(2,0)]*n)
psi0= hadamard_transform(n)*psi0
H_0 = id_n-psi0*psi0.trans()
rot_H0, rot_psi0 = LH_tools.rotate_by_had(H_0, psi0)
in_state = tensor([basis(2,0),basis(2,1),
                   basis(2,0),basis(2,1),
#                    basis(2,0),basis(2,1),
#                    basis(2,0),basis(2,1),
                   basis(2,0),basis(2,1)])

H_1 = id_n - in_state*in_state.trans()


eps = 0.3
s = lambda t : LH_tools.s_function(t,N,eps)
tmax = LH_tools.find_s_one(N,eps)
tlist = np.linspace(0, tmax , 10)

reload(LH_tools)

#roland
h_t= [[H_0,'1-1/2*(1-(np.sqrt(N - 1) * np.tan((2 * t * epsilon * np.sqrt(N - 1) - N * np.arctan(np.sqrt(N - 1))) / N))/(1-N))'],
      [H_1, '1/2*(1-(np.sqrt(N - 1) * np.tan((2 * t * epsilon * np.sqrt(N - 1) - N * np.arctan(np.sqrt(N - 1))) / N))/(1-N))']]
args = {'t_max':tmax, 'N':N, 'epsilon': eps}
in_state = Qobj.evaluate(h_t,0,args).eigenstates(eigvals=1)[1][0]
args = {'t_max':tmax, 'N':N, 'epsilon': eps}
# import pydevd; pydevd.settrace('localhost', port=5000, stdoutToServer=True, stderrToServer=True)
P_mat,EV_mat,psis_qutip = LH_tools.simulate_adiabatic_process2(tlist, h_t, args ,in_state, False)
f = LH_tools.plot_PandEV(P_mat,EV_mat,tlist)
plt.show()