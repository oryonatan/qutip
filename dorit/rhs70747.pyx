# This file is generated automatically by QuTiP.
# (C) 2011 and later, P. D. Nation & J. R. Johansson

import numpy as np
cimport numpy as np
cimport cython
from qutip.cy.spmatfuncs import spmv_csr, spmvpy

cdef double pi = 3.14159265358979323

include '/home/yonatan/anaconda3/envs/qutip/lib/python3.5/site-packages/qutip/cy/complex_math.pxi'

ctypedef np.complex128_t CTYPE_t
ctypedef np.float64_t DTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
def cy_td_ode_rhs(double t, np.ndarray[CTYPE_t, ndim=1] vec, np.ndarray[CTYPE_t, ndim=1] data0, np.ndarray[int, ndim=1] idx0, np.ndarray[int, ndim=1] ptr0, np.ndarray[CTYPE_t, ndim=1] data1, np.ndarray[int, ndim=1] idx1, np.ndarray[int, ndim=1] ptr1, np.float64_t t_max):
    
    cdef Py_ssize_t row
    cdef int num_rows = len(vec)
    cdef np.ndarray[CTYPE_t, ndim=1] out = np.zeros((num_rows),dtype=np.complex)
     
    spmvpy(data0, idx0, ptr0, vec, (t_max-t)/t_max, out)
    spmvpy(data1, idx1, ptr1, vec, t/t_max, out)
    return out
