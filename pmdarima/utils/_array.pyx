#cython: boundscheck=False
#cython: cdivision=True
#cython: wraparound=False
#cython: nonecheck=False
#cython: language_level=3
#
# This is the Cython translation of the diffinv function R source code
#
# Author: Charles Drotar

import numpy as np
cimport numpy as np
cimport cython

ctypedef np.npy_intp INTP

cdef fused floating1d:
    float[::1]
    double[::1]

np.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:] C_intgrt_vec(floating1d x,
                             floating1d xi,
                             INTP lag):
    """Inverse diff

    References
    ----------
    .. [1] https://github.com/wch/r-source/blob/trunk/src/library/stats/R/diffinv.R#L39
    .. [2] https://github.com/mirror/r/blob/65a0e33a4b0a119703586fcd1f9742654738ae54/src/library/stats/src/PPsum.c#L46
    """
    cdef INTP i, n = x.shape[0]
    cdef np.ndarray[double, ndim=1, mode='c'] ans = \
        np.zeros(n + lag, dtype=np.float64, order='c')

    with nogil:
        for i in range(lag, lag + n):
            ans[i] = x[i - lag] + ans[i - lag]
    return ans
