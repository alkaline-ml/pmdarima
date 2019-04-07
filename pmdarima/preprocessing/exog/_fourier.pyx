#cython: boundscheck=False
#cython: cdivision=True
#cython: wraparound=False
#cython: nonecheck=False
#cython: language_level=3
#
# Author: Taylor G Smith <taylor.smith@alkaline-ml.com>

import numpy as np

from cython.view cimport array as cvarray
from libc.math cimport sin, cos, M_PI
cimport numpy as np
cimport cython

ctypedef float [:, :] float_array_2d_t
ctypedef double [:, :] double_array_2d_t

ctypedef np.npy_intp INTP

np.import_array()


cpdef double[:, :] C_fourier_terms(double[:] p, double[:] times):
    cdef INTP i, j, k = p.shape[0], n = times.shape[0], m
    cdef float v

    cdef double [:, :] X = cvarray(shape=(k * 2, n),
                                   itemsize=sizeof(double),
                                   format="d")  # d for double and also DUH

    with nogil:
        j = 0
        for i in range(0, k * 2, 2):

            # 2 * p[j] * times * PI
            v = p[j] * 2 * M_PI

            for m in range(n):
                X[i, m] = sin(v * times[m])
                X[i + 1, m] = cos(v * times[m])

            j += 1

    return X
