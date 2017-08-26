#cython: boundscheck=False
#cython: cdivision=True
#cython: wraparound=False
#
# This is the Cython translation of the tseries test (https://github.com/cran/tseries/)
# R source code. If you make amendments (especially to the loop declarations!) try to
# comment with the original code inline (where possible) so later debugging can be performed
# much more simply.
#
# Author: Taylor G Smith <taylor.smith@alkaline-ml.com>

from libc.string cimport memset
from libc.math cimport pow
import numpy as np
cimport numpy as np

ctypedef float [:, :] float_array_2d_t
ctypedef double [:, :] double_array_2d_t
ctypedef np.npy_intp INTP
ctypedef np.npy_float FLOAT
ctypedef np.float64_t DOUBLE

cdef fused floating1d:
    float[::1]
    double[::1]

cdef fused floating_array_2d_t:
    float_array_2d_t
    double_array_2d_t


np.import_array()


def C_tseries_pp_sum(np.ndarray[DOUBLE, ndim=1, mode='c'] u, INTP n, INTP L, DOUBLE s):
    """Translation of the ``tseries_pp_sum`` C source code located at:
    https://github.com/cran/tseries/blob/8ceb31fa77d0b632dd511fc70ae2096fa4af3537/src/ppsum.c

    This code provides efficient computation of the sums involved in the Phillips-Perron tests.
    """
    cdef INTP i, j
    cdef DOUBLE tmp1, tmp2

    tmp1 = 0.0
    for i in range(1, L + 1):  # for (i=1; i<=(*l); i++)
        tmp2 = 0.0

        for j in range(i, n):  # for (j=i; j<(*n); j++)
            tmp2 += u[j] * u[j - i]  # u[j]*u[j-i]

        # tmp2 *= 1.0-((double)i/((double)(*l)+1.0))
        tmp2 *= 1.0 - (float(i) / (float(L) + 1.0))
        tmp1 += tmp2

    tmp1 /= float(n)
    tmp1 *= 2.0

    return s + tmp1


cdef DOUBLE approx1(DOUBLE v, np.ndarray[DOUBLE, ndim=1, mode='c'] x,
            np.ndarray[DOUBLE, ndim=1, mode='c'] y,
            INTP n, DOUBLE ylow, DOUBLE yhigh, INTP kind,
            DOUBLE f1, DOUBLE f2):

    # Approximate  y(v),  given (x,y)[i], i = 0,..,n-1
    cdef INTP i, j, ij
    if n == 0:
        return np.nan

    i = 0
    j = n - 1

    # out-of-domain points
    if v < x[i]:
        return ylow
    if v > x[j]:
        return yhigh

    # find the correct interval by bisection
    while i < j - 1:  # x[i] <= v <= x[j]
        ij = (i + j) / 2  # i+1 <= ij <= j-1
        if v < x[ij]:
            j = ij
        else:
            i = ij
        # still i < j

    # probably i == j-1

    # interpolate
    if v == x[j]:
        return y[j]
    if v == x[i]:
        return y[i]

    # impossible: if x[j] == x[i] return y[i]
    if kind == 1:  # linear
        return y[i] + (y[j] - y[i]) * ((v - x[i]) / (x[j] - x[i]))
    else:  # 2 == constant
        # is this necessary? if f1 or f2 is zero, won't the multiplication cause 0.0 anyways?
        return (y[i] * f1 if f1 != 0.0 else 0.0) + (y[j] * f2 if f2 != 0.0 else 0.0)


def C_Approx(np.ndarray[DOUBLE, ndim=1, mode='c'] x,
             np.ndarray[DOUBLE, ndim=1, mode='c'] y,
             np.ndarray[DOUBLE, ndim=1, mode='c'] xout,
             INTP method, INTP f, DOUBLE yleft, DOUBLE yright):

    cdef INTP i, nxy = x.shape[0], nout = xout.shape[0]
    cdef INTP f2 = f, f1 = 1 - f
    cdef DOUBLE v

    # make yout
    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] yout = np.zeros(nout, dtype=np.float64)

    for i in range(nout):
        v = xout[i]

        # yout[i] = ISNAN(xout[i]) ? xout[i] : approx1(xout[i], x, y, nxy, &M);
        if not np.isnan(v):
            v = approx1(v, x, y, nxy, yleft, yright, method, f1, f2)

        # assign to the interpolation vector
        yout[i] = v

    # return
    return yout


def C_pop_A(np.ndarray[np.int64_t, ndim=2, mode='c'] A,
            np.ndarray[np.int64_t, ndim=1, mode='c'] frecob):

    cdef i, j, n = frecob.shape[0]

    j = 0
    for i in range(n):
        if frecob[i] == 1:
            A[i, j] = 1
            j += 1
