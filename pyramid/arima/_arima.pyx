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

import numpy as np
cimport numpy as np

cdef extern from "_arima_fast_helpers.h":
    bint pyr_isfinite(double) nogil

ctypedef float [:, :] float_array_2d_t
ctypedef double [:, :] double_array_2d_t
ctypedef int [:, :] int_array_2d_t
ctypedef long [:, :] long_array_2d_t

ctypedef np.npy_intp INTP
ctypedef np.npy_float FLOAT
ctypedef np.float64_t DOUBLE

cdef fused floating1d:
    float[::1]
    double[::1]

cdef fused floating_array_2d_t:
    float_array_2d_t
    double_array_2d_t

cdef fused intp1d:
    int[::1]
    long[::1]

cdef fused intp_array_2d_t:
    int_array_2d_t
    long_array_2d_t


np.import_array()


# This is simply here to test the pyr_finite function against nan values.
# This shouldn't be used internally, since it has the overhead of being
# exposed as a Python object.
def C_is_not_finite(v):
    return not pyr_isfinite(v)


def C_tseries_pp_sum(floating1d u, INTP n, INTP L, DOUBLE s):
    """Translation of the ``tseries_pp_sum`` C source code located at:
    https://github.com/cran/tseries/blob/8ceb31fa77d0b632dd511fc70ae2096fa4af3537/src/ppsum.c

    This code provides efficient computation of the sums involved in the Phillips-Perron tests.
    """
    cdef INTP i, j
    cdef DOUBLE tmp1, tmp2, result

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
    result = s + tmp1

    return result


cdef DOUBLE approx1(DOUBLE v, floating1d x, floating1d y, INTP n, DOUBLE ylow,
                    DOUBLE yhigh, INTP kind, DOUBLE f1, DOUBLE f2):

    # Approximate  y(v),  given (x,y)[i], i = 0,..,n-1
    cdef INTP i, j, ij
    if n == 0:
        return np.nan

    with nogil:
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


def C_Approx(floating1d x, floating1d y, floating1d xout,
             INTP method, INTP f, DOUBLE yleft, DOUBLE yright):

    cdef INTP i, nxy, nout, f1, f2
    cdef DOUBLE v

    nxy = x.shape[0]
    nout = xout.shape[0]
    f1 = 1 - f
    f2 = f

    # make yout
    # cdef double[::1] yout = np.zeros(nout)
    cdef np.ndarray[double, ndim=1, mode='c'] yout = np.zeros(nout,
                                                              dtype=np.float64,
                                                              order='c')

    for i in range(nout):
        v = xout[i]

        # yout[i] = ISNAN(xout[i]) ? xout[i] : approx1(xout[i], x, y, nxy, &M);
        # XXX: Does this work?
        if pyr_isfinite(v):
            v = approx1(v, x, y, nxy, yleft, yright, method, f1, f2)

        # assign to the interpolation vector
        yout[i] = v

    # return
    return yout


def C_pop_A(intp_array_2d_t A, intp1d frecob):

    cdef int i, j, n
    n = frecob.shape[0]
    j = 0

    with nogil:
        for i in range(n):
            if frecob[i] == 1:
                A[i, j] = 1
                j += 1


def C_compute_frecob_fast(intp1d frec, INTP s, floating_array_2d_t Omfhat):
    cdef int i, j, a, half_s
    cdef INTP n, v

    i = 0
    j = 0
    a = 0
    half_s = <int>(s / 2) - 1
    n = frec.shape[0]

    cdef np.ndarray[int, ndim=1, mode='c'] sq = np.arange(0, s - 1, 2, dtype=np.int32)
    cdef np.ndarray[int, ndim=1, mode='c'] frecob = np.zeros(s - 1, dtype=np.int32)
    cdef np.ndarray[int, ndim=2, mode='c'] A = None

    with nogil:
        for i in range(n):
            v = frec[i]

            if v == 1 and i == half_s:
                frecob[sq[i]] = 1
            if v == 1 and i < half_s:
                frecob[sq[i]] = frecob[sq[i] + 1] = 1

        # sum of == 1
        for i in range(s - 1):
            if frecob[i] == 1:
                a += 1

    A = np.zeros((s - 1, a), dtype=np.int32, order='c')
    return A, frecob
