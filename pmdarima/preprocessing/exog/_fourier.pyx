#cython: boundscheck=False
#cython: cdivision=True
#cython: wraparound=False
#cython: nonecheck=False
#cython: language_level=3
#
# Author: Taylor G Smith <taylor.smith@alkaline-ml.com>

import numpy as np

cimport numpy as np
from libc.stdlib cimport malloc, free
cimport cython

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


cpdef C_fourier(INTP p, INTP times):
    cdef np.ndarray

    X_t = []
    for e in p:
        X_t.append(sinpi(2 * e * times))
        X_t.append(cospi(2 * e * times))

    return np.array(X_t).T


@cython.boundscheck(False)
@cython.wraparound(False)
def C_canova_hansen_sd_test(INTP ltrunc,
                            INTP Ne,
                            np.float64_t[:,:] Fhataux,
                            intp1d frec,
                            INTP s):
    """As of v0.9.0, this is used to compute the Omnw matrix iteratively.
    The python loop took extremely long, since it's a series of repeated
    matrix products.

    TODO: make this faster still?...
    """
    cdef int k, i, j, a, half_s
    cdef INTP n, v
    cdef unsigned int n_features, n_samples

    k = 0
    i = 0
    j = 0
    a = 0
    half_s = <int>(s / 2) - 1
    n = frec.shape[0]
    n_features = Fhataux.shape[1]

    # Define vector wnw, Omnw matrix
    cdef np.ndarray[double, ndim=2, mode='c'] Omnw, Omfhat
    cdef np.ndarray[int, ndim=2, mode='c'] A
    cdef np.float64_t[:, :] FhatauxT = Fhataux.T

    # Omnw is a square matrix of n x n
    Omnw = np.zeros((n_features, n_features), dtype=np.float64, order='c')

    # R code: wnw <- 1 - seq(1, ltrunc, 1)/(ltrunc + 1)
    cdef double* wnw
    cdef double wnw_denom = <double>(ltrunc + 1.)
    cdef double wnw_elmt

    cdef int* sq
    cdef int* frecob
    try:
        # Allocate memory
        wnw = <double*>malloc(ltrunc * sizeof(double))
        sq = <int*>malloc((s - 1) * sizeof(int))
        frecob = <int*>malloc((s - 1) * sizeof(int))

        # init wnw
        for i in range(0, ltrunc):
            wnw[i] = 1. - ((i + 1) / wnw_denom)

        # original R code:
        # for (k in 1:ltrunc)
        #     Omnw <- Omnw + (t(Fhataux)[, (k + 1):Ne] %*%
        #         Fhataux[1:(Ne - k), ]) * wnw[k]
        # This is a gigantic bottleneck, but I can't think of any better way
        # to solve it, and even R's auto ARIMA chokes on big CH tests. See:
        # https://stackoverflow.com/questions/53981660/efficiently-sum-complex-matrix-products-with-numpy
        Omnw = sum(np.matmul(FhatauxT[:, k + 1:],
                             Fhataux[:Ne - (k + 1), :]) * wnw[k]
                   for k in range(ltrunc))

        # Omfhat <- (crossprod(Fhataux) + Omnw + t(Omnw))/Ne
        Omfhat = (np.dot(Fhataux.T, Fhataux) + Omnw + Omnw.T) / float(Ne)

        with nogil:
            # Init sq and frecob
            for i in range(0, s - 1):
                sq[i] = 2 * i
                frecob[i] = 0

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

        # C_pop_A
        i = 0
        j = 0
        with nogil:
            for i in range(s - 1):
                if frecob[i] == 1:
                    A[i, j] = 1
                    j += 1

        # Now create the 'tmp' matrix pre-SVD
        return A, np.dot(np.dot(A.T, Omfhat), A)

    finally:
        free(wnw)
        free(sq)
        free(frecob)
