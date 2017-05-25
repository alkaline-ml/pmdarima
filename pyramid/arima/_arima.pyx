#cython: boundscheck=False
#cython: cdivision=True
#cython: wraparound=False
#
# This is the Cython translation of the https://github.com/SurajGupta/r-source/blob/master/src/library/stats/src/arima.c
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


def C_TSconv(np.ndarray[np.float_t, ndim=1, mode='c'] a,
             np.ndarray[np.float_t, ndim=1, mode='c'] b,
             np.ndarray[np.float_t, ndim=1, mode='c'] ab):
    cdef INTP na = a.shape[0]
    cdef INTP nb = b.shape[0]

    # with nogil:
    for i in range(na):
        for j in range(nb):
            ab[i + j] += a[i] * b[j]

#############################################################
# Methods for creating the lags


cpdef INTP _inclu2(INTP n_p,
           np.ndarray[DOUBLE, ndim=1, mode='c'] xnext,
           np.ndarray[DOUBLE, ndim=1, mode='c'] xrow, DOUBLE ynext,
           np.ndarray[np.float_t, ndim=1, mode='c'] res,
           np.ndarray[DOUBLE, ndim=1, mode='c'] rbar,
           np.ndarray[DOUBLE, ndim=1, mode='c'] thetab):

    # https://github.com/SurajGupta/r-source/blob/master/src/library/stats/src/arima.c#L823
    cdef DOUBLE cbar, sbar, di, xi, xk, rbthis, dpi
    cdef INTP i, k, ithisr = 0

    # This subroutine updates d, rbar, thetab by the inclusion
    # of xnext and ynext.
    for i in range(n_p):
        xrow[i] = xnext[i]

    for i in range(n_p):
        if xrow[i] != 0.0:
            xi = xrow[i]
            di = res[i]
            dpi = di + xi * xi
            res[i] = dpi
            cbar = di / dpi
            sbar = xi / dpi

            for k in range(i + 1, k < n_p):
                xk = xrow[k]
                rbthis = rbar[ithisr]
                xrow[k] = xk - xi * rbthis

                # increment after set
                rbar[ithisr] = cbar * rbthis + sbar * xk
                ithisr += 1

            xk = ynext
            ynext = xk - xi * thetab[i]
            thetab[i] = cbar * thetab[i] + sbar * xk

            if di == 0.0:
                return 0

        else:
            ithisr = ithisr + n_p - i - 1
    return 0


def C_getQ0(np.ndarray[np.float_t, ndim=1, mode='c'] phi,
            np.ndarray[np.float_t, ndim=1, mode='c'] theta,
            np.ndarray[np.float_t, ndim=1, mode='c'] res):  # this is r x r in length
    """Used in conjunction with the Gardner 1980 method for computing
    and allocation the lag observations.
    """

    # see https://github.com/SurajGupta/r-source/blob/master/src/library/stats/src/arima.c#L1000
    cdef INTP p = phi.shape[0]
    cdef INTP q = theta.shape[0]

    # this is also defined in the calling method, but we need it again here?
    cdef INTP r = max(p, q + 1)

    # the C code defines this as type size_t... what is that? And can it overflow?
    cdef INTP n_p = r * (r + 1) / 2  # this looks like r choose 2 + the length of the diag?
    cdef INTP nrbar = n_p * (n_p - 1) / 2
    cdef INTP j, i
    cdef DOUBLE vj, vi
    V = []

    for j in range(r):
        vj = 0.0
        if j == 0:
            vj = 1.0
        elif j - 1 < q:
            vj = theta[j - 1]

        for i in range(j, r):
            vi = 0.0
            if i == 0:
                vi = 1.0
            elif i - 1 < q:
                vi = theta[i - 1]
            V.append(vi * vj)

    # https://github.com/SurajGupta/r-source/blob/master/src/library/stats/src/arima.c#L1036
    cdef DOUBLE val
    if r == 1:
        if p == 0:
            val = 1.0
        else:
            val = 1.0 / (1.0 - phi[0] * phi[0])
        res[0] = val
        return res

    # indices
    cdef INTP ind = 0, ind1 = -1, npr = n_p - r
    cdef INTP npr1 = npr + 1
    cdef INTP indi, indn, indj = npr, ind2 = npr - 1
    cdef INTP im, jm, ithisr

    # res is already initialized to zeros, so initialize some others to zeros
    cdef np.ndarray[DOUBLE, ndim=1] rbar = np.zeros(nrbar, dtype=np.float64)
    cdef np.ndarray[DOUBLE, ndim=1] thetab = np.zeros(n_p, dtype=np.float64)
    cdef np.ndarray[DOUBLE, ndim=1] xnext = np.zeros(n_p, dtype=np.float64)
    cdef np.ndarray[DOUBLE, ndim=1] xrow = np.zeros(n_p, dtype=np.float64)

    # phi/theta vals
    cdef DOUBLE phij, ynext, phii, bi

    # https://github.com/SurajGupta/r-source/blob/master/src/library/stats/src/arima.c#L1042
    if p > 0:
        # The set of equations s * vec(P0) = vec(v) is solved for
        # vec(P0).  s is generated row by row in the array xnext.  The
        # order of elements in P is changed, so as to bring more leading
        # zeros into the rows of s.

        for j in range(r):
            phij = phi[j] if j < p else 0.0

            # assign xnext and increment
            xnext[indj] = 0.0
            indj += 1

            indi = npr1 + j
            for i in range(j, r):
                # assign ynext and increment ind
                ynext = V[ind]
                ind += 1

                # assign phii
                phii = phi[i] if i < p else 0.0

                # re-value some
                if j != r - 1:
                    xnext[indj] = -phii
                    if i != r - 1:
                        xnext[indi] -= phij

                        # increment and THEN assign:
                        ind1 += 1
                        xnext[ind1] = -1.0

                xnext[npr] = -phii * phij

                # increment ind2 and THEN test:
                # https://github.com/SurajGupta/r-source/blob/master/src/library/stats/src/arima.c#L1075
                ind2 += 1
                if ind2 >= n_p:
                    ind2 = 0

                # assign idx for xnext, operate, then reset to 0.0
                xnext[ind2] += 1.0
                _inclu2(n_p, xnext, xrow, ynext, res, rbar, thetab)
                xnext[ind2] = 0.0

                # https://github.com/SurajGupta/r-source/blob/master/src/library/stats/src/arima.c#L1080
                if i != r - 1:
                    # assign one and THEN increment, then assign another
                    xnext[indi] = 0.0
                    indi += 1
                    xnext[ind1] = 0.0

        # now outside of the big J loop
        ithisr = nrbar - 1
        im = n_p - 1
        for i in range(n_p):
            bi = thetab[im]

            jm = n_p - 1
            for j in range(i):
                bi -= rbar[ithisr] * res[jm]

                # decrement
                ithisr -= 1
                jm -= 1

            # re-assign then decrement
            res[im] = bi
            im -= 1

        # now re-order: https://github.com/SurajGupta/r-source/blob/master/src/library/stats/src/arima.c#L1095
        ind = npr

        # for (i = 0; i < r; i++) xnext[i] = P[ind++];
        for i in range(r):
            xnext[i] = res[ind]
            ind += 1

        ind = n_p - 1
        ind1 = npr - 1

        # for (i = 0; i < npr; i++) P[ind--] = P[ind1--];
        for i in range(npr):
            res[ind] = res[ind1]
            ind -= 1
            ind1 -= 1

        # for (i = 0; i < r; i++) P[i] = xnext[i];
        for i in range(r):
            res[i] = xnext[i]

    else:
        # P0 is obtained by back substitution for a moving average process.
        indn = n_p
        ind = n_p

        for i in range(r):
            for j in range(i + 1):  # for (j = 0; j <= i; j++)
                ind -= 1
                res[ind] = V[ind]

                if j != 0:
                    res[ind] += res[indn]
                    indn -= 1

    # now unpack to a full matrix
    ind = np
    for i in range(r - 1, 0, -1):  # for (i = r - 1, ind = np; i > 0; i--)
        for j in range(r - 1, i - 1, -1):  # for (j = r - 1; j >= i; j--)
            res[r * i + j] = res[ind]  # P[r * i + j] = P[--ind];
            ind -= 1

    for i in range(r - 1):  # for (i = 0; i < r - 1; i++)
        for j in range(i + 1, r):  # for (j = i + 1; j < r; j++)
            res[i + r * j] = res[j + r * i]

    return res


cpdef INTP _partrans(INTP p, np.ndarray[np.float_t, ndim=1, mode='c'] raw,
              np.ndarray[np.float_t, ndim=1, mode='c'] new):

    cdef INTP j, k
    cdef DOUBLE a
    cdef np.ndarray[DOUBLE, ndim=1] work = np.zeros(new.shape[0], dtype=np.float64)

    # Step one: map (-Inf, Inf) to (-1, 1) via tanh
    # The parameters are now the pacf phi_{kk}
    for j in range(p):
        work[j] = new[j] = np.tanh(raw[j])

    # Step two: run the Durbin-Levinson recursions to find phi_{j.},
    # j = 2, ..., p and phi_{p.} are the auto-regression coefficients
    for j in range(1, p):
        a = new[j]
        for k in range(j):
            work[k] -= a * new[j - k - 1]

        # needs to be in separate loop so we don't overwrite the
        # next slot we need to look at in 'new' (above)
        for k in range(j):
            new[k] = work[k]
    return 0


cpdef INTP C_ARIMA_transPars(np.ndarray[np.float_t, ndim=1, mode='c'] sin,
                             np.ndarray[np.int_t, ndim=1, mode='c'] arma, INTP trans,
                             np.ndarray[np.float_t, ndim=1] phi,
                             np.ndarray[np.float_t, ndim=1] theta):

    cdef INTP mp = arma[0], mq = arma[1], msp = arma[2], msq = arma[3], ns = arma[4]
    cdef INTP p = mp + ns * msp
    cdef INTP q = mq + ns * msq
    cdef INTP i, j, v, n

    # define params as a copy of sin
    cdef np.ndarray[np.float_t, ndim=1, mode='c'] params = sin.copy()  # just copy it

    if trans == 1:
        n = mp + mq + msp + msq
        if (mp > 0):
            _partrans(mp, sin, params)
        v = mp + mq
        if (msp > 0):
            _partrans(msp, sin + v, params + v)

    # these originally were in the if AND the else... reduce copy/paste of the C code
    for i in range(mp):
        phi[i] = params[i]

    for i in range(mq):
        theta[i] = params[i + mp]

    if ns > 0:
        # expand out seasonal ARMA models
        for i in range(mp, p):
            phi[i] = 0.0

        for i in range(mq, q):
            theta[i] = 0.0

        for j in range(msp):
            phi[(j + 1) * ns - 1] += params[j + mp + mq]
            for i in range(mp):
                phi[(j + 1) * ns + i] -= params[i] * params[j + mp + mq]

        for j in range(msq):
            theta[(j + 1) * ns - 1] += params[j + mp + mq + msp]
            for i in range(mq):
                theta[(j + 1) * ns + i] += params[i + mp] * params[j + mp + mq + msp]

    # this impacts phi/theta in place so no returning them required
    return 0


def C_ARIMA_Like(np.ndarray[np.float_t, ndim=1, mode='c'] y,
                 np.ndarray[np.float_t, ndim=1, mode='c'] phi,
                 np.ndarray[np.float_t, ndim=1, mode='c'] theta,
                 np.ndarray[np.float_t, ndim=1, mode='c'] delta,
                 np.ndarray[np.float_t, ndim=1, mode='c'] a,
                 np.ndarray[np.float_t, ndim=1, mode='c'] P,
                 np.ndarray[np.float_t, ndim=1, mode='c'] Pn,
                 INTP sUP, INTP giveResid):

    cdef INTP n = y.shape[0], rd = a.shape[0], p = phi.shape[0]
    cdef INTP q = theta.shape[0], d = delta.shape[0]
    cdef INTP r = rd - d
    cdef DOUBLE sumlog = 0.0
    cdef DOUBLE ssq = 0.0
    cdef INTP nu = 0

    # vectors - init and allocate
    cdef np.ndarray[DOUBLE, ndim=1] rsResid, anew, mm, M
    anew = np.zeros(rd, dtype=np.float64)
    M = np.zeros(rd, dtype=np.float64)
    mm = None

    if d > 0:
        mm = np.zeros(rd * rd, dtype=np.float64)

    if giveResid == 1:
        rsResid = np.zeros(n)

    cdef INTP L, i, j
    cdef DOUBLE tmp, vi, resid, gain
    for L in range(n):  # for (int l = 0; l < n; l++)
        for i in range(r):  # for (int i = 0; i < r; i++)
            tmp = a[i + 1] if (i < r - 1) else 0.0
            if i < p:
                tmp += phi[i] * a[0]
            anew[i] = tmp

        if d > 0:
            for i in range(r + 1, rd):  # for (int i = r + 1; i < rd; i++) anew[i] = a[i - 1];
                anew[i] = a[i - 1]
            tmp = a[0]
            for i in range(d):  # for (int i = 0; i < d; i++) tmp += delta[i] * a[r + i];
                tmp += delta[i] * a[r + i]
            anew[r] = tmp

        if L > sUP:
            if d == 0:
                for i in range(r):  # for (int i = 0; i < r; i++)

                    vi = 0.0
                    if i == 0:
                        vi = 1.0
                    elif i - 1 < q:
                        vi = theta[i - 1]

                    for j in range(r):  # for (int j = 0; j < r; j++)
                        tmp = 0.0
                        if j == 0:
                            tmp = vi
                        elif (j - 1 < q):
                            tmp = vi * theta[j - 1]

                        if (i < p and j < p):
                            tmp += phi[i] * phi[j] * P[0]
                        if (i < r - 1 and j < r - 1):
                            tmp += P[i + 1 + r * (j + 1)]
                        if (i < p and j < r - 1):
                            tmp += phi[i] * P[j + 1]
                        if (j < p and i < r - 1):
                            tmp += phi[j] * P[i + 1]
                        Pn[i + r * j] = tmp

            else:
                # mm = TP
                for i in range(r):  # for (int i = 0; i < r; i++)
                    for j in range(rd):  # for (int j = 0; j < rd; j++)
                        tmp = 0.0
                        if i < p:
                            tmp += phi[i] * P[rd * j]
                        if i < r - 1:
                            tmp += P[i + 1 + rd * j]
                        mm[i + rd * j] = tmp

                for j in range(rd):  # for (int j = 0; j < rd; j++)
                    tmp = P[rd * j]
                    for k in range(d):  # for (int k = 0; k < d; k++)
                        tmp += delta[k] * P[r + k + rd * j]
                        mm[r + rd * j] = tmp

                for i in range(1, d):  # for (int i = 1; i < d; i++)
                    for j in range(rd):  # for (int j = 0; j < rd; j++)
                        mm[r + i + rd * j] = P[r + i - 1 + rd * j]

                # Pnew = mmT'
                for i in range(r):  # for (int i = 0; i < r; i++)
                    for j in range(rd):  # for (int j = 0; j < rd; j++)
                        tmp = 0.0
                        if i < p:
                            tmp += phi[i] * mm[j]
                        if i < r - 1:
                            tmp += mm[rd * (i + 1) + j]
                        Pn[j + rd * i] = tmp

                for j in range(rd):  # for (int j = 0; j < rd; j++)
                    tmp = mm[j]
                    for k in range(d):  # for (int k = 0; k < d; k++)
                        tmp += delta[k] * mm[rd * (r + k) + j]
                        Pn[rd * r + j] = tmp

                for i in range(1, d):  # for (int i = 1; i < d; i++)
                    for j in range(rd):  # for (int j = 0; j < rd; j++)
                        Pn[rd * (r + i) + j] = mm[rd * (r + i - 1) + j]

                # Pnew <- Pnew + (1 theta) %o% (1 theta)
                for i in range(q + 1):  # for(int i = 0; i <= q; i++)
                    vi = 1. if (i == 0) else theta[i - 1]
                    for j in range(q + 1):  # for(int j = 0; j <= q; j++)
                        Pn[i + rd * j] += vi * (1. if (j == 0) else theta[j - 1])

        if not np.isnan(y[L]):
            resid = y[L] - anew[0]
            for i in range(d):  # for (int i = 0; i < d; i++)
                resid -= delta[i] * anew[r + i]  # resid -= delta[i] * anew[r + i]

            for i in range(rd):  # for (int i = 0; i < rd; i++)
                tmp = Pn[i]
                for j in range(d):  # for (int j = 0; j < d; j++)
                    tmp += Pn[i + (r + j) * rd] * delta[j]
                M[i] = tmp

            gain = M[0]
            for j in range(d):  # for (int j = 0; j < d; j++) gain += delta[j] * M[r + j];
                gain += delta[j] * M[r + j]

            if gain < 1e4:
                nu += 1
                ssq += resid * resid / gain
                sumlog += np.log(gain)

            if giveResid == 1:
                rsResid[L] = resid / np.sqrt(gain)

            for i in range(rd):  # for (int i = 0; i < rd; i++)
                a[i] = anew[i] + M[i] * resid / gain
            for i in range(rd):  # for (int i = 0; i < rd; i++)
                for j in range(rd):  # for (int j = 0; j < rd; j++)
                    P[i + j * rd] = Pn[i + j * rd] - M[i] * M[j] / gain

        else:
            for i in range(rd):  # for (int i = 0; i < rd; i++) a[i] = anew[i];
                a[i] = anew[i]
            for i in range(rd * rd):  # for (int i = 0; i < rd * rd; i++) P[i] = Pnew[i];
                P[i] = Pn[i]

            if giveResid == 1:
                rsResid[L] = np.nan  # rsResid[l] = NA_REAL;

    # done with the L loop
    if giveResid == 1:
        return ssq, sumlog, nu, rsResid
    return ssq, sumlog, nu


# ARIMA differencing happens here
# arma is p, q, sp, sq, ns, d, sd
def C_ARIMA_CSS(np.ndarray[np.float_t, ndim=1, mode='c'] y,
                np.ndarray[np.int_t, ndim=1, mode='c'] arma,
                np.ndarray[np.float_t, ndim=1, mode='c'] phi,
                np.ndarray[np.float_t, ndim=1, mode='c'] theta,
                INTP ncond, INTP useResid):

    # init counters, sizes & ssq
    cdef DOUBLE tmp, ssq = 0.0
    cdef INTP L, i, n = y.shape[0], p = phi.shape[0], q = theta.shape[0]
    cdef INTP ns, nu = 0

    # init w vector
    cdef np.ndarray[DOUBLE, ndim=1] w = np.zeros(n, dtype=np.float64)

    # copy the y vector into w
    for L in range(n):
        w[L] = y[L]

    for i in range(arma[5]):  # C (not R, so zero idxd) code: i < arma[5]
        for L in range(n - 1, 0, -1):  # for (int l = n - 1; l > 0; l--)
            w[L] -= w[L - 1]

    ns = arma[4]
    for i in range(arma[6]):
        for L in range(n - 1, ns - 1, -1):  # for (int l = n - 1; l >= ns; l--)
            w[L] -= w[L - ns]

    # init the residual vector and set appropriate slots to 0
    cdef np.ndarray[DOUBLE, ndim=1] resid = np.ones(n, dtype=np.float64) * np.nan
    if useResid == 1:
        for L in range(ncond):
            resid[L] = 0.0

    for L in range(ncond, n):
        tmp = w[L]

        for j in range(p):  # for (int j = 0; j < p; j++) tmp -= phi[j] * w[l - j - 1]
            tmp -= phi[j] * w[L - j - 1]

        for j in range(min(L - ncond, q)):  # for (int j = 0; j < min(l - ncond, q); j++)
            tmp -= theta[j] * resid[L - j - 1]

        resid[L] = tmp
        if not np.isnan(tmp):
            nu += 1
            ssq += tmp * tmp

    if useResid == 1:
        return ssq / float(nu), resid

    else:
        return ssq / float(nu)
