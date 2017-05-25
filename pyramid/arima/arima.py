# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# ARIMA

from __future__ import print_function, absolute_import, division

import warnings
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import column_or_1d, check_array
from sklearn.linear_model import LinearRegression
from scipy.linalg import svd
import sys

# local relative imports
from ..utils.array import diff, c
from .utils import get_callable, frequency

# A relative import means the .so file needs to be right HERE
# (as in, for nosetests, inside this development dist), which it
# will not be unless using 'setup.py develop' with setuptools, which
# we don't want. Use an absolute import for now to avoid having to
# even involve setuptools... if we can ever get around this issue, we'll
# go back to the preferred relative import method.
# from ._arima import C_TSconv, C_getQ0, C_ARIMA_transPars, C_ARIMA_CSS
from pyramid.arima._arima import (C_TSconv, C_getQ0, C_ARIMA_transPars,
                                  C_ARIMA_CSS, C_ARIMA_Like)

__all__ = [
    'ARIMA'
]

# todo
dummy = lambda t: t


def _check_lag(r, max_lag=350):
    """This is really just to stay true to the C code that backs
    ARIMA. I'm not sure whether this could actually break anything, but
    in the R code's C code, it's suspected it could... so I'll make the same
    assumption.
    """
    if r > max_lag:
        raise ValueError('maximum supported lag -- max(p, q + 1) -- is %i, '
                         'but got %i' % (max_lag, r))


def _get_r(p, q):
    """Get the max between ``p`` and ``q+1`` and return the result.
    The documentation of the R code is pretty poor and does not explain
    exactly what ``r`` represents, but it seems to be the number of lags.

    Parameters
    ----------
    p : int
        ``p`` is the order (number of time lags) of the auto-regressive
        model, and is a non-negative integer.

    q : int
        ``q`` is the order of the moving-average model, and is a non-negative
        integer.

    Returns
    -------
    int : ``max(p, q + 1)``
    """
    return max(p, q + 1)


def _C_TSconv(a, b):
    # call the C_TSconv cython function
    res = np.zeros(a.shape[0] + b.shape[1] - 1, dtype=a.dtype)
    C_TSconv(a, b, res)
    return res


def _C_ss_routine(fun, phi, theta, r):
    """Run the C ss_init routine given the function (default will be
    C_getQ0 unless it's decided to implement the other one...) and
    return the reshaped result.

    Parameters
    ----------
    fun : callable
        The C routine to call (C_getQ0)

    phi : array-like
        The phi vector.

    theta : array-like
        The theta vector

    r : int
        The number of lags as a product of ``p``, ``q``. See
        ``_get_r``.
    """
    _check_lag(r)  # ensure valid
    res = np.zeros(r * r, dtype=phi.dtype)
    fun(phi, theta, res)
    return res.reshape((r, r), order='c')


# https://github.com/SurajGupta/r-source/blob/master/src/library/stats/R/arima.R#L452
def _make_arima(phi, theta, delta, kappa, ss_fun):
    # check for NAs
    phi, theta = phi.copy(), theta.copy()
    if any(np.isnan(array).any() for array in (phi, theta)):
        # phi/theta is not even created by user... do we want to warn?
        warnings.warn('NAs found in phi or theta')

    p = phi.shape[0]
    q = theta.shape[0]
    r = _get_r(p, q)
    d = delta.shape[0]
    rd = r + d

    # create the matrix and vector that will be stored in the mod
    Z = c(1.0, np.zeros(r - 1), delta)
    T = np.zeros((rd, rd))

    # if p > 0, make the first column:p in the T matrix = phi
    if p > 0:
        T[:p, 0] = phi  # if(p > 0) T[1L:p, 1L] <- phi

    if r > 1:
        ind = np.arange(1, r + 1)  # ind <- 2:r (R is 1-based indexing)

        # T[cbind(ind-1L, ind)] <- 1 (this is some wicked savage R-hackery)
        # it basically makes an adjusted diagonal of ones... we can do this
        # in a slightly less elegant fashion...
        for x in ind.tolist():
            T[x - 1, x] = 1.0

    if d > 0:
        # T[r+1L, ] <- Z
        T[r, :] = Z  # should it be r + 1 or is this more R indexing trickery?
        if d > 1:
            ind = np.arange(r + 1, d + 1)  # ind <- r + 2:d
            # T[cbind(ind, ind-1)] <- 1 (more of this???)
            for x in ind.tolist():
                T[x, x - 1] = 1.0  # (note this one is reversed from the above, similar swap)

    if q < r - 1:
        theta = c(theta, np.zeros(r - 1 - q))

    R = c(1, theta, np.zeros(d))
    V = np.outer(R, R)

    # no, idrk why the R code did the following...
    h = 0.
    a = np.zeros(rd)
    Pn = P = np.zeros((rd, rd))

    # might have to call another C routine here...
    if r > 1:
        Pn[:r, :r] = _C_ss_routine(ss_fun, phi, theta, r)
    else:
        Pn[0, 0] = 1. / (1. - phi * phi) if p > 0 else 1.

    # more R cbind hackery
    if d > 0:
        for i in range(r, d):  # Pn[cbind(r+1L:d, r+1L:d)] <- kappa
            Pn[i, i] = kappa

    return dict(
        phi=phi, theta=theta,
        delta=delta, Z=Z, a=a,
        P=P, T=T, V=V, h=h, Pn=Pn
    )


def _copy_mod(mod):
    """Note that R is pass-by-value, and not pass-by-reference.
    so passing an arg into a function implicitly copies it and
    does not impact its state outside of the function. Therefore,
    when passing things like phi, mod, etc., we need to explicitly
    copy them to hold true to what R is doing in those functions...
    """
    return dict(
        phi=mod['phi'].copy(), theta=mod['theta'].copy(),
        delta=mod['delta'].copy(), Z=mod['Z'].copy(), a=mod['a'].copy(),
        P=mod['P'].copy(), T=mod['T'].copy(), V=mod['V'].copy(),
        h=mod['h'],  # no copy necessary here since it's scalar
        Pn=mod['Pn'].copy()
    )


def _transPars(par, trans, p, q, arma):
    """This is an internal method that runs the C transPars
    subroutine and returns a tuple: ``(par, phi, theta)``

    Parameters
    ----------
    par : array-like, shape=(fixed.shape[0],)
        The par vector

    trans : int
        Whether to transform. 0 for False, 1 for true.
    """
    phi = np.zeros(p, dtype=DTYPE)
    theta = np.zeros(q, dtype=DTYPE)
    C_ARIMA_transPars(par.copy(), arma.copy(), int(trans), phi, theta)
    return par, phi, theta


def _C_ARIMA_Like(x, mod, resid):
    """Call C_ARIMA_Like with a copy of the ``mod`` dict"""
    mod = _copy_mod(mod)
    return C_ARIMA_Like(x.copy(), mod['phi'], mod['theta'], mod['delta'],
                        mod['a'], mod['P'], mod['Pn'], 0, int(resid))


def _upARIMA(mod, phi, theta, ss_init_func):
    # note that R is pass-by-value, and not pass-by-reference.
    # so passing an arg into a function implicitly copies it and
    # does not impact its state outside of the function. Therefore,
    # when passing things like phi, mod, etc., we need to explicitly
    # copy them to hold true to what R is doing here...
    phi, theta = phi.copy(), theta.copy()
    mod = _copy_mod(mod)

    _p = phi.shape[0]
    _q = theta.shape[0]

    # reassign the phi, theta in mod to the ones passed in
    mod['phi'] = phi
    mod['theta'] = theta
    r = _get_r(_p, _q)

    # the matrices we'll alter
    T, Pn = mod['T'], mod['Pn']

    if _p > 0:
        # set first col vector to phi in T
        T[:_p, 0] = phi

    if r > 1:
        # mod$Pn[1L:r, 1L:r] (R seq inclusive & starts @1)
        Pn[:r, :r] = _C_ss_routine(ss_init_func, phi, theta, r)
    else:
       Pn[0, 0] = 1. / (1. - phi * phi) if _p > 0 else 1.

    # in R, this just zeros out the whole array: mod$a[] <- 0
    mod['a'] *= 0  # (this just zeros out the np array)
    return mod


def _arimaSS(y, mod):
    ssq, sumlog, nu, rsResid = _C_ARIMA_Like(y, mod, True)

    # return these explicitly (rather than just the function call) so that
    # inspection knows we're returning 4 and warns us if we unpack improperly.
    return ssq, sumlog, nu, rsResid


def _armafn(x, xreg, ncxreg, p, q, trans, mod, coef, mask, narma, ss_init_func):
    par = coef.copy()  # make sure to copy...
    par[mask] = p

    # call the C subroutine
    par, phi, theta = _transPars(par=par, trans=trans)

    # not super clear what the R code is trying to do here:
    # if(is.null(Z <- tryCatch(upARIMA(mod, trarma[[1L]], trarma[[2L]]),
    #         error = function(e) NULL)))
    #     return(.Machine$double.xmax)# bad parameters giving error, e.g. in solve(.)

    try:
        new_mod = _upARIMA(mod=mod, phi=phi, theta=theta,
                           ss_init_func=ss_init_func)
    except:
        # bad parameters giving error, e.g. in solve(.)
        return sys.float_info.max

    x_prime = x.copy()
    if ncxreg > 0:
        # if ncxreg > 0, x_css will now be a matrix..
        x_prime = (x_prime - xreg).dot(par[np.arange(ncxreg) + narma]).ravel()

    ssq, sumlog, nu = _C_ARIMA_Like(x_prime, mod, False)
    s2 = ssq / nu
    return 0.5 * (np.log(s2) + sumlog / nu)


def _armaCSS(p, fixed, mask, x, xreg, ncxreg, narma, arma, n_cond):
    par = fixed.copy()  # copy fixed
    par[mask] = p  # assign p to the nan values in the fixed vector

    # call first C subroutine
    par, phi, theta = _transPars(par=par, trans=0)

    # alter x if needed
    x_css = x.copy()
    if ncxreg > 0:
        # if ncxreg > 0, x_css will now be a matrix..
        x_css = (x_css - xreg).dot(par[np.arange(ncxreg) + narma]).ravel()

    # call the next sub-routine...
    res = C_ARIMA_CSS(x_css, arma, phi, theta, n_cond, 0)
    return 0.5 * np.log(res)


def _arCheck(ar):
    rng = c(1, -ar)
    p = max(np.where(rng != 0)[0]) - 1
    if not p:
        return True
    return all( > 1)


class ARIMA(BaseEstimator):
    """An ARIMA, or autoregressive integrated moving average, is a generalization of an autoregressive
    moving average (ARMA) and is fitted to time-series data in an effort to forecast future points.
    ARIMA models can be especially efficacious in cases where data shows evidence of non-stationarity.

    The "AR" part of ARIMA indicates that the evolving variable of interest is regressed on its own
    lagged (i.e., prior observed) values. The "MA" part indicates that the regression error is actually a linear
    combination of error terms whose values occurred contemporaneously and at various times in the past.
    The "I" (for "integrated") indicates that the data values have been replaced with the difference
    between their values and the previous values (and this differencing process may have been performed
    more than once). The purpose of each of these features is to make the model fit the data as well as possible.

    Non-seasonal ARIMA models are generally denoted ``ARIMA(p,d,q)`` where parameters ``p``, ``d``, and ``q`` are
    non-negative integers, ``p`` is the order (number of time lags) of the autoregressive model, ``d`` is the degree
    of differencing (the number of times the data have had past values subtracted), and ``q`` is the order of the
    moving-average model. Seasonal ARIMA models are usually denoted ``ARIMA(p,d,q)(P,D,Q)m``, where ``m`` refers
    to the number of periods in each season, and the uppercase ``P``, ``D``, ``Q`` refer to the autoregressive,
    differencing, and moving average terms for the seasonal part of the ARIMA model.

    When two out of the three terms are zeros, the model may be referred to based on the non-zero parameter,
    dropping "AR", "I" or "MA" from the acronym describing the model. For example, ``ARIMA(1,0,0)`` is ``AR(1)``,
    ``ARIMA(0,1,0)`` is ``I(1)``, and ``ARIMA(0,0,1)`` is ``MA(1)``. [1]

    Parameters
    ----------
    p : int, optional (default=0)
        ``p`` is the order (number of time lags) of the auto-regressive model,
        and is a non-negative integer.

    d : int, optional (default=0)
        ``d`` is the degree of differencing (the number of times the data have
        had past values subtracted), and is a non-negative integer.

    q : int, optional (default=0)
        ``q`` is the order of the moving-average model, and is a non-negative integer.

    P : int, optional (default=0)

    D : int, optional (default=0)

    Q : int, optional (default=0)

    m : int, optional (default=None)

    xreg : array-like, shape=[n_samples, n_features], optional (default=None)
        The predictor matrix. Should be ``n_samples`` in length, and the
        desired number of columns. Depending on the values of ``allow_mean`` and
        ``allow_drift``, drift and intercept columns may be added.

    allow_mean

    allow_drift

    bias_adjust

    method

    include_constant

    optim_method

    n_cond

    fixed

    transform_pars

    kappa


    References
    ----------
    [1] https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average
    [2] https://github.com/robjhyndman/forecast/blob/19b0711e554524bf6435b7524517715658c07699/R/arima.R
    """
    def __init__(self, p=0, d=0, q=0, P=0, D=0, Q=0, m=None, xreg=None, allow_mean=True,
                 allow_drift=False, bias_adjust=False, method='css-ml', include_constant=None,
                 optim_method='l_bfgs', n_cond=None, fixed=None, transform_pars=True, kappa=1e6):
        super(ARIMA, self).__init__()

        self.p = p
        self.d = d
        self.q = q

        self.P = P
        self.D = D
        self.Q = Q

        self.m = m
        self.method = method
        self.include_constant = include_constant

        self.xreg = xreg
        self.allow_mean = allow_mean
        self.allow_drift = allow_drift
        self.bias_adjust = bias_adjust

        # extra args for the arima
        self.optim_method = optim_method
        self.n_cond = n_cond
        self.fixed = fixed
        self.transform_pars = transform_pars
        self.kappa = kappa

    def fit(self, x):
        """Fit the ARIMA on the ``x`` vector.

        Parameters
        ----------
        x : ``np.array`` or array-like, shape=(n_samples,)
            The vector to fit.
        """
        # ensure that x is a 1d ts, and is a float
        x = check_array(x, ensure_2d=False, force_all_finite=True, dtype=DTYPE)
        n = x.shape[0]

        # get some args locally and get the callables for methods
        p, d, q = self.p, self.d, self.q
        P, D, Q = self.P, self.D, self.Q
        ss_init = C_getQ0  # get_callable(self.ss_init, VALID_SS_INIT)
        optim = get_callable(self.optim_method, VALID_OPTIM)
        narma = p + d + P + Q  # sum of p, q, P, Q
        nd = d + D

        # xreg
        xreg = self.xreg
        if xreg is not None:
            # xreg should be a matrix
            xreg = check_array(xreg, force_all_finite=True, dtype=DTYPE, ensure_2d=True)
            if xreg.shape[0] != n:
                raise ValueError('dim mismatch in xreg length and x')

        # determine periodicity
        period = self.m
        if period is None:
            period = frequency(x)

        # check on inclusions of drift and mean (intercept) vectors.
        # if include_constant is None, we use the user-provided values for
        # allow_mean/allow_drift, however if include_constant is specifically
        # set to True/False, the mean/drift values might change.
        ic = self.include_constant
        include_mean, include_drift = self.allow_mean, self.allow_drift
        if ic is not None:

            # if the user has manually set that we will include the constant,
            # automatically include the mean (intercept) term and then only
            # include the drift if the differencing terms sum to 1
            if ic:
                include_mean = True

                # original R code: if((order[2] + seasonal$order[2]) == 1)
                if nd == 1:
                    include_drift = True

            # if explicitly set to false, make mean/drift vars false too
            else:
                include_mean = include_drift = False

        # check on drift status + d, D...
        # original R code: if((order[2] + seasonal$order[2]) > 1 & include.drift)
        if (nd > 1) and include_drift:
            # note that this condition can only be true if the user has left out include_constant
            # and explicitly set include_drift to be true, since we check otherwise above...
            include_drift = False
            warnings.warn("No drift term fitted as the order of difference is 2 or more, "
                          "i.e., d + D > 1 (d=%i, D=%i)" % (d, D), UserWarning)

        # now we come back to xreg... this is the number of columns
        ncxreg = 0 if xreg is None else xreg.shape[1]

        # add the drift term into the xreg matrix if we need to...
        drift_col = None
        if include_drift:
            drift = np.arange(n).reshape(n, 1)

            # append as first column in predictor matrix
            if xreg is not None:
                xreg = np.hstack([drift, xreg])
            else:
                xreg = drift
            drift_col = 0
            ncxreg += 1

        # initialize delta to 1.0 and then update it d times
        # https://github.com/SurajGupta/r-source/blob/master/src/library/stats/R/arima.R#L141
        delta = np.ones(1, dtype=DTYPE)
        for i in range(d):
            delta = _C_TSconv(delta, np.array([1.0, -1.0]))
        for i in range(D):
            delta = _C_TSconv(delta, c(1, np.zeros(period - 1), -1))

        # remove first index, make negative (Delta <- - Delta[-1L])
        delta = -delta[1:]

        # https://github.com/SurajGupta/r-source/blob/master/src/library/stats/R/arima.R#L145
        na_mask = np.isnan(x)  # since we are currently forcing all finite, not an issue, but might be later
        na_sum = na_mask.sum()
        n_used = (n - na_sum) - delta.shape[0]

        # if the the degree of differencing is 0, create an intercept in front of xreg
        mean_col = None
        if include_mean and nd == 0:
            intercept = np.ones(n, dtype=DTYPE).reshape(n, 1)
            if xreg is not None:
                xreg = np.hstack([intercept, xreg])
            else:
                xreg = intercept
            ncxreg += 1

            # adjust the column idcs where necessary
            mean_col = 0
            if drift_col is not None:
                drift_col += 1  # because we shifted it over

        # !!note that the xreg could still be None at this point!! but we'll get there...
        # for now, move on with computing the fixed vector and other fun stuff...

        # now figure out what method we'll use
        method_key = self.method
        if method_key not in VALID_METHODS:
            raise ValueError('method must be one of %r' % VALID_METHODS)

        # different methods will require a diff n_cond, and we need it before
        # actually calling those methods.
        n_cond = 0
        if method_key in ('css', 'css-ml'):
            n_cond = d + D * period
            n_cond1 = p + period * P

            if self.n_cond:
                n_cond += max(self.n_cond, n_cond1)
            else:
                n_cond += n_cond1

        # set up the fixed vector
        fixed = self.fixed
        if fixed is None:
            fixed = np.ones(narma + ncxreg) * np.nan
        else:
            fixed = check_array(column_or_1d(fixed), dtype=DTYPE, ensure_2d=False, force_all_finite=False)
            if fixed.shape[0] != narma + ncxreg:
                raise ValueError('wrong dimensions for "fixed" vector')

        # get the mask for whether there are any missing in the fixed array.
        # since fixed is default None, and in the default case is made into an array
        # of all NaNs, basically we're determining whether we need to optimize or not.
        # We also don't check for NaNs (or rather, we don't prevent them) in the case of
        # a user-provided `fixed`. The only case where we do not optimize is where there are
        # NO missing values in the fixed vector.
        fixed_mask = np.isnan(fixed)  # is.na(fixed)
        no_optim = ~(fixed_mask.any())  # no.optim <- !any(mask)

        # if we are not going to optimize, we don't need to transform pars
        transform_pars = self.transform_pars
        if no_optim:
            transform_pars = False

        # this is largely used just for the closures
        arma = np.asarray([p, q, P, Q, period, d, D])

        # ind <- arma[1L] + arma[2L] + seq_len(arma[3L])
        if transform_pars:
            ind = p + q + np.arange(P)
            if (~fixed_mask[np.arange(p)]).any() or (~fixed_mask[ind]).any():
                warnings.warn('some AR parameters were fixed: setting transform_pars to False')
                transform_pars = False

        init0 = np.zeros(narma)
        parscale = np.ones(narma)

        # if we have an xreg matrix (otherwise it's None)
        if ncxreg:
            # https://github.com/SurajGupta/r-source/blob/master/src/library/stats/R/arima.R#L193
            orig_xreg = ncxreg == 1 or (~fixed_mask[narma + np.arange(ncxreg)]).any()

            if not orig_xreg:
                _, s, _ = svd(xreg)  # we know there are no NaNs, so safe
                xreg = xreg.dot(s)

            dx = x
            dxreg = xreg
            if d > 0:
                dx = diff(dx, lag=1, differences=d)
                dxreg = diff(dxreg, lag=1, differences=d)

            if period > 1 and D > 0:
                dx = diff(dx, lag=period, differences=D)
                dxreg = diff(dxreg, lag=period, differences=D)

            fit = None
            if len(dx) > dxreg.shape[1]:
                # do fit of the linear model (why does the R package subtract 1?...)
                fit = LinearRegression().fit(X=dxreg - 1, y=dx)

            # if our fit is degenerate, so perform it on NON diff'd data
            # https://github.com/SurajGupta/r-source/blob/master/src/library/stats/R/arima.R#L211
            if fit is None or fit.rank_ == 0:
                fit = LinearRegression().fit(X=xreg - 1, y=x)

            # see which have null elements (shouldn't have any if we're enforcing non-null)
            isna = np.isnan(x) | np.apply_along_axis(lambda row: np.isnan(row).any(), arr=xreg, axis=1)
            n_used = (~isna).sum() - delta.shape[0]
            init0 = c(init0, fit.coef_)

            # todo: ses, parscale?

        # todo: what if init is not null? Right now we're not even bothering
        # todo: to take init as a model parameter, so we'll just ignore it...

        # https://github.com/SurajGupta/r-source/blob/master/src/library/stats/R/arima.R#L240
        coef = fixed.astype(DTYPE)

        # todo: parscale in optim.control?

        # https://github.com/SurajGupta/r-source/blob/master/src/library/stats/R/arima.R#L244
        if method_key == 'css':
            if no_optim:
                pass



        return self


# The DTYPE we'll use for everything here. Since there are
# lots of spots where we define the DTYPE in a numpy array,
# it's easier to define as a global for this module.
DTYPE = np.float32

# Any valid solver method. These are the three implemented
# in the R auto-arima package, so they're the three that will
# be included here.
VALID_METHODS = {'css', 'css-ml', 'ml'}

# The valid optimization strategies. These are implemented on the
# backend using scipy's optimization engine.
VALID_OPTIM = {
    'l_bfgs': dummy
}

# Deprecate for now, since I'm not sure we want to use the Rossigno version (requires F77...)
# VALID_SS_INIT = {
#     'Gardner1980': C_getQ0,
#     'Rossignol2011': C_getQ0bis
# }
