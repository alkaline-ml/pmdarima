# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# R approx function

import numpy as np

from ..utils.array import c, check_endog
from ..utils import get_callable
from ..compat.numpy import DTYPE

# since the C import relies on the C code having been built with Cython,
# and since the platform might name the .so file something funky (like
# _arima.cpython-35m-darwin.so), import this absolutely and not relatively.
from pmdarima.arima._arima import C_Approx

__all__ = [
    'approx'
]

# the ints get passed to C code
VALID_APPROX = {
    'constant': 2,
    'linear': 1
}

# get the valid tie funcs
VALID_TIES = {
    'ordered': None,  # never really used...
    'mean': np.average
}

# identity function defined once to avoid multiple lambda calls
# littered throughout
_identity = (lambda t: t)


def _regularize(x, y, ties):
    """Regularize the values, make them ordered and remove duplicates.
    If the ``ties`` parameter is explicitly set to 'ordered' then order
    is already assumed. Otherwise, the removal process will happen.

    Parameters
    ----------
    x : array-like, shape=(n_samples,)
        The x vector.

    y : array-like, shape=(n_samples,)
        The y vector.

    ties : str
        One of {'ordered', 'mean'}, handles the ties.
    """
    x, y = [
        check_endog(arr, dtype=DTYPE, preserve_series=False)
        for arr in (x, y)
    ]

    nx = x.shape[0]
    if nx != y.shape[0]:
        raise ValueError('array dim mismatch: %i != %i' % (nx, y.shape[0]))

    # manipulate x if needed. if ties is 'ordered' we assume that x is
    # already ordered and everything has been handled already...
    if ties != 'ordered':
        o = np.argsort(x)

        # keep ordered with one another
        x = x[o]
        y = y[o]

        # what if any are the same?
        ux = np.unique(x)
        if ux.shape[0] < nx:
            # Do we want to warn for this?
            # warnings.warn('collapsing to unique "x" values')

            # vectorize this function to apply to each "cell" in the array
            def tie_apply(f, u_val):
                vals = y[x == u_val]  # mask y where x == the unique value
                return f(vals)

            # replace the duplicates in the y array with the "tie" func
            func = VALID_TIES.get(ties, _identity)

            # maybe expensive to vectorize on the fly? Not sure; would need
            # to do some benchmarking. However, we need to in order to keep y
            # and x in scope...
            y = np.vectorize(tie_apply)(func, ux)

            # does ux need ordering? hmm..
            x = ux

    return x, y


def approx(x, y, xout, method='linear', rule=1, f=0, yleft=None,
           yright=None, ties='mean'):
    """Linearly interpolate points.

    Return a list of points which (linearly) interpolate given data points,
    or a function performing the linear (or constant) interpolation.

    Parameters
    ----------
    x : array-like, shape=(n_samples,)
        Numeric vector giving the coordinates of the points
        to be interpolated.

    y : array-like, shape=(n_samples,)
        Numeric vector giving the coordinates of the points
        to be interpolated.

    xout : int, float or iterable
        A scalar or iterable of numeric values specifying where
        interpolation is to take place.

    method : str, optional (default='linear')
        Specifies the interpolation method to be used.
        Choices are "linear" or "constant".

    rule : int, optional (default=1)
        An integer describing how interpolation is to take place
        outside the interval ``[min(x), max(x)]``. If ``rule`` is 1 then
        np.nans are returned for such points and if it is 2, the value at the
        closest data extreme is used.

    f : int, optional (default=0)
        For ``method`` = "constant" a number between 0 and 1 inclusive,
        indicating a compromise between left- and right-continuous step
        functions. If y0 and y1 are the values to the left and right of the
        point then the value is y0 if f == 0, y1 if f == 1, and y0*(1-f)+y1*f
        for intermediate values. In this way the result is right-continuous
        for f == 0 and left-continuous for f == 1, even for non-finite
        ``y`` values.

    yleft : float, optional (default=None)
        The value to be returned when input ``x`` values are less than
        ``min(x)``. The default is defined by the value of rule given below.

    yright : float, optional (default=None)
        The value to be returned when input ``x`` values are greater than
        ``max(x)``. The default is defined by the value of rule given below.

    ties : str, optional (default='mean')
        Handling of tied ``x`` values. Choices are "mean" or "ordered".
    """
    if method not in VALID_APPROX:
        raise ValueError('method must be one of %r' % VALID_APPROX)

    # make sure xout is an array
    xout = c(xout).astype(np.float64)  # ensure double

    # check method
    method_key = method

    # not a callable, actually, but serves the purpose..
    method = get_callable(method_key, VALID_APPROX)

    # copy/regularize vectors
    x, y = _regularize(x, y, ties)
    nx = x.shape[0]

    # if len 1? (we've already handled where the size is 0, since we check that
    # in the _regularize function when we call c1d)
    if nx == 1:
        if method_key == 'linear':
            raise ValueError('need at least two points to '
                             'linearly interpolate')

    # get yleft, yright
    if yleft is None:
        yleft = y[0] if rule != 1 else np.nan
    if yright is None:
        yright = y[-1] if rule != 1 else np.nan

    # call the C subroutine
    yout = C_Approx(x, y, xout, method, f, yleft, yright)  # MemoryView
    return xout, np.asarray(yout)
