# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Array utilities

from __future__ import absolute_import, division
from sklearn.utils.validation import check_array
import numpy as np

__all__ = [
    'c',
    'diff'
]


def c(*args):
    """Since this whole library is aimed at re-creating in
    Python what R has already done so well, why not add a ``c`` function
    that wraps ``numpy.concatenate``? Similar to R, this works with scalars,
    iterables, and any mix therein.

    Examples
    --------
    >>> from pyramid.utils import c
    >>> c(1, 2, 3, 4)
    array([1, 2, 3, 4])

    >>> from pyramid.utils import c
    >>> c([1, 2], 4, c(5, 4))
    array([1, 2, 4, 5, 4])

    References
    ----------
    .. [1] https://stat.ethz.ch/R-manual/R-devel/library/base/html/c.html
    """
    # R returns NULL for this
    if not args:
        return None

    # just an array of len 1
    if len(args) == 1:
        element = args[0]

        # if it's iterable, make it an array
        if hasattr(element, '__iter__'):
            return np.asarray(element)

        # otherwise it's not iterable, put it in an array
        return np.asarray([element])

    # concat all
    return np.concatenate([a if hasattr(a, '__iter__') else [a] for a in args])


def _diff_vector(x, lag):
    # compute the lag for a vector (not a matrix)
    n = x.shape[0]
    lag = min(n, lag)  # if lag > n, then we just want an empty array back
    return x[lag:n] - x[:n-lag]


def _diff_matrix(x, lag):
    # compute the lag for a matrix (not a vector)
    m, _ = x.shape
    lag = min(m, lag)  # if lag > n, then we just want an empty array back
    return x[lag:m, :] - x[:m-lag, :]


def diff(x, lag=1, differences=1):
    """A python implementation of the R ``diff`` function (documentation found
    at https://stat.ethz.ch/R-manual/R-devel/library/base/html/diff.html). This
    computes lag differences from an array given a ``lag`` and ``differencing``
    term.

    If ``x`` is a vector of length n, ``lag`` = 1 and ``differences`` = 1, then
    the computed result is equal to the successive differences
    ``x[lag:n] - x[:n-lag]``.

    Examples
    --------
    >>> from pyramid.utils import c, diff
    >>>
    >>> # lag 1, diff 1
    >>> x = c(10, 4, 2, 9, 34)
    >>> diff(x, 1, 1)
    array([ -6.,  -2.,   7.,  25.], dtype=float32)

    >>> from pyramid.utils import c, diff
    >>>
    >>> # lag 1, diff 2
    >>> x = c(10, 4, 2, 9, 34)
    >>> diff(x, 1, 2)
    array([  4.,   9.,  18.], dtype=float32)

    >>> from pyramid.utils import c, diff
    >>>
    >>> # lag 3, diff 1
    >>> x = c(10, 4, 2, 9, 34)
    >>> diff(x, 3, 1)
    array([ -1.,  30.], dtype=float32)

    >>> from pyramid.utils import c, diff
    >>>
    >>> # lag 6 (larger than the vec), diff 1
    >>> x = c(10, 4, 2, 9, 34)
    >>> diff(x, 6, 1)
    array([], dtype=float32)

    >>> from pyramid.utils import c, diff
    >>> import numpy as np
    >>>
    >>> x = np.arange(1, 10).reshape((3, 3)).T
    >>> diff(x, 1, 1)
    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.]], dtype=float32)

    Parameters
    ----------
    x : array-like, shape=(n_samples, [n_features])
        The array to difference.

    lag : int, optional (default=1)
        An integer > 0 indicating which lag to use.

    differences : int, optional (default=1)
        An integer > 0 indicating the order of the difference.

    Returns
    -------
    res : np.ndarray, shape=(n_samples, [n_features])
        The result of the differenced arrays.

    References
    ----------
    .. [1] https://stat.ethz.ch/R-manual/R-devel/library/base/html/diff.html
    """
    if any(v < 1 for v in (lag, differences)):
        raise ValueError('lag and differences must be positive (> 0) integers')

    x = check_array(x, ensure_2d=False, dtype=np.float32)
    fun = _diff_vector if len(x.shape) == 1 else _diff_matrix
    res = x

    # "recurse" over range of differences
    for i in range(differences):
        res = fun(res, lag)
        # if it ever comes back empty, just return it as is
        if not res.shape[0]:
            return res

    return res
