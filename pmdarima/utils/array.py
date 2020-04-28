# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Array utilities

from sklearn.utils.validation import check_array, column_or_1d

import numpy as np
import pandas as pd

from ..compat import DTYPE
from ._array import C_intgrt_vec

__all__ = [
    'as_series',
    'c',
    'check_endog',
    'check_exog',
    'diff',
    'diff_inv',
    'is_iterable'
]


def as_series(x):
    """Cast as pandas Series.

    Cast an iterable to a Pandas Series object. Note that the index
    will simply be a positional ``arange`` and cannot be set in this
    function.

    Parameters
    ----------
    x : array-like, shape=(n_samples,)
        The 1d array on which to compute the auto correlation.

    Examples
    --------
    >>> as_series([1, 2, 3])
    0    1
    1    2
    2    3
    dtype: int64

    >>> as_series(as_series((1, 2, 3)))
    0    1
    1    2
    2    3
    dtype: int64

    >>> import pandas as pd
    >>> as_series(pd.Series([4, 5, 6], index=['a', 'b', 'c']))
    a    4
    b    5
    c    6
    dtype: int64

    Returns
    -------
    s : pd.Series
        A pandas Series object.
    """
    if isinstance(x, pd.Series):
        return x
    return pd.Series(column_or_1d(x))


def c(*args):
    r"""Imitates the ``c`` function from R.

    Since this whole library is aimed at re-creating in
    Python what R has already done so well, the ``c`` function was created to
    wrap ``numpy.concatenate`` and mimic the R functionality. Similar to R,
    this works with scalars, iterables, and any mix therein.

    Note that using the ``c`` function on multi-nested lists or iterables
    will fail!

    Examples
    --------
    Using ``c`` with varargs will yield a single array:

    >>> c(1, 2, 3, 4)
    array([1, 2, 3, 4])

    Using ``c`` with nested lists and scalars will also yield a single array:

    >>> c([1, 2], 4, c(5, 4))
    array([1, 2, 4, 5, 4])

    However, using ``c`` with multi-level lists will fail!

    >>> c([1, 2, 3], [[1, 2]])  # doctest: +SKIP
    ValueError: all the input arrays must have same number of dimensions

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
        if is_iterable(element):
            return np.asarray(element)

        # otherwise it's not iterable, put it in an array
        return np.asarray([element])

    # np.concat all. This can be slow, as noted by numerous threads on
    # numpy concat efficiency, however an alternative using recursive
    # yields was tested and performed far worse:
    #
    # >>> def timeit(func, ntimes, *args):
    # ...     times = []
    # ...     for i in range(ntimes):
    # ...         start = time.time()
    # ...         func(*args)
    # ...         times.append(time.time() - start)
    # ...     arr = np.asarray(times)
    # ...     print("%s (%i times) - Mean: %.5f sec, "
    # ...           "Min: %.5f sec, Max: %.5f" % (func.__name__, ntimes,
    # ...                                         arr.mean(), arr.min(),
    # ...                                         arr.max()))
    # >>> y = [np.arange(10000), range(500), (1000,), 100, np.arange(50000)]
    # >>> timeit(c1, 100, *y)
    # c1 (100 times) - Mean: 0.00009 sec, Min: 0.00006 sec, Max: 0.00065
    # >>> timeit(c2, 100, *y)
    # c2 (100 times) - Mean: 0.08708 sec, Min: 0.08273 sec, Max: 0.10115
    #
    # So we stick with c1, which is this variant.
    return np.concatenate([a if is_iterable(a) else [a] for a in args])


def check_endog(y, dtype=DTYPE, copy=True, force_all_finite=False):
    """Wrapper for ``check_array`` and ``column_or_1d`` from sklearn

    Parameters
    ----------
    y : array-like, shape=(n_samples,)
        The 1d endogenous array.

    dtype : string, type or None (default=np.float64)
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.

    copy : bool, optional (default=False)
        Whether a forced copy will be triggered. If copy=False, a copy might
        still be triggered by a conversion.

    force_all_finite : bool, optional (default=False)
        Whether to raise an error on np.inf and np.nan in an array. The
        possibilities are:

        - True: Force all values of array to be finite.
        - False: accept both np.inf and np.nan in array.

    Returns
    -------
    y : np.ndarray, shape=(n_samples,)
        A 1d numpy ndarray
    """
    return column_or_1d(
        check_array(y, ensure_2d=False, force_all_finite=force_all_finite,
                    copy=copy, dtype=dtype))  # type: np.ndarray


def check_exog(X, dtype=DTYPE, copy=True, force_all_finite=True):
    """A wrapper for ``check_array`` for 2D arrays

    Parameters
    ----------
    X : array-like, shape=(n_samples, n_features)
        The exogenous array. If a Pandas frame, a Pandas frame will be returned
        as well. Otherwise, a numpy array will be returned.

    dtype : string, type or None (default=np.float64)
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.

    copy : bool, optional (default=True)
        Whether a forced copy will be triggered. If copy=False, a copy might
        still be triggered by a conversion.

    force_all_finite : bool, optional (default=True)
        Whether to raise an error on np.inf and np.nan in an array. The
        possibilities are:

        - True: Force all values of array to be finite.
        - False: accept both np.inf and np.nan in array.

    Returns
    -------
    X : pd.DataFrame or np.ndarray, shape=(n_samples, n_features)
        Either a 2-d numpy array or pd.DataFrame
    """
    if hasattr(X, 'ndim') and X.ndim != 2:
        raise ValueError("Must be a 2-d array or dataframe")

    if isinstance(X, pd.DataFrame):
        # if not copy, go straight to asserting finite
        if copy and dtype is not None:
            X = X.astype(dtype)  # tantamount to copy
        if force_all_finite and (~X.apply(np.isfinite)).any().any():
            raise ValueError("Found non-finite values in dataframe")
        return X

    # otherwise just a pass-through to the scikit-learn method
    return check_array(X, ensure_2d=True, dtype=DTYPE,
                       copy=copy, force_all_finite=force_all_finite)


def _diff_vector(x, lag):
    # compute the lag for a vector (not a matrix)
    n = x.shape[0]
    lag = min(n, lag)  # if lag > n, then we just want an empty array back
    return x[lag: n] - x[: n-lag]  # noqa: E226


def _diff_matrix(x, lag):
    # compute the lag for a matrix (not a vector)
    m, _ = x.shape
    lag = min(m, lag)  # if lag > n, then we just want an empty array back
    return x[lag: m, :] - x[: m-lag, :]  # noqa: E226


def diff(x, lag=1, differences=1):
    """Difference an array.

    A python implementation of the R ``diff`` function [1]. This computes lag
    differences from an array given a ``lag`` and ``differencing`` term.

    If ``x`` is a vector of length :math:`n`, ``lag=1`` and ``differences=1``,
    then the computed result is equal to the successive differences
    ``x[lag:n] - x[:n-lag]``.

    Examples
    --------
    Where ``lag=1`` and ``differences=1``:

    >>> x = c(10, 4, 2, 9, 34)
    >>> diff(x, 1, 1)
    array([ -6.,  -2.,   7.,  25.], dtype=float32)

    Where ``lag=1`` and ``differences=2``:

    >>> x = c(10, 4, 2, 9, 34)
    >>> diff(x, 1, 2)
    array([  4.,   9.,  18.], dtype=float32)

    Where ``lag=3`` and ``differences=1``:

    >>> x = c(10, 4, 2, 9, 34)
    >>> diff(x, 3, 1)
    array([ -1.,  30.], dtype=float32)

    Where ``lag=6`` (larger than the array is) and ``differences=1``:

    >>> x = c(10, 4, 2, 9, 34)
    >>> diff(x, 6, 1)
    array([], dtype=float32)

    For a 2d array with ``lag=1`` and ``differences=1``:

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

    x = check_array(x, ensure_2d=False, dtype=DTYPE, copy=False)
    fun = _diff_vector if x.ndim == 1 else _diff_matrix
    res = x

    # "recurse" over range of differences
    for i in range(differences):
        res = fun(res, lag)
        # if it ever comes back empty, just return it as is
        if not res.shape[0]:
            return res

    return res


def _diff_inv_vector(x, lag, differences, xi):
    # R code: if (missing(xi)) xi < - rep(0., lag * differences)
    # R code: if (length(xi) != lag * differences)
    # R code:   stop("'xi' does not have the right length")
    if xi is None:
        xi = np.zeros(lag * differences, dtype=DTYPE)
    else:
        xi = check_endog(xi, dtype=DTYPE, copy=False, force_all_finite=False)
        if xi.shape[0] != lag * differences:
            raise IndexError('"xi" does not have the right length')

    if differences == 1:
        return np.asarray(C_intgrt_vec(x=x, xi=xi, lag=lag))
    else:
        # R code: diffinv.vector(diffinv.vector(x, lag, differences - 1L,
        # R code:               diff(xi, lag=lag, differences=1L)),
        # R code:               lag, 1L, xi[1L:lag])
        return diff_inv(
            x=diff_inv(x=x, lag=lag, differences=differences - 1,
                       xi=diff(x=xi, lag=lag, differences=1)),
            lag=lag,
            differences=1,
            xi=xi[:lag]  # R: xi[1L:lag]
        )


def _diff_inv_matrix(x, lag, differences, xi):
    n, m = x.shape
    y = np.zeros((n + lag * differences, m), dtype=DTYPE)

    if m >= 1:  # todo: R checks this. do we need to?
        # R: if(missing(xi)) xi <- matrix(0.0, lag*differences, m)
        if xi is None:
            xi = np.zeros((lag * differences, m), dtype=DTYPE)
        else:
            xi = check_array(
                xi, dtype=DTYPE, copy=False, force_all_finite=False,
                ensure_2d=True)
            if xi.shape != (lag * differences, m):
                raise IndexError('"xi" does not have the right shape')

        # TODO: can we vectorize?
        for i in range(m):
            y[:, i] = _diff_inv_vector(x[:, i], lag, differences, xi[:, i])

    return y


def diff_inv(x, lag=1, differences=1, xi=None):
    """
    Inverse the difference of an array.

    A python implementation of the R ``diffinv`` function [1]. This computes
    the inverse of lag differences from an array given a ``lag``
    and ``differencing`` term.

    If ``x`` is a vector of length :math:`n`, ``lag=1`` and ``differences=1``,
    then the computed result is equal to the cumulative sum plus left-padding
    of zeros equal to ``lag * differences``.

    Examples
    --------
    Where ``lag=1`` and ``differences=1``:

    >>> x = c(10, 4, 2, 9, 34)
    >>> diff_inv(x, 1, 1)
    array([ 0., 10., 14., 16., 25., 59.])

    Where ``lag=1`` and ``differences=2``:

    >>> x = c(10, 4, 2, 9, 34)
    >>> diff_inv(x, 1, 2)
    array([  0.,   0.,  10.,  24.,  40.,  65., 124.])

    Where ``lag=3`` and ``differences=1``:

    >>> x = c(10, 4, 2, 9, 34)
    >>> diff_inv(x, 3, 1)
    array([ 0.,  0.,  0., 10.,  4.,  2., 19., 38.])

    Where ``lag=6`` (larger than the array is) and ``differences=1``:

    >>> x = c(10, 4, 2, 9, 34)
    >>> diff_inv(x, 6, 1)
    array([ 0.,  0.,  0.,  0.,  0.,  0., 10.,  4.,  2.,  9., 34.])

    For a 2d array with ``lag=1`` and ``differences=1``:

    >>> import numpy as np
    >>>
    >>> x = np.arange(1, 10).reshape((3, 3)).T
    >>> diff_inv(x, 1, 1)
    array([[ 0.,  0.,  0.],
           [ 1.,  4.,  7.],
           [ 3.,  9., 15.],
           [ 6., 15., 24.]])

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
        The result of the inverse of the difference arrays.

    References
    ----------
    .. [1] https://stat.ethz.ch/R-manual/R-devel/library/stats/html/diffinv.html
    """  # noqa: E501
    x = check_array(
        x, dtype=DTYPE, copy=False, force_all_finite=False, ensure_2d=False)

    # R code: if (lag < 1L || differences < 1L)
    # R code: stop ("bad value for 'lag' or 'differences'")
    if any(v < 1 for v in (lag, differences)):
        raise ValueError('lag and differences must be positive (> 0) integers')

    if x.ndim == 1:
        return _diff_inv_vector(x, lag, differences, xi)
    elif x.ndim == 2:
        return _diff_inv_matrix(x, lag, differences, xi)
    raise ValueError("only vector and matrix inverse differencing "
                     "are supported")


def is_iterable(x):
    """Test a variable for iterability.

    Determine whether an object ``x`` is iterable. In Python 2, this
    was as simple as checking for the ``__iter__`` attribute. However, in
    Python 3, strings became iterable. Therefore, this function checks for the
    ``__iter__`` attribute, returning True if present (except for strings,
    for which it will return False).

    Parameters
    ----------
    x : str, iterable or object
        The object in question.

    Examples
    --------
    Strings and other objects are not iterable:

    >>> x = "not me"
    >>> y = 123
    >>> any(is_iterable(v) for v in (x, y))
    False

    Tuples, lists and other structures with ``__iter__`` are:

    >>> x = ('a', 'tuple')
    >>> y = ['a', 'list']
    >>> all(is_iterable(v) for v in (x, y))
    True

    This even applies to numpy arrays:

    >>> import numpy as np
    >>> is_iterable(np.arange(10))
    True

    Returns
    -------
    isiter : bool
        True if iterable, else False.
    """
    if isinstance(x, str):
        return False
    return hasattr(x, '__iter__')
