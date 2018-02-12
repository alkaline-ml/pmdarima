# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Array utilities

from __future__ import absolute_import, division

from sklearn.utils.validation import check_array
from sklearn.externals import six

import numpy as np

__all__ = [
    'c',
    'diff',
    'is_iterable'
]


def c(*args):
    """Since this whole library is aimed at re-creating in
    Python what R has already done so well, why not add a ``c`` function
    that wraps ``numpy.concatenate``? Similar to R, this works with scalars,
    iterables, and any mix therein.

    Note that using the ``c`` function on multi-nested lists or iterables
    will fail!

    Examples
    --------
    Using ``c`` with *args will yield a single array:
    >>> c(1, 2, 3, 4)
    array([1, 2, 3, 4])

    Using ``c`` with nested lists and scalars will also yield a single array:
    >>> c([1, 2], 4, c(5, 4))
    array([1, 2, 4, 5, 4])

    However, using ``c`` with multi-level lists will fail!
    >>> c([1, 2, 3], [[1, 2]])
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "pyramid/utils/array.py", line 64, in c
        return np.concatenate([a if is_iterable(a) else [a] for a in args])
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

    >>> from pyramid.utils import diff
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

    x = check_array(x, ensure_2d=False, dtype=np.float32)  # type: np.ndarray
    fun = _diff_vector if len(x.shape) == 1 else _diff_matrix
    res = x

    # "recurse" over range of differences
    for i in range(differences):
        res = fun(res, lag)
        # if it ever comes back empty, just return it as is
        if not res.shape[0]:
            return res

    return res


def is_iterable(x):
    """Determine whether an object ``x`` is iterable. In Python 2, this
    was as simple as checking for the ``__iter__`` attribute. However, in
    Python 3, strings became iterable. Therefore, this function checks for the
    ``__iter__`` attribute, returning True if present (except for strings,
    for which it will return False).

    Parameters
    ----------
    x : str, iterable or object
        The object in question.

    Returns
    -------
    isiter : bool
        True if iterable, else False.
    """
    if isinstance(x, six.string_types):
        return False
    return hasattr(x, '__iter__')
