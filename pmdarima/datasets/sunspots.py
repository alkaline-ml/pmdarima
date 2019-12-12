# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# This is the sunspots dataset found in R.

import numpy as np
import pandas as pd

from os.path import join
import calendar

from ..compat import DTYPE
from . import _base as base

__all__ = [
    'load_sunspots'
]


def load_sunspots(as_series=False, dtype=DTYPE):
    """Monthly Sunspot Numbers, 1749 - 1983

    Monthly mean relative sunspot numbers from 1749 to 1983. Collected at Swiss
    Federal Observatory, Zurich until 1960, then Tokyo Astronomical
    Observatory.

    Parameters
    ----------
    as_series : bool, optional (default=False)
        Whether to return a Pandas series. If True, the index will be set to
        the observed years/months. If False, will return a 1d numpy array.

    dtype : type, optional (default=np.float64)
        The type to return for the array. Default is np.float64, which is used
        throughout the package as the default type.

    Notes
    -----
    This is monthly data, so *m* should be set to 12 when using in a seasonal
    context.

    Examples
    --------
    >>> from pmdarima.datasets import load_sunspots
    >>> load_sunspots()
    array([58. , 62.6, 70. , ..., 55.8, 33.3, 33.4])

    >>> load_sunspots(True).head()
    Jan 1749    58.0
    Feb 1749    62.6
    Mar 1749    70.0
    Apr 1749    55.7
    May 1749    85.0
    dtype: float64

    References
    ----------
    .. [1] https://www.rdocumentation.org/packages/datasets/versions/3.6.1/topics/sunspots  # noqa: E501

    Returns
    -------
    rslt : array-like, shape=(n_samples,)
        The sunspots dataset. There are 2820 observations.
    """
    rslt = base._cache.get('sunspots', None)
    if rslt is None:
        data_path = join(base.get_data_path(), 'sunspots.txt.gz')
        rslt = np.loadtxt(data_path).ravel()
        base._cache['sunspots'] = rslt

    # don't want to cache type conversion
    rslt = rslt.astype(dtype)

    if not as_series:
        return rslt

    # Otherwise we want a series and have to cleverly create the index
    index = [
        "%s %i" % (calendar.month_abbr[i + 1], year)
        for year in range(1749, 1984)
        for i in range(12)
    ]

    return pd.Series(rslt, index=index)
