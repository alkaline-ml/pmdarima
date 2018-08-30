# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# This is the woolyrnq dataset found in the R forecast package.

from __future__ import absolute_import

import numpy as np
import pandas as pd

__all__ = [
    'load_woolyrnq'
]


def load_woolyrnq(as_series=False):
    """Quarterly production of woollen yarn in Australia.

    This time-series records the quarterly production (in tonnes) of woollen
    yarn in Australia between Mar 1965 and Sep 1994.

    Parameters
    ----------
    as_series : bool, optional (default=False)
        Whether to return a Pandas series. If True, the index will be set to
        the observed years/quarters. If False, will return a 1d numpy array.

    Examples
    --------
    >>> from pyramid.datasets import load_woolyrnq
    >>> load_woolyrnq()
    array([6172, 6709, 6633, 6660, 6786, 6800, 6730, 6765, 6720, 7133, 6946,
           7095, 7047, 6757, 6915, 6921, 7064, 7206, 7190, 7402, 7819, 7300,
           7105, 7259, 7001, 7475, 6840, 7061, 5845, 7529, 7819, 6943, 5714,
           6556, 7045, 5947, 5463, 6127, 5540, 4235, 3324, 4793, 5906, 5834,
           5240, 5458, 5505, 5002, 3999, 4826, 5318, 4681, 4442, 5305, 5466,
           4995, 4573, 5081, 5696, 5079, 4373, 4986, 5341, 4800, 4161, 5007,
           5464, 5127, 4240, 5338, 5129, 4437, 3642, 4602, 5524, 4895, 4380,
           5186, 6080, 5588, 5009, 5663, 6540, 6262, 5169, 5819, 6339, 5981,
           4766, 5976, 6590, 5590, 5135, 5762, 6077, 5882, 4247, 5264, 5146,
           4868, 4329, 4869, 5127, 4868, 3827, 4987, 5222, 4928, 3930, 4469,
           4954, 4752, 3888, 4588, 5309, 4732, 4837, 6135, 6396])

    >>> load_woolyrnq(True).head()
    Q1 1965    6172
    Q2 1965    6709
    Q3 1965    6633
    Q4 1965    6660
    Q1 1966    6786
    dtype: int64

    References
    ----------
    .. [1] https://www.rdocumentation.org/packages/forecast/versions/8.1/topics/woolyrnq

    Returns
    -------
    rslt : array-like, shape=(n_samples,)
        The woolyrnq dataset. There are 119 observations.
    """
    rslt = np.array([
        6172, 6709, 6633, 6660,
        6786, 6800, 6730, 6765,
        6720, 7133, 6946, 7095,
        7047, 6757, 6915, 6921,
        7064, 7206, 7190, 7402,
        7819, 7300, 7105, 7259,
        7001, 7475, 6840, 7061,
        5845, 7529, 7819, 6943,
        5714, 6556, 7045, 5947,
        5463, 6127, 5540, 4235,
        3324, 4793, 5906, 5834,
        5240, 5458, 5505, 5002,
        3999, 4826, 5318, 4681,
        4442, 5305, 5466, 4995,
        4573, 5081, 5696, 5079,
        4373, 4986, 5341, 4800,
        4161, 5007, 5464, 5127,
        4240, 5338, 5129, 4437,
        3642, 4602, 5524, 4895,
        4380, 5186, 6080, 5588,
        5009, 5663, 6540, 6262,
        5169, 5819, 6339, 5981,
        4766, 5976, 6590, 5590,
        5135, 5762, 6077, 5882,
        4247, 5264, 5146, 4868,
        4329, 4869, 5127, 4868,
        3827, 4987, 5222, 4928,
        3930, 4469, 4954, 4752,
        3888, 4588, 5309, 4732,
        4837, 6135, 6396])

    if not as_series:
        return rslt

    # Otherwise we want a series and have to cleverly create the index
    # (with quarters, and we don't want Q4 in 1994)
    index = [
        "Q%i %i" % (i + 1, year)
        for year in range(1965, 1995)
        for i in range(4)
    ][:-1]  # trim off the last one.

    return pd.Series(rslt, index=index)
