# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# This is the lynx dataset found in R.

import numpy as np
import pandas as pd

from ..compat import DTYPE

__all__ = [
    'load_lynx'
]


def load_lynx(as_series=False, dtype=DTYPE):
    """Annual numbers of lynx trappings for 1821–1934 in Canada.

    This time-series records the number of skins of predators (lynx) that were
    collected over several years by the Hudson's Bay Company. The dataset was
    taken from Brockwell & Davis (1991) and appears to be the series
    considered by Campbell & Walker (1977).

    Parameters
    ----------
    as_series : bool, optional (default=False)
        Whether to return a Pandas series. If True, the index will be set to
        the observed years. If False, will return a 1d numpy array.

    dtype : type, optional (default=np.float64)
        The type to return for the array. Default is np.float64, which is used
        throughout the package as the default type.

    Examples
    --------
    >>> from pmdarima.datasets import load_lynx
    >>> load_lynx()
    array([ 269,  321,  585,  871, 1475, 2821, 3928, 5943, 4950, 2577,  523,
             98,  184,  279,  409, 2285, 2685, 3409, 1824,  409,  151,   45,
             68,  213,  546, 1033, 2129, 2536,  957,  361,  377,  225,  360,
            731, 1638, 2725, 2871, 2119,  684,  299,  236,  245,  552, 1623,
           3311, 6721, 4254,  687,  255,  473,  358,  784, 1594, 1676, 2251,
           1426,  756,  299,  201,  229,  469,  736, 2042, 2811, 4431, 2511,
            389,   73,   39,   49,   59,  188,  377, 1292, 4031, 3495,  587,
            105,  153,  387,  758, 1307, 3465, 6991, 6313, 3794, 1836,  345,
            382,  808, 1388, 2713, 3800, 3091, 2985, 3790,  674,   81,   80,
            108,  229,  399, 1132, 2432, 3574, 2935, 1537,  529,  485,  662,
           1000, 1590, 2657, 3396])

    >>> load_lynx(True).head()
    1821     269
    1822     321
    1823     585
    1824     871
    1825    1475
    dtype: int64

    Notes
    -----
    This is annual data and not seasonal in nature (i.e., :math:`m=1`)

    References
    ----------
    .. [1] Brockwell, P. J. and Davis, R. A. (1991)
           Time Series and Forecasting Methods. Second edition.
           Springer. Series G (page 557).

    .. [2] https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/lynx.html  # noqa: E501

    Returns
    -------
    lynx : array-like, shape=(n_samples,)
        The lynx dataset. There are 114 observations.
    """
    rslt = np.array([269, 321, 585, 871, 1475, 2821, 3928, 5943, 4950,
                     2577, 523, 98, 184, 279, 409, 2285, 2685, 3409,
                     1824, 409, 151, 45, 68, 213, 546, 1033, 2129,
                     2536, 957, 361, 377, 225, 360, 731, 1638, 2725,
                     2871, 2119, 684, 299, 236, 245, 552, 1623, 3311,
                     6721, 4254, 687, 255, 473, 358, 784, 1594, 1676,
                     2251, 1426, 756, 299, 201, 229, 469, 736, 2042,
                     2811, 4431, 2511, 389, 73, 39, 49, 59, 188,
                     377, 1292, 4031, 3495, 587, 105, 153, 387, 758,
                     1307, 3465, 6991, 6313, 3794, 1836, 345, 382, 808,
                     1388, 2713, 3800, 3091, 2985, 3790, 674, 81, 80,
                     108, 229, 399, 1132, 2432, 3574, 2935, 1537, 529,
                     485, 662, 1000, 1590, 2657, 3396]).astype(dtype)

    # Set the index if necessary
    if as_series:
        return pd.Series(rslt, index=range(1821, 1935))
    return rslt
