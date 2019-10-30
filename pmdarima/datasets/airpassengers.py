# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from ..compat import DTYPE

__all__ = [
    'load_airpassengers'
]


def load_airpassengers(as_series=False, dtype=DTYPE):
    """Monthly airline passengers.

    The classic Box & Jenkins airline data. Monthly totals of international
    airline passengers, 1949 to 1960.

    Parameters
    ----------
    as_series : bool, optional (default=False)
        Whether to return a Pandas series. If False, will return a 1d
        numpy array.

    dtype : type, optional (default=np.float64)
        The type to return for the array. Default is np.float64, which is used
        throughout the package as the default type.

    Returns
    -------
    rslt : array-like, shape=(n_samples,)
        The time series vector.

    Examples
    --------
    >>> from pmdarima.datasets import load_airpassengers
    >>> load_airpassengers()  # doctest: +SKIP
    np.array([
        112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
        115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
        145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
        171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
        196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
        204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
        242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
        284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
        315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
        340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
        360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
        417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432])

    >>> load_airpassengers(True).head()
    0    112.0
    1    118.0
    2    132.0
    3    129.0
    4    121.0
    dtype: float64

    Notes
    -----
    This is monthly data, so *m* should be set to 12 when using in a seasonal
    context.

    References
    ----------
    .. [1] Box, G. E. P., Jenkins, G. M. and Reinsel, G. C. (1976)
           "Time Series Analysis, Forecasting and Control. Third Edition."
           Holden-Day. Series G.
    """
    rslt = np.array([
        112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
        115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
        145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
        171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
        196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
        204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
        242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
        284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
        315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
        340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
        360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
        417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432
    ]).astype(dtype)

    if as_series:
        return pd.Series(rslt)
    return rslt
