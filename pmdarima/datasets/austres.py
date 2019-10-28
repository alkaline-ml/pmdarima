# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from ..compat import DTYPE

__all__ = [
    'load_austres'
]


def load_austres(as_series=False, dtype=DTYPE):
    """Quarterly residential data.

    Numbers (in thousands) of Australian residents measured quarterly from
    March 1971 to March 1994.

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
        The austres vector.

    Examples
    --------
    >>> from pmdarima.datasets import load_austres
    >>> load_austres()
    np.array([13067.3, 13130.5, 13198.4, 13254.2, 13303.7, 13353.9,
              13409.3, 13459.2, 13504.5, 13552.6, 13614.3, 13669.5,
              13722.6, 13772.1, 13832.0, 13862.6, 13893.0, 13926.8,
              13968.9, 14004.7, 14033.1, 14066.0, 14110.1, 14155.6,
              14192.2, 14231.7, 14281.5, 14330.3, 14359.3, 14396.6,
              14430.8, 14478.4, 14515.7, 14554.9, 14602.5, 14646.4,
              14695.4, 14746.6, 14807.4, 14874.4, 14923.3, 14988.7,
              15054.1, 15121.7, 15184.2, 15239.3, 15288.9, 15346.2,
              15393.5, 15439.0, 15483.5, 15531.5, 15579.4, 15628.5,
              15677.3, 15736.7, 15788.3, 15839.7, 15900.6, 15961.5,
              16018.3, 16076.9, 16139.0, 16203.0, 16263.3, 16327.9,
              16398.9, 16478.3, 16538.2, 16621.6, 16697.0, 16777.2,
              16833.1, 16891.6, 16956.8, 17026.3, 17085.4, 17106.9,
              17169.4, 17239.4, 17292.0, 17354.2, 17414.2, 17447.3,
              17482.6, 17526.0, 17568.7, 17627.1, 17661.5])

    >>> load_austres(True).head()
    0    13067.3
    1    13130.5
    2    13198.4
    3    13254.2
    4    13303.7
    dtype: float64

    Notes
    -----
    This is quarterly data, so *m* should be set to 4 when using in a seasonal
    context.

    References
    ----------
    .. [1] P. J. Brockwell and R. A. Davis (1996)
           "Introduction to Time Series and Forecasting." Springer
    """
    rslt = np.array([
        13067.3, 13130.5, 13198.4, 13254.2, 13303.7, 13353.9,
        13409.3, 13459.2, 13504.5, 13552.6, 13614.3, 13669.5,
        13722.6, 13772.1, 13832.0, 13862.6, 13893.0, 13926.8,
        13968.9, 14004.7, 14033.1, 14066.0, 14110.1, 14155.6,
        14192.2, 14231.7, 14281.5, 14330.3, 14359.3, 14396.6,
        14430.8, 14478.4, 14515.7, 14554.9, 14602.5, 14646.4,
        14695.4, 14746.6, 14807.4, 14874.4, 14923.3, 14988.7,
        15054.1, 15121.7, 15184.2, 15239.3, 15288.9, 15346.2,
        15393.5, 15439.0, 15483.5, 15531.5, 15579.4, 15628.5,
        15677.3, 15736.7, 15788.3, 15839.7, 15900.6, 15961.5,
        16018.3, 16076.9, 16139.0, 16203.0, 16263.3, 16327.9,
        16398.9, 16478.3, 16538.2, 16621.6, 16697.0, 16777.2,
        16833.1, 16891.6, 16956.8, 17026.3, 17085.4, 17106.9,
        17169.4, 17239.4, 17292.0, 17354.2, 17414.2, 17447.3,
        17482.6, 17526.0, 17568.7, 17627.1, 17661.5]).astype(dtype)

    if as_series:
        return pd.Series(rslt)
    return rslt
