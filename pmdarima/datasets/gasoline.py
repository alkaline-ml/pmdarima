# -*- coding: utf-8 -*-

import pandas as pd

from ..compat.numpy import DTYPE
from ._base import fetch_from_web_or_disk

__all__ = [
    'load_gasoline'
]

url = 'http://alkaline-ml.com/datasets/gasoline.csv'


def load_gasoline(as_series=False, dtype=DTYPE):
    """Weekly US finished motor gasoline products

    A weekly time series of US finished motor gasoline products supplied (in
    thousands of barrels per day) from February 1991 to May 2005.

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
    The seasonal periodicity of this example is rather difficult, since it's
    not an integer. To be exact, the periodicity is ``365.25 / 7``
    (~=52.1785714285714). To fit the best possible model to this data, you'll
    need to explore using exogenous features

    See Also
    --------
    :class:`pmdarima.preprocessing.exog.FourierFeaturizer`

    Examples
    --------
    >>> from pmdarima.datasets import load_gasoline
    >>> load_gasoline()
    array([6621. , 6433. , 6582. , ..., 9024. , 9175. , 9269. ])

    >>> load_gasoline(True).head()
    0    6621.0
    1    6433.0
    2    6582.0
    3    7224.0
    4    6875.0
    dtype: float64

    References
    ----------
    .. [1] http://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=wgfupus2&f=W  # noqa
    .. [2] https://robjhyndman.com/hyndsight/forecasting-weekly-data/

    Returns
    -------
    rslt : array-like, shape=(n_samples,)
        The gasoline dataset. There are 745 examples.
    """
    rslt = fetch_from_web_or_disk(url, 'gasoline', cache=True).astype(dtype)
    if not as_series:
        return rslt

    return pd.Series(rslt)
