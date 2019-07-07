# -*- coding: utf-8 -*-

from os.path import join, abspath, dirname

try:
    import cPickle as pickle
except ImportError:
    import pickle

__all__ = ['load_msft']


def _load_internal(key):
    base_path = abspath(dirname(__file__))
    file_path = join(base_path, "data", key)
    with open(file_path, "rb") as pkl:
        return pickle.load(pkl)


def load_msft():
    """Load the microsoft stock data

    Financial data for the MSFT stock between the dates of Mar 13, 1986 and
    Nov 10, 2017. This data is part of the Kaggle stock dataset [1]. Features
    are as follows:

        * Date : datetime
        * Open : float32
        * High : float32
        * Low : float32
        * Close : float32
        * Volume : long
        * OpenInt : int

    References
    ----------
    .. [1] https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs  # noqa:E501

    Returns
    -------
    df : pd.DataFrame, shape=(7983, 7)
        A dataframe of endog and exog values.
    """
    return _load_internal("msft.pkl")
