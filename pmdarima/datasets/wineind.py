# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# This is the wineind dataset found in R.

import numpy as np
import pandas as pd

import calendar

from ..compat import DTYPE

__all__ = [
    'load_wineind'
]


def load_wineind(as_series=False, dtype=DTYPE):
    """Australian total wine sales by wine makers in bottles <= 1 litre.

    This time-series records wine sales by Australian wine makers between
    Jan 1980 -- Aug 1994. This dataset is found in the R ``forecast`` package.

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
    >>> from pmdarima.datasets import load_wineind
    >>> load_wineind()
    array([15136, 16733, 20016, 17708, 18019, 19227, 22893, 23739, 21133,
           22591, 26786, 29740, 15028, 17977, 20008, 21354, 19498, 22125,
           25817, 28779, 20960, 22254, 27392, 29945, 16933, 17892, 20533,
           23569, 22417, 22084, 26580, 27454, 24081, 23451, 28991, 31386,
           16896, 20045, 23471, 21747, 25621, 23859, 25500, 30998, 24475,
           23145, 29701, 34365, 17556, 22077, 25702, 22214, 26886, 23191,
           27831, 35406, 23195, 25110, 30009, 36242, 18450, 21845, 26488,
           22394, 28057, 25451, 24872, 33424, 24052, 28449, 33533, 37351,
           19969, 21701, 26249, 24493, 24603, 26485, 30723, 34569, 26689,
           26157, 32064, 38870, 21337, 19419, 23166, 28286, 24570, 24001,
           33151, 24878, 26804, 28967, 33311, 40226, 20504, 23060, 23562,
           27562, 23940, 24584, 34303, 25517, 23494, 29095, 32903, 34379,
           16991, 21109, 23740, 25552, 21752, 20294, 29009, 25500, 24166,
           26960, 31222, 38641, 14672, 17543, 25453, 32683, 22449, 22316,
           27595, 25451, 25421, 25288, 32568, 35110, 16052, 22146, 21198,
           19543, 22084, 23816, 29961, 26773, 26635, 26972, 30207, 38687,
           16974, 21697, 24179, 23757, 25013, 24019, 30345, 24488, 25156,
           25650, 30923, 37240, 17466, 19463, 24352, 26805, 25236, 24735,
           29356, 31234, 22724, 28496, 32857, 37198, 13652, 22784, 23565,
           26323, 23779, 27549, 29660, 23356])

    >>> load_wineind(True).head()
    Jan 1980    15136
    Feb 1980    16733
    Mar 1980    20016
    Apr 1980    17708
    May 1980    18019
    dtype: int64

    References
    ----------
    .. [1] https://www.rdocumentation.org/packages/forecast/versions/8.1/topics/wineind

    Returns
    -------
    rslt : array-like, shape=(n_samples,)
        The wineind dataset. There are 176 observations.
    """  # noqa: E501
    rslt = np.array([
        15136, 16733, 20016, 17708, 18019, 19227, 22893, 23739,
        21133, 22591, 26786, 29740, 15028, 17977, 20008, 21354,
        19498, 22125, 25817, 28779, 20960, 22254, 27392, 29945,
        16933, 17892, 20533, 23569, 22417, 22084, 26580, 27454,
        24081, 23451, 28991, 31386, 16896, 20045, 23471, 21747,
        25621, 23859, 25500, 30998, 24475, 23145, 29701, 34365,
        17556, 22077, 25702, 22214, 26886, 23191, 27831, 35406,
        23195, 25110, 30009, 36242, 18450, 21845, 26488, 22394,
        28057, 25451, 24872, 33424, 24052, 28449, 33533, 37351,
        19969, 21701, 26249, 24493, 24603, 26485, 30723, 34569,
        26689, 26157, 32064, 38870, 21337, 19419, 23166, 28286,
        24570, 24001, 33151, 24878, 26804, 28967, 33311, 40226,
        20504, 23060, 23562, 27562, 23940, 24584, 34303, 25517,
        23494, 29095, 32903, 34379, 16991, 21109, 23740, 25552,
        21752, 20294, 29009, 25500, 24166, 26960, 31222, 38641,
        14672, 17543, 25453, 32683, 22449, 22316, 27595, 25451,
        25421, 25288, 32568, 35110, 16052, 22146, 21198, 19543,
        22084, 23816, 29961, 26773, 26635, 26972, 30207, 38687,
        16974, 21697, 24179, 23757, 25013, 24019, 30345, 24488,
        25156, 25650, 30923, 37240, 17466, 19463, 24352, 26805,
        25236, 24735, 29356, 31234, 22724, 28496, 32857, 37198,
        13652, 22784, 23565, 26323, 23779, 27549, 29660, 23356]).astype(dtype)

    if not as_series:
        return rslt

    # Otherwise we want a series and have to cleverly create the index
    # (we don't want after aug in 1994, so trip Sep, Oct, Nov and Dec)
    index = [
        "%s %i" % (calendar.month_abbr[i + 1], year)
        for year in range(1980, 1995)
        for i in range(12)
    ][:-4]

    return pd.Series(rslt, index=index)
