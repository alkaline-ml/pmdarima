# -*- coding: utf-8 -*-

from __future__ import absolute_import

from pyramid.datasets import load_heartrate, load_lynx, \
    load_wineind, load_woolyrnq

import numpy as np
import pandas as pd


# Simply test loading the datasets and that we get the expected type
def test_load():
    for f in (load_heartrate, load_lynx, load_wineind, load_woolyrnq):
        for as_series in (True, False):
            x = f(as_series=as_series)

            if as_series:
                assert isinstance(x, pd.Series)
            else:
                assert isinstance(x, np.ndarray)
