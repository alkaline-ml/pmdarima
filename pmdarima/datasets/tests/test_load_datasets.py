# -*- coding: utf-8 -*-

from __future__ import absolute_import

from pmdarima.datasets import load_heartrate, load_lynx, load_wineind,\
    load_woolyrnq, load_ausbeer, load_austres,\
    load_airpassengers, load_taylor, load_msft

import numpy as np
import pandas as pd

import pytest


# Simply test loading the datasets and that we get the expected type
@pytest.mark.parametrize(
    'f', [load_heartrate,
          load_lynx,
          load_wineind,
          load_woolyrnq,
          load_ausbeer,
          load_austres,
          load_taylor,
          load_airpassengers])
def test_load(f):
    for as_series in (True, False):
        x = f(as_series=as_series)

        if as_series:
            assert isinstance(x, pd.Series)
        else:
            assert isinstance(x, np.ndarray)


@pytest.mark.parametrize(
    'f', [load_msft])
def test_df_loads(f):
    df = f()
    assert isinstance(df, pd.DataFrame)
