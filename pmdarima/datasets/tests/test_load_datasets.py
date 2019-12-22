# -*- coding: utf-8 -*-

from pmdarima.datasets import load_heartrate, load_lynx, load_wineind,\
    load_woolyrnq, load_ausbeer, load_austres,\
    load_airpassengers, load_taylor, load_msft, load_sunspots, _base as base

import numpy as np
import pandas as pd

import pytest


def _inner_load(f):
    n = None
    for as_series in (True, False):
        x = f(as_series=as_series)

        # ensure shape is same for both
        if n is None:
            n = x.shape[0]
        else:
            assert x.shape[0] == n

        if as_series:
            assert isinstance(x, pd.Series)
        else:
            assert isinstance(x, np.ndarray)


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
    _inner_load(f)


@pytest.mark.parametrize(
    'f', [load_msft])
def test_df_loads(f):
    df = f()
    assert isinstance(df, pd.DataFrame)


@pytest.mark.parametrize(
    'f, cache_name', [
        pytest.param(load_sunspots, 'sunspots'),
    ])
def test_load_from_gzip(f, cache_name):
    _inner_load(f)
    assert cache_name in base._cache
