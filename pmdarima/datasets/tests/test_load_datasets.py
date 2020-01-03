# -*- coding: utf-8 -*-

from pmdarima.datasets import load_heartrate, load_lynx, load_wineind,\
    load_woolyrnq, load_ausbeer, load_austres, load_gasoline, \
    load_airpassengers, load_taylor, load_msft, load_sunspots, _base as base

import numpy as np
import pandas as pd
import os
import shutil

from numpy.testing import assert_array_equal
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


@pytest.mark.parametrize(
    'func, key', [
        pytest.param(load_gasoline, 'gasoline'),
    ]
)
def test_load_from_web(func, key):
    # make sure there is no data folder
    disk_cache_folder = base.get_data_cache_path()
    if os.path.exists(disk_cache_folder):
        shutil.rmtree(disk_cache_folder)

    try:
        # loads from web
        y = func(as_series=False)

        # show the key is in _cache
        assert key in base._cache

        # show exists on disk
        assert os.path.exists(os.path.join(disk_cache_folder, key + '.csv.gz'))

        # pop from cache so we can load it from disk
        base._cache.pop(key)
        x = func(as_series=True)  # true for coverage

        assert_array_equal(y, x.values)

    finally:
        if os.path.exists(disk_cache_folder):
            shutil.rmtree(disk_cache_folder)
