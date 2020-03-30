# -*- coding: utf-8 -*-

import os
from os.path import abspath, dirname, join, expanduser
import numpy as np
import pandas as pd
import urllib3

from ..compat.numpy import DTYPE

try:
    import cPickle as pickle
except ImportError:
    import pickle

# caches anything read from disk to avoid re-reads
_cache = {}
http = urllib3.PoolManager()


def get_data_path():
    """Get the absolute path to the ``data`` directory"""
    dataset_dir = abspath(dirname(__file__))
    data_dir = join(dataset_dir, 'data')
    return data_dir


def get_data_cache_path():
    """Get the absolute path to where we cache data from the web"""
    return abspath(expanduser(join("~", ".pmdarima-data")))


def fetch_from_web_or_disk(url, key, cache=True, dtype=DTYPE):
    """Fetch a dataset from the web, and save it in the pmdarima cache"""
    if key in _cache:
        return _cache[key]

    disk_cache_path = get_data_cache_path()

    # don't ask, just tell. avoid race conditions
    os.makedirs(disk_cache_path, exist_ok=True)

    # See if it's already there
    data_path = join(disk_cache_path, key + '.csv.gz')
    if os.path.exists(data_path):
        rslt = np.loadtxt(data_path).ravel()

    else:
        r = None
        rslt = None
        try:
            r = http.request('GET', url)
            # rank 1 because it's a time series
            rslt = np.asarray(
                r.data.decode('utf-8').split('\n'), dtype=dtype)

        finally:
            if rslt is not None:
                try:
                    r.release_conn()
                except Exception:
                    pass

        # if we got here, rslt is good. We need to save it to disk
        np.savetxt(fname=data_path, X=rslt)

    # If we get here, we have rslt.
    if cache:
        _cache[key] = rslt

    return rslt


def _load_pickle(key):
    """Internal method for loading a pickle file"""
    base_path = abspath(dirname(__file__))
    file_path = join(base_path, "data", key)
    with open(file_path, "rb") as pkl:
        return pickle.load(pkl)


def load_date_example():
    """Loads a nondescript dated example for internal use"""
    X = _load_pickle("dated.pkl")
    # make sure it's a date time
    X.loc[:, 'date'] = pd.to_datetime(X['date'])
    y = X.pop('y')
    return y, X
