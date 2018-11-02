# -*- coding: utf-8 -*-
#
# Private configuration

from __future__ import absolute_import

import os
from os.path import expanduser

import warnings

# The directory in which we'll store TimeSeries models from statsmodels
# during the internal ARIMA pickling operation.
PYRAMID_ARIMA_CACHE = os.environ.get('PYRAMID_ARIMA_CACHE',
                                     expanduser('~/.pyramid-arima-cache'))

# The pattern of the pickle file for a saved ARIMA
PICKLE_HASH_PATTERN = '%s-%s-%i.pmdpkl'

# The size of the pyramid cache above which to warn the user
cwb = os.environ.get('PYRAMID_ARIMA_CACHE_WARN_SIZE', 1e8)
try:
    CACHE_WARN_BYTES = int(cwb)
except ValueError:
    warnings.warn('The value of PYRAMID_ARIMA_CACHE_WARN_SIZE should be '
                  'an integer, but got "{cache_val}". Defaulting to 1e8.'
                  .format(cache_val=cwb))
    CACHE_WARN_BYTES = 1e8  # 100MB default


def _warn_for_cache_size():
    """Warn for a cache size that is too large.

    This is called on the initial import and warns if the size of the cached
    statsmodels TS objects exceeds the CACHE_WARN_BYTES value.
    """
    from os.path import join, getsize, isfile
    try:
        cache_size = sum(getsize(join(PYRAMID_ARIMA_CACHE, f))
                         for f in os.listdir(PYRAMID_ARIMA_CACHE)
                         if isfile(join(PYRAMID_ARIMA_CACHE, f)))
    except OSError as ose:
        # If it's OSError no 2, it means the cache doesn't exist yet, which
        # is fine. Otherwise it's something else and we need to raise.
        if ose.errno != 2:
            raise

    else:
        if cache_size > CACHE_WARN_BYTES:
            warnings.warn("The Pyramid cache ({cache_loc}) has grown to "
                          "{nbytes:,} bytes. Consider cleaning out old ARIMA "
                          "models or increasing the max cache bytes with "
                          "'PYRAMID_ARIMA_CACHE_WARN_SIZE' (currently "
                          "{current_max:,} bytes) to avoid this warning in "
                          "the future."
                          .format(cache_loc=PYRAMID_ARIMA_CACHE,
                                  nbytes=cache_size,
                                  current_max=int(CACHE_WARN_BYTES)),
                          UserWarning)
