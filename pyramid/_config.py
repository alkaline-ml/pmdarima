# -*- coding: utf-8 -*-
#
# Private configuration

from __future__ import absolute_import

import os
from os.path import expanduser

# The directory in which we'll store TimeSeries models from statsmodels
# during the internal ARIMA pickling operation.
PYRAMID_ARIMA_CACHE = os.environ.get('PYRAMID_ARIMA_CACHE',
                                     expanduser('~/.pyramid-arima-cache'))

# The pattern of the pickle file for a saved ARIMA
PICKLE_HASH_PATTERN = '%s-%s-%i.pmdpkl'
