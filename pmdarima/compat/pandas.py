# -*- coding: utf-8 -*-

from __future__ import absolute_import

# Importing visualization modules changes in version 0.19
try:  # <= v0.19
    from pandas.tools.plotting import autocorrelation_plot
except ImportError:  # 0.20+
    from pandas.plotting import autocorrelation_plot
