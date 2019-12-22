# -*- coding: utf-8 -*-

# Importing visualization modules changes in version 0.19
try:  # <= v0.19
    from pandas.tools import plotting
except ImportError:  # 0.20+
    from pandas import plotting
