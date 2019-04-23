# pmdarima

[![PyPI version](https://badge.fury.io/py/pmdarima.svg)](https://badge.fury.io/py/pmdarima)
[![Linux build status](https://travis-ci.org/tgsmith61591/pmdarima.svg?branch=master)](https://travis-ci.org/tgsmith61591/pmdarima)
[![CircleCI](https://circleci.com/gh/tgsmith61591/pmdarima.svg?style=svg)](https://circleci.com/gh/tgsmith61591/pmdarima)
[![Build Status](https://dev.azure.com/tgsmith61591gh/pmdarima/_apis/build/status/tgsmith61591.pmdarima?branchName=master)](https://dev.azure.com/tgsmith61591gh/pmdarima/_build/latest?definitionId=1&branchName=master)
[![codecov](https://codecov.io/gh/tgsmith61591/pmdarima/branch/master/graph/badge.svg)](https://codecov.io/gh/tgsmith61591/pmdarima)
![Supported versions](https://img.shields.io/badge/python-3.5+-blue.svg)

Pmdarima (originally `pyramid-arima`, for the anagram of 'py' + 'arima') is a no-nonsense statistical
Python library with a solitary objective: bring R's
[`auto.arima`](https://www.rdocumentation.org/packages/forecast/versions/7.3/topics/auto.arima)
functionality to Python. Pmdarima operates by wrapping
[`statsmodels.tsa.ARIMA`](https://github.com/statsmodels/statsmodels/blob/master/statsmodels/tsa/arima_model.py)
and [`statsmodels.tsa.statespace.SARIMAX`](https://github.com/statsmodels/statsmodels/blob/master/statsmodels/tsa/statespace/sarimax.py)
into one estimator class and creating a more user-friendly estimator interface for programmers familiar with scikit-learn.


## Installation

Pmdarima is on pypi under the package name `pmdarima` and can be downloaded via `pip`:

```bash
$ pip install pmdarima
```
 
Note that legacy versions (<1.0.0) are available under the name
"`pyramid-arima`" and can be pip installed via:

```bash
# Legacy warning:
$ pip install pyramid-arima
# python -c 'import pyramid;'
```

To ensure the package was built correctly, import the following module in python:

```python
from pmdarima.arima import auto_arima
```


### Availability

`pmdarima` is available in pre-built Wheel files for Python 3.5+ for the following platforms:

* Mac (64-bit)
* Linux (64-bit manylinux)
* Windows (32 & 64-bit)
  
If a wheel doesn't exist for your platform, you can still `pip install` and it
will build from the source distribution tarball, however you'll need `cython>=0.29`
and `gcc` (Mac/Linux) or `MinGW` (Windows) in order to build the package from source.


### Documentation

All of your questions and more (including examples and guides) can be answered by
the [`pmdarima` documentation](https://www.alkaline-ml.com/pmdarima). If not, always
feel free to file an issue.
