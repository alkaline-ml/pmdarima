# pmdarima

[![PyPI version](https://badge.fury.io/py/pmdarima.svg)](https://badge.fury.io/py/pmdarima)
[![CircleCI](https://circleci.com/gh/alkaline-ml/pmdarima.svg?style=svg)](https://circleci.com/gh/alkaline-ml/pmdarima)
[![Build Status](https://dev.azure.com/tgsmith61591gh/pmdarima/_apis/build/status/alkaline-ml.pmdarima?branchName=master)](https://dev.azure.com/tgsmith61591gh/pmdarima/_build/latest?definitionId=3&branchName=master)
[![codecov](https://codecov.io/gh/alkaline-ml/pmdarima/branch/master/graph/badge.svg)](https://codecov.io/gh/alkaline-ml/pmdarima)
![Supported versions](https://img.shields.io/badge/python-3.5+-blue.svg)
![Downloads](https://img.shields.io/badge/dynamic/json?color=blue&label=downloads&query=%24.total&url=https%3A%2F%2Fstore.zapier.com%2Fapi%2Frecords%3Fsecret%3D1e061b29db6c4f15af01103d403b0237)
![Downloads/Week](https://img.shields.io/badge/dynamic/json?color=blue&label=downloads%2Fweek&query=%24.weekly&url=https%3A%2F%2Fstore.zapier.com%2Fapi%2Frecords%3Fsecret%3D1e061b29db6c4f15af01103d403b0237)

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
