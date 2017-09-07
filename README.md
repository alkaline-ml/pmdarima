[![PyPI version](https://badge.fury.io/py/pyramid-arima.svg)](https://badge.fury.io/py/pyramid-arima)
[![Linux build status](https://travis-ci.org/tgsmith61591/pyramid.svg?branch=master)](https://travis-ci.org/tgsmith61591/pyramid)
[![Windows build status](https://ci.appveyor.com/api/projects/status/592vawuu69kd6d21?svg=true)](https://ci.appveyor.com/project/tgsmith61591/pyramid)
[![Coverage Status](https://coveralls.io/repos/github/tgsmith61591/pyramid/badge.svg?branch=master)](https://coveralls.io/github/tgsmith61591/pyramid?branch=master)
![Supported versions](https://img.shields.io/badge/python-2.7-blue.svg)
![Supported versions](https://img.shields.io/badge/python-3.5-blue.svg)

# pyramid
Pyramid is a no-nonsense statistical Python library with a solitary objective: bring R's
[`auto.arima`](https://www.rdocumentation.org/packages/forecast/versions/7.3/topics/auto.arima)
functionality to Python. Pyramid operates by wrapping
[`statsmodels.tsa.ARIMA`](https://github.com/statsmodels/statsmodels/blob/master/statsmodels/tsa/arima_model.py) and
[`statsmodels.tsa.statespace.SARIMAX`](https://github.com/statsmodels/statsmodels/blob/master/statsmodels/tsa/statespace/sarimax.py)
into one estimator class and creating a more user-friendly estimator interface for programmers familiar with scikit-learn.


## Installation

Pyramid is on pypi under the package name `pyramid-arima` and can be downloaded via `pip`:

```bash
$ pip install pyramid-arima
```

To ensure the package was built correctly, import the following module in python:

```python
from pyramid.arima import auto_arima
```


### Quickstart

__For an easy, reproducible quick-start example, see [examples/](examples/quick_start_example.ipynb).__


### Other considerations

- How do I make pyramid run as quickly as R?
  - R's code is heavily C-based. Pyramid runs on statsmodels, which is Python based. There will be some differences in performance
    speed-wise, but much of it can be eliminated by using `stepwise=True`. See [this discussion](https://stackoverflow.com/questions/40871602/sarimax-model-fitting-too-slow-in-statsmodels)
    for more thoughts...

- Refreshing ARIMA models
  - Periodically, your ARIMA will need to be refreshed given new observations. See [this discussion](https://stats.stackexchange.com/questions/34139/updating-arima-models-at-frequent-intervals)
    and [this discussion](https://stats.stackexchange.com/questions/57745/what-do-you-consider-a-new-model-versus-an-updated-model-time-series)
    on either re-using `auto_arima`-estimated order terms or re-fitting altogether.
