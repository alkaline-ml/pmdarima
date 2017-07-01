[![Build status](https://travis-ci.org/tgsmith61591/pyramid.svg?branch=master)](https://travis-ci.org/tgsmith61591/pyramid)
[![Coverage Status](https://coveralls.io/repos/github/tgsmith61591/pyramid/badge.svg?branch=master)](https://coveralls.io/github/tgsmith61591/pyramid?branch=master)
![Supported versions](https://img.shields.io/badge/python-2.7-blue.svg)
![Supported versions](https://img.shields.io/badge/python-3.5-blue.svg)

# pyramid :chart_with_upwards_trend:
Pyramid is a no-nonsense statistical Python library with a solitary objective: bring R's
[`auto.arima`](https://www.rdocumentation.org/packages/forecast/versions/7.3/topics/auto.arima)
functionality to Python. Pyramid operates by wrapping
[`statsmodels.tsa.ARIMA`](https://github.com/statsmodels/statsmodels/blob/master/statsmodels/tsa/arima_model.py) and
[`statsmodels.tsa.statespace.SARIMX`](https://github.com/statsmodels/statsmodels/blob/master/statsmodels/tsa/statespace/sarimax.py)
into one estimator class and creating a more user-friendly estimator interface for programmers familiar with scikit-learn.


## Dependencies

Pyramid depends on:
  - numpy >= 1.10
  - scipy >= 0.9
  - scikit-learn >= 0.17
  - statsmodels >= 0.8


## Installation

Pyramid is not currently on pypi or conda. The best way to install, then, is by cloning from git:

```bash
$ git clone https://github.com/tgsmith61591/pyramid.git
$ cd pyramid
$ python setup.py install
```


### Quickstart

__For an easy, reproducible quick-start example, see [examples/](examples/quick_start_example.ipynb).__
