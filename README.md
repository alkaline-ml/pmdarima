[![Build status](https://travis-ci.org/tgsmith61591/pyramid.svg?branch=master)](https://travis-ci.org/tgsmith61591/pyramid)
[![Coverage Status](https://coveralls.io/repos/github/tgsmith61591/pyramid/badge.svg?branch=master)](https://coveralls.io/github/tgsmith61591/pyramid?branch=master)
![Supported versions](https://img.shields.io/badge/python-2.7-blue.svg)
![Supported versions](https://img.shields.io/badge/python-3.5-blue.svg)

# pyramid
Pyramid is a no-nonsense statistical Python library with a solitary objective: bring R's
[`auto.arima`](https://www.rdocumentation.org/packages/forecast/versions/7.3/topics/auto.arima)
functionality to Python. Pyramid operates by wrapping
[`statsmodels.tsa.arima_model`](https://github.com/statsmodels/statsmodels) and creating a more
user-friendly estimator interface for programmers familiar with scikit-learn. Unfortunately, since
statsmodels does not currently support seasonal ARIMA models, neither does Pyramid.
