# -*- coding: utf-8 -*-
#
# Benchmark various approaches to functions to speed things up.
# ... hopefully.

import numpy as np

import time


def _do_time(func, n_iter=10, *args, **kwargs):
    times = []
    for _ in range(n_iter):
        start = time.time()
        func(*args, **kwargs)
        times.append(time.time() - start)

    times = np.asarray(times)
    print("Completed %i iterations (avg=%.6f, min=%.6f, max=%.6f)"
          % (n_iter, times.mean(), times.min(), times.max()))


def benchmark_is_constant():
    """This benchmarks the "is_constant" function from ``pmdarima.arima.utils``
    This was added in 0.6.2.
    """
    # WINNER!
    def is_const1(x):
        """This is the version in Pyramid 0.6.2.

        Parameters
        ----------
        x : np.ndarray
            This is the array.
        """
        return (x == x[0]).all()

    def is_const2(x):
        """This should ostensibly only take O(N) rather than O(2N) like
        its predecessor. But we'll see...

        Parameters
        ----------
        x : np.ndarray
            This is the array.
        """
        return np.unique(x).shape[0] == 1

    x = np.random.choice(np.arange(10), 1000000, replace=True)
    _do_time(is_const1, 25, x)
    _do_time(is_const2, 25, x)


if __name__ == '__main__':
    benchmark_is_constant()
