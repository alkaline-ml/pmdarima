.. _quickstart:

==========
Quickstart
==========

Since pmdarima is intended to replace R's ``auto.arima``, the interface is
designed to be quick to learn and easy to use, even for R users making the switch.
Common functions and tools are elevated to the top-level of the package:

.. code-block:: python

    import pmdarima as pm

    # Create an array like you would in R
    x = pm.c(1, 2, 3, 4, 5, 6, 7)

    # Compute an auto-correlation like you would in R:
    pm.acf(x)

    # Plot an auto-correlation:
    pm.plot_acf(x)


Auto-ARIMA example
------------------

Here's a quick example of how we can fit an ``auto_arima`` with pmdarima:

.. code-block:: python

    import numpy as np
    import pmdarima as pm
    from pmdarima.datasets import load_wineind

    # this is a dataset from R
    wineind = load_wineind().astype(np.float64)

    # fit stepwise auto-ARIMA
    stepwise_fit = pm.auto_arima(wineind, start_p=1, start_q=1,
                                 max_p=3, max_q=3, m=12,
                                 start_P=0, seasonal=True,
                                 d=1, D=1, trace=True,
                                 error_action='ignore',  # don't want to know if an order does not work
                                 suppress_warnings=True,  # don't want convergence warnings
                                 stepwise=True)  # set to stepwise

It's easy to examine your model fit results. Simply use the ``summary`` method:

.. code-block:: python

    >>> stepwise_fit.summary()
    <class 'statsmodels.iolib.summary.Summary'>
    """
                                     Statespace Model Results
    ==========================================================================================
    Dep. Variable:                                  y   No. Observations:                  176
    Model:             SARIMAX(1, 1, 2)x(0, 1, 1, 12)   Log Likelihood               -1527.386
    Date:                            Mon, 04 Sep 2017   AIC                           3066.771
    Time:                                    13:59:01   BIC                           3085.794
    Sample:                                         0   HQIC                          3074.487
                                                - 176
    Covariance Type:                              opg
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    intercept   -100.7446     72.306     -1.393      0.164    -242.462      40.973
    ar.L1         -0.5139      0.390     -1.319      0.187      -1.278       0.250
    ma.L1         -0.0791      0.403     -0.196      0.844      -0.869       0.710
    ma.L2         -0.4438      0.223     -1.988      0.047      -0.881      -0.006
    ma.S.L12      -0.4021      0.054     -7.448      0.000      -0.508      -0.296
    sigma2      7.663e+06    7.3e+05     10.500      0.000    6.23e+06    9.09e+06
    ===================================================================================
    Ljung-Box (Q):                       48.66   Jarque-Bera (JB):                21.62
    Prob(Q):                              0.16   Prob(JB):                         0.00
    Heteroskedasticity (H):               1.18   Skew:                            -0.61
    Prob(H) (two-sided):                  0.54   Kurtosis:                         4.31
    ===================================================================================

    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).
    [2] Covariance matrix is singular or near-singular, with condition number 8.15e+14. Standard errors may be unstable.
    """
