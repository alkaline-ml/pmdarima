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

It's easy to examine your model fit results. Simply use the ``summary``:sup:`[1]` method:

.. code-block:: python

    >>> stepwise_fit.summary()
    <class 'statsmodels.iolib.summary.Summary'>
    """
                                          SARIMAX Results
    ============================================================================================
    Dep. Variable:                                    y   No. Observations:                  176
    Model:             SARIMAX(0, 1, 2)x(0, 1, [1], 12)   Log Likelihood               -1528.766
    Date:                              Wed, 15 Jun 2022   AIC                           3065.533
    Time:                                      12:38:14   BIC                           3077.908
    Sample:                                           0   HQIC                          3070.557
                                                  - 176
    Covariance Type:                                opg
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    ma.L1         -0.5756      0.041    -13.952      0.000      -0.656      -0.495
    ma.L2         -0.1065      0.048     -2.224      0.026      -0.200      -0.013
    ma.S.L12      -0.3848      0.054     -7.156      0.000      -0.490      -0.279
    sigma2      7.866e+06   7.01e+05     11.228      0.000    6.49e+06    9.24e+06
    ===================================================================================
    Ljung-Box (L1) (Q):                   2.84   Jarque-Bera (JB):                18.05
    Prob(Q):                              0.09   Prob(JB):                         0.00
    Heteroskedasticity (H):               1.17   Skew:                            -0.55
    Prob(H) (two-sided):                  0.56   Kurtosis:                         4.21
    ===================================================================================

    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (complex-step).

[1] The summary output was generated using the following versions:

.. code-block:: python

    >>> import pmdarima as pm
    >>> pm.show_versions()

    System:
        python: 3.9.7 (default, Nov 10 2021, 08:50:17)  [Clang 13.0.0 (clang-1300.0.29.3)]
    executable: /Users/asmith/venv/bin/python
       machine: macOS-11.6.6-x86_64-i386-64bit

    Python dependencies:
            pip: 21.2.3
     setuptools: 57.4.0
        sklearn: 1.1.1
    statsmodels: 0.13.2
          numpy: 1.22.4
          scipy: 1.8.1
         Cython: 0.29.30
         pandas: 1.4.2
         joblib: 1.1.0
       pmdarima: 1.8.5
