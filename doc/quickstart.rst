==================
Pyramid Quickstart
==================

Here is a simple example of Pyramid use:

.. code-block:: python

    >>> import numpy as np
    >>> from pyramid.arima import auto_arima
    >>> from pyramid.datasets import load_wineind

    # this is a dataset from R
    >>> wineind = load_wineind().astype(np.float64)

    >>> stepwise_fit = auto_arima(wineind, start_p=1, start_q=1,
    ...                           max_p=3, max_q=3, m=12,
    ...                           start_P=0, seasonal=True,
    ...                           d=1, D=1, trace=True,
    ...                           error_action='ignore',  # don't want to know if an order does not work
    ...                           suppress_warnings=True,  # don't want convergence warnings
    ...                           stepwise=True)  # set to stepwise
    Fit ARIMA: order=(1, 1, 1) seasonal_order=(0, 1, 1, 12); AIC=3066.811, BIC=3082.663, Fit time=0.517 seconds
    Fit ARIMA: order=(0, 1, 0) seasonal_order=(0, 1, 0, 12); AIC=nan, BIC=nan, Fit time=nan seconds
    Fit ARIMA: order=(1, 1, 0) seasonal_order=(1, 1, 0, 12); AIC=3099.735, BIC=3112.417, Fit time=0.162 seconds
    Fit ARIMA: order=(0, 1, 1) seasonal_order=(0, 1, 1, 12); AIC=3066.983, BIC=3079.665, Fit time=0.164 seconds
    Fit ARIMA: order=(1, 1, 1) seasonal_order=(1, 1, 1, 12); AIC=3067.666, BIC=3086.688, Fit time=0.645 seconds
    Fit ARIMA: order=(1, 1, 1) seasonal_order=(0, 1, 0, 12); AIC=3088.109, BIC=3100.791, Fit time=0.136 seconds
    Fit ARIMA: order=(1, 1, 1) seasonal_order=(0, 1, 2, 12); AIC=3067.669, BIC=3086.692, Fit time=1.512 seconds
    Fit ARIMA: order=(1, 1, 1) seasonal_order=(1, 1, 2, 12); AIC=3068.757, BIC=3090.951, Fit time=1.651 seconds
    Fit ARIMA: order=(2, 1, 1) seasonal_order=(0, 1, 1, 12); AIC=3067.485, BIC=3086.508, Fit time=0.445 seconds
    Fit ARIMA: order=(1, 1, 0) seasonal_order=(0, 1, 1, 12); AIC=3094.578, BIC=3107.260, Fit time=0.174 seconds
    Fit ARIMA: order=(1, 1, 2) seasonal_order=(0, 1, 1, 12); AIC=3066.771, BIC=3085.794, Fit time=0.425 seconds
    Fit ARIMA: order=(2, 1, 3) seasonal_order=(0, 1, 1, 12); AIC=3070.642, BIC=3096.006, Fit time=0.966 seconds
    Fit ARIMA: order=(1, 1, 2) seasonal_order=(1, 1, 1, 12); AIC=3068.086, BIC=3090.280, Fit time=0.411 seconds
    Fit ARIMA: order=(1, 1, 2) seasonal_order=(0, 1, 0, 12); AIC=3090.977, BIC=3106.830, Fit time=0.249 seconds
    Fit ARIMA: order=(1, 1, 2) seasonal_order=(0, 1, 2, 12); AIC=3067.766, BIC=3089.959, Fit time=1.170 seconds
    Fit ARIMA: order=(1, 1, 2) seasonal_order=(1, 1, 2, 12); AIC=3069.717, BIC=3095.081, Fit time=2.000 seconds
    Fit ARIMA: order=(0, 1, 2) seasonal_order=(0, 1, 1, 12); AIC=nan, BIC=nan, Fit time=nan seconds
    Fit ARIMA: order=(2, 1, 2) seasonal_order=(0, 1, 1, 12); AIC=3068.701, BIC=3090.895, Fit time=0.523 seconds
    Fit ARIMA: order=(1, 1, 3) seasonal_order=(0, 1, 1, 12); AIC=3068.842, BIC=3091.036, Fit time=0.590 seconds
    Total fit time: 11.745 seconds

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