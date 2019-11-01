# -*- coding: utf-8 -*-

from .utils import check_endog
import numpy as np

__all__ = ['smape']


def smape(y_true, y_pred):
    r"""Compute the Symmetric Mean Absolute Percentage Error.

    The symmetric mean absolute percentage error (SMAPE) is an accuracy measure
    based on percentage (or relative) errors. Defined as follows:

        :math:`\frac{100\%}{n}\sum_{t=1}^{n}{\frac{|F_{t}-A_{t}|}{
        (|A_{t}|+|F_{t}|)/2}}`

    Where a perfect SMAPE score is 0.0, and a higher score indicates a higher
    error rate.

    Parameters
    ----------
    y_true : array-like, shape=(n_samples,)
        The true test values of y.

    y_pred : array-like, shape=(n_samples,)
        The forecasted values of y.

    Examples
    --------
    A typical case:
    >>> import numpy as np
    >>> y_true = np.array([0.07533, 0.07533, 0.07533, 0.07533,
    ...                    0.07533, 0.07533, 0.0672, 0.0672])
    >>> y_pred = np.array([0.102, 0.107, 0.047, 0.1,
    ...                    0.032, 0.047, 0.108, 0.089])
    >>> smape(y_true, y_pred)
    42.60306631890196

    A perfect score:
    >>> smape(y_true, y_true)
    0.0

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error  # noqa: E501
    """
    y_true = check_endog(y_true)  # type: np.ndarray
    y_pred = check_endog(y_pred)  # type: np.ndarray
    abs_diff = np.abs(y_pred - y_true)
    return np.mean((abs_diff * 200 / (np.abs(y_pred) + np.abs(y_true))))
