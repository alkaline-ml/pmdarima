# -*- coding: utf-8 -*-
from .utils import check_endog
import numpy as np

__all__ = ['smape']

def smape(y_true, y_pred):
    r"""Compute the Symmetric Mean Absolute Percentage Error (SMAPE).

    SMAPE = (1/n) * sum( 2 * |y_pred - y_true| / (|y_pred| + |y_true|) ) * 100
    A perfect score is 0.0; higher values mean higher error.

    Parameters
    ----------
    y_true : array-like
        Actual values.
    y_pred : array-like
        Forecasted values.

    Returns
    -------
    float
        The SMAPE score.
    """
    # Convert inputs to NumPy arrays
    y_true = np.asarray(check_endog(y_true, copy=False, preserve_series=False))
    y_pred = np.asarray(check_endog(y_pred, copy=False, preserve_series=False))

    # Compute absolute differences
    abs_diff = np.abs(y_pred - y_true)
    denominator = np.abs(y_pred) + np.abs(y_true)

    # Avoid division by zero
    denominator_safe = np.where(denominator == 0, 1, denominator)

    # Calculate SMAPE
    smape_value = np.mean((abs_diff * 200) / denominator_safe)

    return smape_value
