# -*- coding: utf-8 -*-
#
# Handle inconsistencies in the statsmodels API versions

from collections.abc import Iterable
from pkg_resources import parse_version
import statsmodels as sm

__all__ = [
    'bind_df_model'
]

_sm_version = sm.__version__


def bind_df_model(model_fit, arima_results):
    """Set model degrees of freedom.

    Older versions of statsmodels don't handle this issue. Sets the
    model degrees of freedom in place if not already present.

    Parameters
    ----------
    model_fit : ARMA, ARIMA or SARIMAX
        The fitted model.

    arima_results : ModelResultsWrapper
        The results wrapper.
    """
    if not hasattr(arima_results, 'df_model'):
        df_model = model_fit.k_exog + model_fit.k_trend + \
            model_fit.k_ar + model_fit.k_ma + \
            model_fit.k_seasonal_ar + model_fit.k_seasonal_ma
        setattr(arima_results, 'df_model', df_model)


def check_seasonal_order(order):
    """Check the seasonal order

    Statsmodels 0.11.0 introduced a check for seasonal order == 1 that can
    raise a ValueError, but some of our old defaults allow for m == 1 in an
    otherwise null seasonal order.

    Parameters
    ----------
    order : tuple
        The existing seasonal order
    """

    # If order[0] is an iterable, but not a string then we don't perform check.
    # Otherwise we perform the check and override order if it satisfies check.
    # See issue#370: https://github.com/alkaline-ml/pmdarima/issues/370
    if isinstance(order[0], Iterable) and not isinstance(order[0], str):
        return order
    else:
        if sum(order[:3]) == 0 and order[-1] == 1:
            order = (0, 0, 0, 0)

        # user's order may be invalid, but we'll let statsmodels' validation
        # handle that.
        return order


def _use_sm13():
    return parse_version(sm.__version__) >= parse_version("0.13.0")
