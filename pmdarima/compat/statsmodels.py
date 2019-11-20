# -*- coding: utf-8 -*-
#
# Handle inconsistencies in the statsmodels API versions

from __future__ import absolute_import

__all__ = [
    'bind_df_model'
]


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
