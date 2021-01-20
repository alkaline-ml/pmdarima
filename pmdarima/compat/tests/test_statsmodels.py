# -*- coding: utf-8 -*-

from pmdarima.compat.statsmodels import bind_df_model, check_seasonal_order


# Test binding the degrees of freedom to a class in place. It's hard to test
# on a potentially non-existent version of statsmodels, so we have to mock the
# class
def test_bind_df_model():
    class ModelFit(object):
        k_exog = 2
        k_trend = 1
        k_ar = 3
        k_ma = 2
        k_seasonal_ar = 1
        k_seasonal_ma = 2

    class ARIMAResults(object):
        pass

    fit = ModelFit()
    res = ARIMAResults()

    # First, there is no 'df_model' in arima res
    assert not hasattr(res, 'df_model')
    bind_df_model(fit, res)

    # Now it should
    assert hasattr(res, 'df_model')
    assert res.df_model == 11, res.df_model


def test_check_seasonal_order():
    # issue370, using an iterable at position 0 returns
    order = ([1, 2, 3, 52], 0, 1, 7)
    checked_order = check_seasonal_order(order)
    assert order == checked_order

    # Special case where we override the seasonal order that is passed in.
    order = (0, 0, 0, 1)
    checked_order = check_seasonal_order(order)
    assert checked_order == (0, 0, 0, 0)
