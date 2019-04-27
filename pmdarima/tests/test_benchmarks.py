import pytest
# from pmdarima.arima.tests import test_arima, test_seasonality


# pytest runtime variables
ch_test_parameters = [pytest.param(365),
                      pytest.param(366)]


# class TestBenchMarkTop12SlowestTests():
#
#     def test_with_seasonality2(self, benchmark):
#         benchmark(test_arima.test_with_seasonality2)
#
#     def test_with_seasonality3(self, benchmark):
#         benchmark(test_arima.test_with_seasonality3)
#
#     def test_with_seasonality4(self, benchmark):
#         benchmark(test_arima.test_with_seasonality4)
#
#     def test_with_seasonality5(self, benchmark):
#         benchmark(test_arima.test_with_seasonality5)
#
#     def test_with_seasonality7(self, benchmark):
#         benchmark(test_arima.test_with_seasonality7)
#
#     def test_warn_for_large_differences(self, benchmark):
#         benchmark(test_arima.test_warn_for_large_differences)
#
#     def test_seasonal_xreg_differencing(self, benchmark):
#         benchmark(test_arima.test_seasonal_xreg_differencing)
#
#     def test_oob_for_issue_28(self, benchmark):
#         benchmark(test_arima.test_oob_for_issue_28)
#
#     def test_oob_for_issue_29(self, benchmark):
#         benchmark(test_arima.test_oob_for_issue_29)
#
#     def test_oob_sarimax(self, benchmark):
#         benchmark(test_arima.test_oob_sarimax)
#
#     @pytest.mark.parametrize('m', ch_test_parameters)
#     def test_ch_test_long(self, benchmark, m):
#         benchmark(test_seasonality.test_ch_test_long, m=m)
