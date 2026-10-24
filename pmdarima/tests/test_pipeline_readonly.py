import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pmdarima.arima import ARIMA
from pmdarima.pipeline import Pipeline
from pmdarima.preprocessing import BoxCoxEndogTransformer


@pytest.mark.parametrize('method', ['predict', 'predict_in_sample'])
@pytest.mark.parametrize('writeable', [False, True])
def test_inverse_transform_owns_confidence_intervals(
        monkeypatch, method, writeable):
    pipeline = Pipeline([
        ('boxcox', BoxCoxEndogTransformer(lmbda=0.5)),
        ('arima', ARIMA(order=(0, 0, 0))),
    ]).fit(np.arange(1., 21.) ** 2)

    predictions = np.array([0., 2., 4.])
    intervals = np.array([[-1., 1.], [1., 3.], [3., 5.]])
    original_intervals = intervals.copy()
    intervals.flags.writeable = writeable
    monkeypatch.setattr(
        pipeline._final_estimator, method,
        lambda **kwargs: (predictions, intervals))

    kwargs = ({'n_periods': 3} if method == 'predict'
              else {'start': 0, 'end': 2})
    means, bounds = getattr(pipeline, method)(return_conf_int=True, **kwargs)

    # With lambda=1/2, the inverse is (y/2 + 1)**2.
    assert_array_equal(means, [1., 4., 9.])
    assert_array_equal(bounds, [[.25, 2.25], [2.25, 6.25], [6.25, 12.25]])
    assert_array_equal(predictions, [0., 2., 4.])
    assert_array_equal(intervals, original_intervals)
    assert intervals.flags.writeable == writeable
