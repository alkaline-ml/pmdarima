# -*- coding: utf-8 -*-

from pmdarima.compat.pytest import pytest_error_str
from pmdarima.model_selection import train_test_split
from pmdarima.pipeline import Pipeline, _warn_for_deprecated
from pmdarima.preprocessing import BoxCoxEndogTransformer, FourierFeaturizer, \
    DateFeaturizer
from pmdarima.arima import ARIMA, AutoARIMA
from pmdarima.datasets import load_wineind
from pmdarima.datasets._base import load_date_example
import numpy as np

from numpy.testing import assert_array_almost_equal
import pytest

rs = np.random.RandomState(42)
wineind = load_wineind()
xreg = rs.rand(wineind.shape[0], 2)

train, test, x_train, x_test = train_test_split(
    wineind, xreg, train_size=125)

y_dates, X_dates = load_date_example()


class TestIllegal:

    def test_non_unique_names(self):
        # Will fail since the same name repeated twice
        with pytest.raises(ValueError) as ve:
            Pipeline([
                ("stage", BoxCoxEndogTransformer()),
                ("stage", ARIMA(order=(0, 0, 0)))
            ])

        assert "not unique" in pytest_error_str(ve)

    def test_names_in_params(self):
        # Will fail because 'steps' is a param of Pipeline
        with pytest.raises(ValueError) as ve:
            Pipeline([
                ("steps", BoxCoxEndogTransformer()),
                ("stage", ARIMA(order=(0, 0, 0)))
            ])

        assert "names conflict" in pytest_error_str(ve)

    def test_names_double_underscore(self):
        # Will fail since the "__" is reserved for parameter names
        with pytest.raises(ValueError) as ve:
            Pipeline([
                ("stage__1", BoxCoxEndogTransformer()),
                ("stage", ARIMA(order=(0, 0, 0)))
            ])

        assert "must not contain __" in pytest_error_str(ve)

    def test_non_transformer_in_steps(self):
        # Will fail since the first stage is not a transformer
        with pytest.raises(TypeError) as ve:
            Pipeline([
                ("stage1", (lambda *args, **kwargs: None)),  # Fail
                ("stage2", AutoARIMA())
            ])

        assert "instances of BaseTransformer" in pytest_error_str(ve)

    @pytest.mark.parametrize(
        'stages', [
            # Nothing BUT a transformer
            [("stage1", BoxCoxEndogTransformer())],

            # Two transformers
            [("stage1", BoxCoxEndogTransformer()),
             ("stage2", FourierFeaturizer(m=12))]
        ]
    )
    def test_bad_last_stage(self, stages):
        # Will fail since the last stage is not an estimator
        with pytest.raises(TypeError) as ve:
            Pipeline(stages)

        assert "Last step of Pipeline should be" in pytest_error_str(ve)


@pytest.mark.parametrize(
    'pipe,kwargs,expected', [
        pytest.param(
            Pipeline([
                ("boxcox", BoxCoxEndogTransformer()),
                ("arima", AutoARIMA())
            ]),
            {},
            {"boxcox": {}, "arima": {}}
        ),

        pytest.param(
            Pipeline([
                ("boxcox", BoxCoxEndogTransformer()),
                ("arima", AutoARIMA())
            ]),
            {"boxcox__lmdba1": 0.001},
            {"boxcox": {"lmdba1": 0.001}, "arima": {}}
        ),
    ]
)
def test_get_kwargs(pipe, kwargs, expected):
    # Test we get the kwargs we expect
    kw = pipe._get_kwargs(**kwargs)
    assert kw == expected

    # show we can convert steps to dict
    assert pipe.named_steps


def test_pipeline_behavior():
    pipeline = Pipeline([
        ("fourier", FourierFeaturizer(m=12)),
        ("boxcox", BoxCoxEndogTransformer()),
        ("arima", AutoARIMA(seasonal=False, stepwise=True,
                            suppress_warnings=True, d=1, max_p=2, max_q=0,
                            start_q=0, start_p=1,
                            maxiter=3, error_action='ignore'))
    ])

    # Quick assertions on indexing
    assert len(pipeline) == 3

    pipeline.fit(train)
    preds = pipeline.predict(5)
    assert preds.shape[0] == 5

    assert pipeline._final_estimator.model_.fit_with_exog_

    # Assert that when the n_periods kwarg is set manually and incorrectly for
    # the fourier transformer, we get a ValueError
    kwargs = {
        "fourier__n_periods": 10
    }

    with pytest.raises(ValueError) as ve:
        pipeline.predict(3, **kwargs)
    assert "'n_periods'" in pytest_error_str(ve)

    # Assert that we can update the model
    pipeline.update(test, maxiter=5)

    # And that the fourier transformer was updated properly...
    assert pipeline.steps_[0][1].n_ == wineind.shape[0]


@pytest.mark.parametrize('pipeline', [
    Pipeline([
        ("arma", ARIMA(order=(2, 0, 0)))
    ]),

    Pipeline([
        ("arima", ARIMA(order=(2, 1, 0)))
    ]),

    Pipeline([
        ("sarimax", ARIMA(order=(2, 1, 0), seasonal_order=(1, 0, 0, 12)))
    ]),

    Pipeline([
        ("fourier", FourierFeaturizer(m=12)),
        ("arma", ARIMA(order=(2, 0, 0)))
    ]),

    Pipeline([
        ("fourier", FourierFeaturizer(m=12)),
        ("arima", ARIMA(order=(2, 1, 0)))
    ]),

    # one with a boxcox transformer
    Pipeline([
        ("boxcox", BoxCoxEndogTransformer()),
        ("fourier", FourierFeaturizer(m=12)),
        ("arima", AutoARIMA(seasonal=False, stepwise=True,
                            suppress_warnings=True, d=1, max_p=2, max_q=0,
                            start_q=0, start_p=1,
                            maxiter=3, error_action='ignore'))
    ]),
])
@pytest.mark.parametrize('X', [(None, None), (x_train, x_test)])
@pytest.mark.parametrize('inverse_transform', [True, False])
@pytest.mark.parametrize('return_conf_ints', [True, False])
def test_pipeline_predict_inverse_transform(pipeline, X, inverse_transform,
                                            return_conf_ints):
    X_train, X_test = X

    pipeline.fit(train, X=X_train)

    # show we can get a summary
    pipeline.summary()

    # first predict
    predictions = pipeline.predict(
        n_periods=test.shape[0],
        X=X_test,
        inverse_transform=inverse_transform,
        return_conf_int=return_conf_ints)

    if return_conf_ints:
        assert isinstance(predictions, tuple) and len(predictions) == 2
        y_pred, conf_ints = predictions
        assert conf_ints.shape[1] == 2
        assert np.all(
            (conf_ints[:, 0] <= y_pred) & (y_pred <= conf_ints[:, 1])
        )

    # now in sample
    in_sample = pipeline.predict_in_sample(
        X=X_train,
        inverse_transform=inverse_transform,
        return_conf_int=return_conf_ints)

    if return_conf_ints:
        assert isinstance(in_sample, tuple) and len(in_sample) == 2
        y_pred, conf_ints = predictions
        assert conf_ints.shape[1] == 2
        assert np.all(
            (conf_ints[:, 0] <= y_pred) & (y_pred <= conf_ints[:, 1])
        )


def test_deprecation_warning():
    kwargs = {'typ': 'foo'}
    with pytest.warns(DeprecationWarning) as we:
        kwargs = _warn_for_deprecated(**kwargs)
    assert not kwargs
    assert we


def test_order_does_not_matter_with_date_transformer():
    train_y_dates, test_y_dates, train_X_dates, test_X_dates = \
        train_test_split(y_dates, X_dates, test_size=15)

    pipeline_a = Pipeline([
        ('fourier', FourierFeaturizer(m=3, prefix="FOURIER")),
        ('dates', DateFeaturizer(column_name="date", prefix="DATE")),
        ("arima", AutoARIMA(seasonal=False, stepwise=True,
                            suppress_warnings=True,
                            maxiter=3, error_action='ignore'))
    ]).fit(train_y_dates, train_X_dates)
    Xt_a = pipeline_a.transform(X=test_X_dates)
    pred_a = pipeline_a.predict(X=test_X_dates)

    pipeline_b = Pipeline([
        ('dates', DateFeaturizer(column_name="date", prefix="DATE")),
        ('fourier', FourierFeaturizer(m=3, prefix="FOURIER")),
        ("arima", AutoARIMA(seasonal=False, stepwise=True,
                            suppress_warnings=True,
                            maxiter=3, error_action='ignore'))
    ]).fit(train_y_dates, train_X_dates)
    Xt_b = pipeline_b.transform(X=test_X_dates)
    pred_b = pipeline_b.predict(X=test_X_dates)

    # dates in A should differ from those in B
    assert pipeline_a.x_feats_[0].startswith("FOURIER")
    assert pipeline_a.x_feats_[-1].startswith("DATE")

    assert pipeline_b.x_feats_[0].startswith("DATE")
    assert pipeline_b.x_feats_[-1].startswith("FOURIER")

    # columns should be identical once ordered appropriately
    assert Xt_a.equals(Xt_b[pipeline_a.x_feats_])

    # forecasts should be identical
    assert_array_almost_equal(pred_a, pred_b, decimal=3)
