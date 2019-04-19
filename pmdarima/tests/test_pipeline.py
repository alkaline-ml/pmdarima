# -*- coding: utf-8 -*-

from pmdarima.pipeline import Pipeline
from pmdarima.preprocessing import BoxCoxEndogTransformer, FourierFeaturizer
from pmdarima.arima import ARIMA, AutoARIMA
from pmdarima.datasets import load_wineind

import pytest


class TestIllegal:

    def test_non_unique_names(self):
        # Will fail since the same name repeated twice
        with pytest.raises(ValueError) as ve:
            Pipeline([
                ("stage", BoxCoxEndogTransformer()),
                ("stage", ARIMA(order=(0, 0, 0)))
            ])

        assert "not unique" in str(ve)

    def test_names_in_params(self):
        # Will fail because 'steps' is a param of Pipeline
        with pytest.raises(ValueError) as ve:
            Pipeline([
                ("steps", BoxCoxEndogTransformer()),
                ("stage", ARIMA(order=(0, 0, 0)))
            ])

        assert "names conflict" in str(ve)

    def test_names_double_underscore(self):
        # Will fail since the "__" is reserved for parameter names
        with pytest.raises(ValueError) as ve:
            Pipeline([
                ("stage__1", BoxCoxEndogTransformer()),
                ("stage", ARIMA(order=(0, 0, 0)))
            ])

        assert "must not contain __" in str(ve)

    def test_non_transformer_in_steps(self):
        # Will fail since the first stage is not a transformer
        with pytest.raises(TypeError) as ve:
            Pipeline([
                ("stage1", (lambda *args, **kwargs: None)),  # Fail
                ("stage2", AutoARIMA())
            ])

        assert "instances of BaseTransformer" in str(ve)

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

        assert "Last step of Pipeline should be" in str(ve)


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


def test_pipeline_behavior():
    wineind = load_wineind()
    train, test = wineind[:125], wineind[125:]

    pipeline = Pipeline([
        ("fourier", FourierFeaturizer(m=12)),
        ("arima", AutoARIMA(seasonal=False, stepwise=True,
                            suppress_warnings=True,
                            maxiter=3, error_action='ignore'))
    ])

    # Quick assertions on indexing
    assert len(pipeline) == 2

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
    assert ("'n_periods'" in str(ve))

    # Assert that we can update the model
    pipeline.update(test, maxiter=5)

    # And that the fourier transformer was updated properly...
    assert pipeline.steps_[0][1].n_ == wineind.shape[0]
