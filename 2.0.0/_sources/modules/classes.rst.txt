.. _api_ref:

=============
API Reference
=============

.. include:: ../includes/api_css.rst

This is the class and function reference for ``pmdarima``. Please refer to
the :ref:`full user guide <user_guide>` for further details, as the class and
function raw specifications may not be enough to give full guidelines on their
uses.


.. _arima_ref:

:mod:`pmdarima.arima`: ARIMA estimator & differencing tests
===========================================================

The ``pmdarima.arima`` sub-module defines the ``ARIMA`` estimator and the
``auto_arima`` function, as well as a set of tests of seasonality and
stationarity.

.. automodule:: pmdarima.arima
    :no-members:
    :no-inherited-members:

ARIMA estimator & statistical tests
-----------------------------------

**User guide:** See the :ref:`seasonality` and :ref:`enforcing_stationarity`
sections for further details.

.. currentmodule:: pmdarima

.. autosummary::
    :toctree: generated/
    :template: class.rst

    arima.ADFTest
    arima.ARIMA
    arima.AutoARIMA
    arima.CHTest
    arima.KPSSTest
    arima.OCSBTest
    arima.PPTest
    arima.StepwiseContext

ARIMA auto-parameter selection
------------------------------

**User guide:** See the :ref:`tips_and_tricks` section for further details.

.. currentmodule:: pmdarima

.. autosummary::
    :toctree: generated/
    :template: function.rst

    arima.auto_arima


Differencing helpers
--------------------

.. currentmodule:: pmdarima

.. autosummary::
    :toctree: generated/
    :template: function.rst

    arima.is_constant
    arima.ndiffs
    arima.nsdiffs


Seasonal decomposition
----------------------

.. currentmodule:: pmdarima

.. autosummary::
    :toctree: generated/
    :template: function.rst

    arima.decompose


.. _datasets_ref:

:mod:`pmdarima.datasets`: Toy timeseries datasets
=================================================

The ``pmdarima.datasets`` submodule provides several different univariate time-
series datasets used in various examples and tests across the package. If you
would like to prototype a model, this is a good place to find easy-to-access data.

**User guide:** See the :ref:`datasets` section for further details.

.. automodule:: pmdarima.datasets
    :no-members:
    :no-inherited-members:

Dataset loading functions
-------------------------

.. currentmodule:: pmdarima

.. autosummary::
    :toctree: generated/
    :template: function.rst

    datasets.load_airpassengers
    datasets.load_ausbeer
    datasets.load_austres
    datasets.load_gasoline
    datasets.load_heartrate
    datasets.load_lynx
    datasets.load_msft
    datasets.load_sunspots
    datasets.load_taylor
    datasets.load_wineind
    datasets.load_woolyrnq


.. _metrics_ref:

:mod:`pmdarima.metrics`: Time-series metrics
============================================

The ``metrics`` submodule implements time-series metrics that are not
implemented in scikit-learn.

.. automodule:: pmdarima.metrics
    :no-members:
    :no-inherited-members:

Metrics
-------

.. currentmodule:: pmdarima

.. autosummary::
    :toctree: generated/
    :template: function.rst

    metrics.smape


.. _model_selection_ref:

:mod:`pmdarima.model_selection`: Cross-validation classes
=========================================================

The ``pmdarima.model_selection`` submodule defines several different strategies
for cross-validating time series models

.. automodule:: pmdarima.model_selection
    :no-members:
    :no-inherited-members:

Cross validation & split utilities
----------------------------------

.. currentmodule:: pmdarima

.. autosummary::
    :toctree: generated/
    :template: class.rst

    model_selection.RollingForecastCV
    model_selection.SlidingWindowForecastCV

.. autosummary::
    :toctree: generated/
    :template: function.rst

    model_selection.check_cv
    model_selection.cross_validate
    model_selection.cross_val_predict
    model_selection.cross_val_score
    model_selection.train_test_split


.. _pipeline_ref:

:mod:`pmdarima.pipeline`: Pipelining transformers & ARIMAs
==========================================================

With the ``pipeline.Pipeline`` class, we can pipeline transformers together and
into a final ARIMA stage.

.. automodule:: pmdarima.pipeline
    :no-members:
    :no-inherited-members:

Pipelines
---------

.. currentmodule:: pmdarima

.. autosummary::
    :toctree: generated/
    :template: class.rst

    pipeline.Pipeline


.. _preprocessing_ref:

:mod:`pmdarima.preprocessing`: Preprocessing transformers
=========================================================

The ``pmdarima.preprocessing`` submodule provides a number of transformer
classes for pre-processing time series or exogenous arrays.

.. automodule:: pmdarima.preprocessing
    :no-members:
    :no-inherited-members:

Endogenous transformers
-----------------------

.. currentmodule:: pmdarima

.. autosummary::
    :toctree: generated/
    :template: class.rst

    preprocessing.BoxCoxEndogTransformer
    preprocessing.LogEndogTransformer

Exogenous transformers
----------------------

.. currentmodule:: pmdarima

.. autosummary::
    :toctree: generated/
    :template: class.rst

    preprocessing.DateFeaturizer
    preprocessing.FourierFeaturizer


.. _utils_ref:

:mod:`pmdarima.utils`: Utilities
================================

Utilities and array differencing functions used commonly across the package.

.. automodule:: pmdarima.utils
    :no-members:
    :no-inherited-members:

Array helper functions & metaestimators
---------------------------------------

.. currentmodule:: pmdarima

.. autosummary::
    :toctree: generated/
    :template: function.rst

    utils.acf
    utils.as_series
    utils.c
    utils.check_endog
    utils.diff
    utils.diff_inv
    utils.if_has_delegate
    utils.is_iterable
    utils.pacf

Plotting utilities & wrappers
-----------------------------

.. currentmodule:: pmdarima

.. autosummary::
    :toctree: generated/
    :template: function.rst

    utils.autocorr_plot
    utils.decomposed_plot
    utils.plot_acf
    utils.plot_pacf
