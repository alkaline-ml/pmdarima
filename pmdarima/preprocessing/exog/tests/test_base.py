# -*- coding: utf-8 -*-

from pmdarima.preprocessing.exog import base
from pmdarima import datasets
import numpy as np
import pandas as pd

wineind = datasets.load_wineind()


class RandomExogFeaturizer(base.BaseExogFeaturizer):
    """Creates random exog features. This is just used to test base func"""

    def _get_prefix(self):
        return "RND"

    def fit(self, y, exogenous):
        return self

    def transform(self, y, exogenous=None, n_periods=0, **_):
        exog = np.random.rand(y.shape[0], 4)
        exog = self._safe_hstack(exogenous, exog)
        return y, exog


def test_default_get_feature_names():
    feat = RandomExogFeaturizer()
    y_trans, exog = feat.fit_transform(wineind)
    assert y_trans is wineind
    assert exog.columns.tolist() == \
        ['RND_0', 'RND_1', 'RND_2', 'RND_3']


def test_default_get_feature_names_with_exog():
    feat = RandomExogFeaturizer()
    exog = pd.DataFrame.from_records(np.random.rand(wineind.shape[0], 2),
                                     columns=['a', 'b'])
    y_trans, exog_trans = feat.fit_transform(wineind, exog)
    assert y_trans is wineind
    assert exog_trans.columns.tolist() == \
        ['a', 'b', 'RND_0', 'RND_1', 'RND_2', 'RND_3']
