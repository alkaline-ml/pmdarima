# -*- coding: utf-8 -*-

from sklearn.externals import six

import abc

from ..base import BaseTransformer


class BaseExogTransformer(six.with_metaclass(abc.ABCMeta, BaseTransformer)):
    """A base class for exogenous array transformers"""

    def _check_y_exog(self, y, exog, null_allowed=False):
        """Check the endog and exog arrays"""
        y, exog = super(BaseExogTransformer, self)._check_y_exog(y, exog)
        if exog is None and not null_allowed:
            raise ValueError("exog must be non-None for exogenous "
                             "transformers")
        return y, exog
