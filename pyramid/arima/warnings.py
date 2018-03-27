# -*- coding: utf-8 -*-

from __future__ import absolute_import

__all__ = [
    'ModelFitWarning'
]


class ModelFitWarning(UserWarning):
    """Generic warning used for a model fit that might fail. More descriptive
    than simply trying to lump everything into a default UserWarning, which
    gives the user no insight into the reason for the warning apart from a
    (potentially) cryptic message. This allows the user to understand the
    warning emanates from an attempted model fit and originates from within
    the Pyramid package.
    """
    pass
