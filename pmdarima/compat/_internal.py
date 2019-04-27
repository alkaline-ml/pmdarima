# -*- coding: utf-8 -*-

import platform

__all__ = [
    'get_pytest_mpl_threshold'
]


def get_pytest_mpl_threshold(threshold_dict):
    """Get pytest-mpl image comparison threshold based on ``platform.system()``

    Parameters
    ----------
    threshold_dict : dict,
        Dictionary of thresholds with platform.system() as keys.

    Returns
    -------
    threshold : int, float.
        The threshold use for image comparisons in pytest.
    """
    return threshold_dict[platform.system()]
