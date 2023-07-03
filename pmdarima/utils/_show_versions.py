"""
Utility methods to print system info for debugging

adapted from ``pandas.show_versions``
adapted from ``sklearn.show_versions``
"""
import os
# License: BSD 3 clause

import platform
import sys

_pmdarima_deps = (
    "pip",
    "setuptools",
    "sklearn",
    "statsmodels",
    "numpy",
    "scipy",
    "Cython",
    "pandas",
    "joblib",
    "pmdarima",
)

# Packages that have a different import name than name on PyPI
_install_mapping = {
    "sklearn": "scikit-learn"
}


def _get_sys_info():
    """System information

    Return
    ------
    sys_info : dict
        system and Python version information

    """
    python = sys.version.replace('\n', ' ')

    blob = [
        ("python", python),
        ('executable', sys.executable),
        ("machine", platform.platform()),
    ]

    return dict(blob)


def _get_deps_info(deps=_pmdarima_deps):
    """Overview of the installed version of main dependencies

    Returns
    -------
    deps_info: dict
        version information on relevant Python libraries
    """
    deps_info = {}

    # TODO: We can get rid of this when we deprecate 3.7
    if sys.version_info.minor <= 7:
        import importlib

        # Needed if pip is imported before setuptools
        # https://github.com/pypa/setuptools/issues/3044#issuecomment-1024972548
        os.environ["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"

        for modname in deps:
            try:
                if modname in sys.modules:
                    mod = sys.modules[modname]
                else:
                    mod = importlib.import_module(modname)

                deps_info[modname] = mod.__version__
            except ImportError:
                deps_info[modname] = None

    else:
        from importlib.metadata import PackageNotFoundError, version

        for modname in deps:
            try:
                deps_info[modname] = version(_install_mapping.get(modname, modname))
            except PackageNotFoundError:
                deps_info[modname] = None

    return deps_info


def show_versions():
    """Print useful debugging information"""
    sys_info = _get_sys_info()
    deps_info = _get_deps_info()

    print('\nSystem:')
    for k, stat in sys_info.items():
        print("{k:>10}: {stat}".format(k=k, stat=stat))

    print('\nPython dependencies:')
    for k, stat in deps_info.items():
        print("{k:>11}: {stat}".format(k=k, stat=stat))
