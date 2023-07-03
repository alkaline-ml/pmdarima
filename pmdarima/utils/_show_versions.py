"""
Utility methods to print system info for debugging

adapted from ``pandas.show_versions``
adapted from ``sklearn.show_versions``
"""
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

        def get_version(module):
            return module.__version__

        for modname in deps:
            try:
                if modname in sys.modules:
                    mod = sys.modules[modname]
                else:
                    mod = importlib.import_module(modname)
                ver = get_version(mod)
                deps_info[modname] = ver
            except ImportError:
                deps_info[modname] = None

    else:
        from importlib.metadata import PackageNotFoundError, version

        for modname in deps:
            # Gotta do a little correction for scikit
            normalized_modname: str
            if modname == "sklearn":
                normalized_modname = "scikit-learn"
            else:
                normalized_modname = modname

            try:
                deps_info[modname] = version(normalized_modname)
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
