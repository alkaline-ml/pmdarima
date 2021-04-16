# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Setup the pmdarima module. This is heavily adapted from the scikit-learn
# setup.py, since they have a similar package structure and build similar
# Cython + C modules.

import sys
import os
import platform
import shutil

from distutils.command.clean import clean as Clean
from pkg_resources import parse_version
import traceback
import importlib

import builtins

# Minimum allowed version
MIN_PYTHON = (3, 6)

# Hacky (!!), adopted from sklearn. This sets a global variable
# so pmdarima __init__ can detect if it's being loaded in the setup
# routine, so it won't load submodules that haven't yet been built.
# This is because of the numpy distutils extensions that are used by pmdarima
# to build the compiled extensions in sub-packages
builtins.__PMDARIMA_SETUP__ = True

# metadata
DISTNAME = 'pmdarima'
PYPIDIST = 'pmdarima'
GIT_REPO_NAME = 'pmdarima'
DESCRIPTION = "Python's forecast::auto.arima equivalent"

# Get the long desc
with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

MAINTAINER = 'Taylor G. Smith'
MAINTAINER_GIT = 'tgsmith61591'
MAINTAINER_EMAIL = 'taylor.smith@alkaline-ml.com'
LICENSE = 'MIT'
URL = 'http://alkaline-ml.com/pmdarima'
DOWNLOAD_URL = 'https://pypi.org/project/pmdarima/#files'
PROJECT_URLS = {
    'Bug Tracker': 'https://github.com/alkaline-ml/pmdarima/issues',
    'Documentation': URL,
    'Source Code': 'https://github.com/alkaline-ml/pmdarima'
}

# import restricted version of pmdarima that does not need the compiled code
import pmdarima
VERSION = pmdarima.__version__  # will be 0.0.0 unless tagging

# get the installation requirements:
with open('requirements.txt') as req:
    REQUIREMENTS = [l for l in req.read().split(os.linesep) if l]
    print(f"Requirements: {REQUIREMENTS}")

# Optional setuptools features
SETUPTOOLS_COMMANDS = {  # this is a set literal, not a dict
    'develop', 'release', 'bdist_egg', 'bdist_rpm',
    'bdist_wininst', 'install_egg_info', 'build_sphinx',
    'egg_info', 'easy_install', 'upload', 'bdist_wheel',
    '--single-version-externally-managed',

    # scikit does NOT do this:
    'sdist,'
}

if SETUPTOOLS_COMMANDS.intersection(sys.argv):
    # we don't use setuptools, but if we don't import it, the "develop"
    # option for setup.py is invalid.
    import setuptools  # noqa

    print('Adding extra setuptools args')
    extra_setuptools_args = dict(
        zip_safe=False,  # the package can run out of an .egg file
        include_package_data=True,
        data_files=[
            ('pmdarima', ['pmdarima/VERSION']),
        ],
        # scikit does this:
        # extras_require={
        #     'alldeps': REQUIREMENTS
        # }
    )
else:
    extra_setuptools_args = dict()


# Custom clean command to remove build artifacts -- adopted from sklearn
class CleanCommand(Clean):
    description = "Remove build artifacts from the source tree"

    # this is mostly in case we ever add a Cython module to SMRT
    def run(self):
        Clean.run(self)
        # Remove c files if we are not within a sdist package
        cwd = os.path.abspath(os.path.dirname(__file__))
        remove_c_files = not os.path.exists(os.path.join(cwd, 'PKG-INFO'))
        if remove_c_files:
            print('Will remove generated .c files')
        if os.path.exists('build'):
            shutil.rmtree('build')
        for dirpath, dirnames, filenames in os.walk(DISTNAME):
            for filename in filenames:
                if any(filename.endswith(suffix) for suffix in
                       (".so", ".pyd", ".dll", ".pyc")):
                    print(f'Removing file: {filename}')
                    os.unlink(os.path.join(dirpath, filename))
                    continue
                extension = os.path.splitext(filename)[1]
                if remove_c_files and extension in ['.c', '.cpp']:
                    pyx_file = str.replace(filename, extension, '.pyx')
                    if os.path.exists(os.path.join(dirpath, pyx_file)):
                        os.unlink(os.path.join(dirpath, filename))
            for dirname in dirnames:
                if dirname == '__pycache__':
                    absdir = os.path.join(dirpath, dirname)
                    print(f'Removing directory: {absdir}')
                    shutil.rmtree(absdir)


cmdclass = {'clean': CleanCommand}


# build_ext has to be imported after setuptools
try:
    import numpy as np
    from numpy.distutils.command.build_ext import build_ext  # noqa

    # This is the preferred way to check numpy version: https://git.io/JtEIb
    if sys.platform == 'darwin' and np.lib.NumpyVersion(np.__version__) >= '1.20.0':
        # https://numpy.org/devdocs/user/building.html#disabling-atlas-and-other-accelerated-libraries
        os.environ['NPY_BLAS_ORDER'] = ''
        os.environ['NPY_LAPACK_ORDER'] = ''

    class build_ext_subclass(build_ext):
        def build_extensions(self):
            build_ext.build_extensions(self)

    cmdclass['build_ext'] = build_ext_subclass

except ImportError:
    # Numpy should not be a dependency just to be able to introspect
    # that python 3.X is required.
    pass

# Here is where scikit configures the wheelhouse uploader, but we don't deal
# with that currently. Maybe in the future...


def configuration(parent_package='', top_path=None):
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    # we know numpy is a valid import now
    from numpy.distutils.misc_util import Configuration
    from pmdarima._build_utils import _check_cython_version

    config = Configuration(None, parent_package, top_path)

    # Avoid non-useful msg
    # "Ignoring attempt to set 'name' (from ... "
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    # Cython is required by config.add_subpackage, so check that we have the
    # correct version of Cython
    _check_cython_version()

    config.add_subpackage(DISTNAME)
    return config


def check_package_status(package, min_version):
    """
    Returns a dictionary containing a boolean specifying whether given package
    is up-to-date, along with the version string (empty string if
    not installed).
    """
    package_status = {}
    try:
        module = importlib.import_module(package)
        package_version = module.__version__
        package_status['up_to_date'] = parse_version(
            package_version) >= parse_version(min_version)
        package_status['version'] = package_version
    except ImportError:
        traceback.print_exc()
        package_status['up_to_date'] = False
        package_status['version'] = ""

    req_str = "pmdarima requires {} >= {}.\n".format(
        package, min_version)

    if package_status['up_to_date'] is False:
        if package_status['version']:
            raise ImportError("Your installation of {} "
                              "{} is out-of-date.\n{}"
                              .format(package, package_status['version'],
                                      req_str))
        else:
            raise ImportError("{} is not "
                              "installed.\n{}"
                              .format(package, req_str))


def do_setup():
    # setup the config
    metadata = dict(name=PYPIDIST,
                    # packages=[DISTNAME],
                    maintainer=MAINTAINER,
                    maintainer_email=MAINTAINER_EMAIL,
                    description=DESCRIPTION,
                    license=LICENSE,
                    url=URL,
                    download_url=DOWNLOAD_URL,
                    project_urls=PROJECT_URLS,
                    version=VERSION,
                    long_description=LONG_DESCRIPTION,
                    long_description_content_type="text/markdown",
                    classifiers=[
                        'Intended Audience :: Science/Research',
                        'Intended Audience :: Developers',
                        'Intended Audience :: Financial and Insurance Industry',
                        'Programming Language :: C',
                        'Programming Language :: Python',
                        'Topic :: Software Development',
                        'Topic :: Scientific/Engineering',
                        'Operating System :: Microsoft :: Windows',
                        'Operating System :: POSIX',
                        'Operating System :: Unix',
                        'Operating System :: MacOS',
                        'Programming Language :: Python :: 3',
                        'Programming Language :: Python :: 3.6',
                        'Programming Language :: Python :: 3.7',
                        'Programming Language :: Python :: 3.8',
                        'Programming Language :: Python :: 3.9',
                        ('Programming Language :: Python :: '
                         'Implementation :: CPython'),
                    ],
                    cmdclass=cmdclass,
                    python_requires=f'>={MIN_PYTHON[0]}.{MIN_PYTHON[1]}',
                    install_requires=REQUIREMENTS,
                    # We have a MANIFEST.in, so I'm not convinced this is fully
                    # necessary, but better to be safe since we've had sdist
                    # problems in the past...
                    package_data=dict(
                        DISTNAME=[
                            '*',
                            'pmdarima/*',
                            'pmdarima/VERSION',
                            'pmdarima/__check_build/*',
                            'pmdarima/_build_utils/*',
                            'pmdarima/arima/*',
                            'pmdarima/arima/tests/data/*',
                            'pmdarima/compat/*',
                            'pmdarima/datasets/*',
                            'pmdarima/datasets/data/*',
                            'pmdarima/model_selection/*',
                            'pmdarima/preprocessing/*',
                            'pmdarima/preprocessing/endog/*',
                            'pmdarima/preprocessing/exog/*',
                            'pmdarima/tests/*',
                            'pmdarima/utils/*',
                        ]
                    ),
                    keywords='arima timeseries forecasting pyramid pmdarima '
                             'pyramid-arima scikit-learn statsmodels',
                    **extra_setuptools_args)

    if len(sys.argv) == 1 or (
            len(sys.argv) >= 2 and ('--help' in sys.argv[1:] or
                                    sys.argv[1] in ('--help-commands',
                                                    'egg_info',
                                                    'dist_info',
                                                    '--version',
                                                    'clean'))):
        # For these actions, NumPy is not required, so we can import the
        # setuptools module. However this is not preferable... the moment
        # setuptools is imported, it monkey-patches distutils' setup and
        # changes its behavior...
        # (https://github.com/scikit-learn/scikit-learn/issues/1016)
        #
        # This is called when installing from pip and numpy might not
        # be on the system yet
        try:
            from setuptools import setup
            print("Setting up with setuptools")
        except ImportError:
            print("Setting up with distutils")
            from distutils.core import setup

        metadata['version'] = VERSION

    else:
        if sys.version_info < MIN_PYTHON:
            raise RuntimeError(
                # Don't make this an F-string so that the error can be rendered
                # on older versions of python
                "pmdarima requires Python {0}.{1} or later. The current "
                "Python version is {2} installed in {3}.".format(
                    MIN_PYTHON[0], MIN_PYTHON[1],
                    platform.python_version(),
                    sys.executable
                )
            )

        # for sdist, use setuptools so we get the long_description_content_type
        if 'sdist' in sys.argv:
            from setuptools import setup
            print("Setting up with setuptools")
        else:
            # we should only need numpy for building. Everything else can be
            # collected via install_requires above
            check_package_status('numpy', '1.16')

            from numpy.distutils.core import setup
            print("Setting up with numpy.distutils.core")

        # add the config to the metadata
        metadata['configuration'] = configuration

    # call setup on the dict
    setup(**metadata)


if __name__ == '__main__':
    do_setup()
