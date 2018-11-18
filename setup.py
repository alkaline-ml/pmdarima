# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Setup the pmdarima module

from __future__ import print_function, absolute_import, division

from distutils.command.clean import clean
import shutil
import os
import sys

if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins

# Hacky (!!), adopted from sklearn. This sets a global variable
# so pmdarima __init__ can detect if it's being loaded in the setup
# routine, so it won't load submodules that haven't yet been built.
# This is because of the numpy distutils extensions that are used by pmdarima
# to build the compiled extensions in sub-packages
builtins.__PMDARIMA_SETUP__ = True

# metadata
DISTNAME = 'pmdarima'
PYPIDIST = 'pmdarima'
GIT_REPO_NAME = 'pyramid'  # TODO: eventually migrate
DESCRIPTION = "Python's forecast::auto.arima equivalent"

# Get the long desc
with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

MAINTAINER = 'Taylor G. Smith'
MAINTAINER_GIT = 'tgsmith61591'
MAINTAINER_EMAIL = 'taylor.smith@alkaline-ml.com'
LICENSE = 'MIT'

# import restricted version
import pmdarima
VERSION = pmdarima.__version__

# get the installation requirements:
with open('requirements.txt') as req:
    REQUIREMENTS = [l for l in req.read().split(os.linesep) if l]
    print("Requirements: %r" % REQUIREMENTS)

SETUPTOOLS_COMMANDS = {  # this is a set literal, not a dict
    'develop', 'release', 'bdist_egg', 'bdist_rpm',
    'bdist_wininst', 'install_egg_info', 'build_sphinx',
    'egg_info', 'easy_install', 'upload', 'bdist_wheel',
    '--single-version-externally-managed'
}

# are we building from install or develop?
we_be_buildin = 'install' in sys.argv
if SETUPTOOLS_COMMANDS.intersection(sys.argv):
    # we don't use setuptools, but if we don't import it, the "develop"
    # option for setup.py is invalid.
    import setuptools
    from setuptools.dist import Distribution

    class BinaryDistribution(Distribution):
        """The goal is to avoid having to later build the C code
        on the system itself.

        References
        ----------
        .. [1] How to avoid building a C library with my Python package:
               http://bit.ly/2vQkW47

        .. [2] https://github.com/spotify/dh-virtualenv/issues/113
        """
        def is_pure(self):
            """Since we are distributing binary (.so, .dll, .dylib) files for
            different platforms we need to make sure the wheel does not build
            without them! See 'Building Wheels':
            http://lucumr.pocoo.org/2014/1/27/python-on-wheels/

            Returns
            -------
            False
            """
            return False

        def has_ext_modules(self):
            """Pmdarima has external modules. Therefore, unsurprisingly, this
            returns True to indicate that there are, in fact, external modules.

            Returns
            -------
            True
            """
            return True

    # only import numpy (later) if we're developing
    if any(cmd in sys.argv for cmd in {'develop', 'bdist_wheel'}):
        we_be_buildin = True

    print('Adding extra setuptools args')
    extra_setuptools_args = dict(
        zip_safe=False,  # the package can run out of an .egg file
        include_package_data=True,
        package_data={'pmdarima': ['*']},
        distclass=BinaryDistribution,
        install_requires=REQUIREMENTS,
    )
else:
    extra_setuptools_args = dict()


# Custom clean command to remove build artifacts -- adopted from sklearn
class CleanCommand(clean):
    description = "Remove build artifacts from the source tree"

    # this is mostly in case we ever add a Cython module to SMRT
    def run(self):
        clean.run(self)
        # Remove c files if we are not within a sdist package
        cwd = os.path.abspath(os.path.dirname(__file__))
        remove_c_files = not os.path.exists(os.path.join(cwd, 'PKG-INFO'))
        if remove_c_files:
            print('Will remove generated .c & .so files')
        if os.path.exists('build'):
            shutil.rmtree('build')
        for dirpath, dirnames, filenames in os.walk(DISTNAME):
            for filename in filenames:
                if any(filename.endswith(suffix) for suffix in
                       (".so", ".pyd", ".dll", ".pyc")):
                    print('Removing file: %s' % filename)
                    os.unlink(os.path.join(dirpath, filename))
                    continue
                extension = os.path.splitext(filename)[1]
                if remove_c_files and extension in ['.c', '.cpp']:
                    pyx_file = str.replace(filename, extension, '.pyx')
                    if os.path.exists(os.path.join(dirpath, pyx_file)):
                        os.unlink(os.path.join(dirpath, filename))
            # this is for FORTRAN modules, which some of my other packages
            # have used in the past...
            for dirname in dirnames:
                if dirname == '__pycache__' or dirname.endswith('.so.dSYM'):
                    print('Removing directory: %s' % dirname)
                    shutil.rmtree(os.path.join(dirpath, dirname))


cmdclass = {'clean': CleanCommand}


def configuration(parent_package='', top_path=None):
    # we know numpy is a valid import now
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)

    # Avoid non-useful msg
    # "Ignoring attempt to set 'name' (from ... "
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage(DISTNAME)
    return config


def do_setup():
    # setup the config
    metadata = dict(name=PYPIDIST,
                    packages=[DISTNAME],
                    url="https://github.com/%s/%s" % (MAINTAINER_GIT,
                                                      GIT_REPO_NAME),
                    maintainer=MAINTAINER,
                    maintainer_email=MAINTAINER_EMAIL,
                    description=DESCRIPTION,
                    long_description=LONG_DESCRIPTION,
                    long_description_content_type="text/markdown",
                    license=LICENSE,
                    version=VERSION,
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
                        'Programming Language :: Python :: 3.5',
                        'Programming Language :: Python :: 3.6',
                    ],
                    keywords='arima timeseries forecasting pyramid pmdarima '
                             'pyramid-arima scikit-learn statsmodels',
                    # this will only work for releases that have the right tag
                    download_url='https://github.com/%s/%s/archive/v%s.tar.gz'
                                 % (MAINTAINER_GIT, GIT_REPO_NAME, VERSION),
                    python_requires='>=3.5',
                    cmdclass=cmdclass,
                    **extra_setuptools_args)

    if len(sys.argv) == 1 or (
            len(sys.argv) >= 2 and ('--help' in sys.argv[1:] or
                                    sys.argv[1] in ('--help-commands',
                                                    'egg-info',
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
        except ImportError:
            from distutils.core import setup

        metadata['version'] = VERSION

    else:
        # if we are building for install, develop or bdist_wheel, we NEED
        # numpy and cython, since they are both used in building the .pyx
        # files into C modules.
        if we_be_buildin:
            try:
                from numpy.distutils.core import setup
            except ImportError:
                raise RuntimeError('Need numpy to build pmdarima')

        # if we are building to or from a wheel, we do not need numpy,
        # because it will be handled in the requirements.txt
        else:
            from setuptools import setup

        # add the config to the metadata
        metadata['configuration'] = configuration

    # call setup on the dict
    setup(**metadata)


if __name__ == '__main__':
    do_setup()
