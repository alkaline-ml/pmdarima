# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Setup the pyramid module

from __future__ import print_function, absolute_import, division
from distutils.command.clean import clean
# from setuptools import setup  # DO NOT use setuptools!!!!!!
import shutil
import os
import sys

if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins

# Hacky, adopted from sklearn. This sets a global variable
# so pyramid __init__ can detect if it's being loaded in the setup
# routine, so it won't load submodules that haven't yet been built.
builtins.__PYRAMID_SETUP__ = True

# metadata
DISTNAME = 'pyramid'
DESCRIPTION = "Python's auto.arima equivalent"

MAINTAINER = 'Taylor G. Smith'
MAINTAINER_EMAIL = 'taylor.smith@alkaline-ml.com'
LICENSE = 'MIT'

# import restricted version
import pyramid
VERSION = pyramid.__version__

# get the installation requirements:
with open('requirements.txt') as req:
    REQUIREMENTS = req.read().split(os.linesep)


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
            cython_hash_file = os.path.join(cwd, 'cythonize.dat')
            if os.path.exists(cython_hash_file):
                os.unlink(cython_hash_file)
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
            # this is for FORTRAN modules, which some of my other packages have used in the past...
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
    metadata = dict(name=DISTNAME,
                    maintainer=MAINTAINER,
                    maintainer_email=MAINTAINER_EMAIL,
                    description=DESCRIPTION,
                    license=LICENSE,
                    version=VERSION,
                    classifiers=['Intended Audience :: Science/Research',
                                 'Intended Audience :: Developers',
                                 'Intended Audience :: Scikit-learn users',
                                 'Programming Language :: Python',
                                 'Topic :: Machine Learning',
                                 'Topic :: Software Development',
                                 'Topic :: Scientific/Engineering',
                                 'Operating System :: Microsoft :: Windows',
                                 'Operating System :: POSIX',
                                 'Operating System :: Unix',
                                 'Operating System :: MacOS',
                                 'Programming Language :: Python :: 2.7'
                                 ],
                    keywords='sklearn scikit-learn arima timeseries',
                    # packages=[DISTNAME],
                    # install_requires=REQUIREMENTS,
                    cmdclass=cmdclass)

    if len(sys.argv) == 1 or (
            len(sys.argv) >= 2 and ('--help' in sys.argv[1:] or
                                    sys.argv[1] in ('--help-commands',
                                                    'egg-info',
                                                    '--version',
                                                    'clean'))):
        # For these actions, NumPy is not required
        try:
            from setuptools import setup
        except ImportError:
            from distutils.core import setup

    else:  # we DO need numpy
        try:
            from numpy.distutils.core import setup
        except ImportError:
            raise RuntimeError('Need numpy to build %s' % DISTNAME)

        # add the config to the metadata
        metadata['configuration'] = configuration

    # call setup on the dict
    setup(**metadata)


if __name__ == '__main__':
    do_setup()
