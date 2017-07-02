# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Setup the pyramid module

from __future__ import print_function, absolute_import, division
from distutils.command.clean import clean
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
PYPIDIST = '%s-arima' % DISTNAME
DESCRIPTION = "Python's forecast::auto.arima equivalent"

MAINTAINER = 'Taylor G. Smith'
MAINTAINER_GIT = 'tgsmith61591'
MAINTAINER_EMAIL = 'taylor.smith@alkaline-ml.com'
LICENSE = 'MIT'

# import restricted version
import pyramid
VERSION = pyramid.__version__

# get the installation requirements:
with open('requirements.txt') as req:
    REQUIREMENTS = req.read().split(os.linesep)

SETUPTOOLS_COMMANDS = {  # this is a set literal, not a dict
    'develop', 'release', 'bdist_egg', 'bdist_rpm',
    'bdist_wininst', 'install_egg_info', 'build_sphinx',
    'egg_info', 'easy_install', 'upload', 'bdist_wheel',
    '--single-version-externally-managed'
}

if SETUPTOOLS_COMMANDS.intersection(sys.argv):
    import setuptools

    extra_setuptools_args = dict(
        zip_safe=False,  # the package can run out of an .egg file
        include_package_data=True,
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
    metadata = dict(name=PYPIDIST,
                    packages=[DISTNAME],
                    url="https://github.com/%s/%s" % (MAINTAINER_GIT, DISTNAME),
                    maintainer=MAINTAINER,
                    maintainer_email=MAINTAINER_EMAIL,
                    description=DESCRIPTION,
                    license=LICENSE,
                    version=VERSION,
                    classifiers=['Intended Audience :: Science/Research',
                                 'Intended Audience :: Developers',
                                 'Programming Language :: C',
                                 'Programming Language :: Python',
                                 'Topic :: Software Development',
                                 'Topic :: Scientific/Engineering',
                                 'Operating System :: Microsoft :: Windows',
                                 'Operating System :: POSIX',
                                 'Operating System :: Unix',
                                 'Operating System :: MacOS',
                                 'Programming Language :: Python :: 2.7',
                                 'Programming Language :: Python :: 3.5',
                                 'Programming Language :: Python :: 3.6',
                                 ],
                    keywords='sklearn scikit-learn arima timeseries',
                    # this will only work for releases that have the appropriate tag...
                    download_url='https://github.com/%s/%s/archive/v%s.tar.gz' % (MAINTAINER_GIT, DISTNAME, VERSION),
                    # install_requires=REQUIREMENTS,
                    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*, <4',
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
        # changes its behavior... (https://github.com/scikit-learn/scikit-learn/issues/1016)
        try:
            from setuptools import setup
        except ImportError:
            from distutils.core import setup

        metadata['version'] = VERSION

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
