#!/usr/bin/env bash

set -e

# if it's a linux build, we need to apt-get update
if [[ "$TRAVIS_OS_NAME" == "linux" ]]; then
  echo "Updating apt-get for Linux build"
  sudo apt-get -qq update

# if it's a mac build, we need to install python from python.org
# (it is not always native on older mac platforms). We want to avoid
# homebrew, since it's ONLY for 64-bit distros and will not build a wheel
# with the 10_6 platform.
else
  # for debugging. newer versions of travis might come with python installed...
  which python || "Python is not yet installed"

  # depending on the python version, get a different version pkg
  if [[ "$PYTHON_VERSION" == "2.7" ]]; then
    PYTHON_VERSION="2.7.13"
    PYTHON_NAME="python-${PYTHON_VERSION}-macosx10.6"
  else
    PYTHON_VERSION="3.5.4"
    # 3.5.4 only ever had a release candidate, never a final, formal release
    PYTHON_NAME="python-${PYTHON_VERSION}rc1-macosx10.6"
  fi

  # curl the pkg from python.org
  cd ..  # go outside of the pyramid directory
  echo "Downloading python ${PYTHON_VERSION}"
  curl https://www.python.org/ftp/python/${PYTHON_VERSION}/${PYTHON_NAME}.pkg > ${PYTHON_NAME}.pkg

  # for some reason the installer complains about the package path being off?
  test -f ${PYTHON_NAME}.pkg || echo "${PYTHON_NAME}.pkg does not exist"
  echo "Changing permissions to 777"
  sudo chmod 777 ${PYTHON_NAME}.pkg

  # install using the installer (following is relevant documentation from `man installer`)
  echo "Installing from pkg"

  # The installer command is used to install Mac OS X installer packages to a specified domain or volume.
  # The installer command installs a single package per invocation, which is specified with the -package
  # parameter ( -pkg is accepted as a synonym).  It may be either a single package or a metapackage.
  # In the case of the metapackage, the packages which are part of the default install will be installed
  # unless disqualified by a package's check tool(s).
  #
  # The target volume is specified with the -target parameter ( -tgt is accepted as a synonym).  It must already
  # be mounted when the installer command is invoked.
  #
  # The installer command requires root privileges to run.  If a package requires authentication (set in a
  # package's .info file) the installer must be either run as root or with the sudo(8) command (but see further
  # discussion under the -store option).
  sudo installer -package ${PYTHON_NAME}.pkg -target /
  which python || "Package is installed, but python is still not on the PATH"

  # add python to the path
  export PATH=/usr/bin/python:${PATH}
  python --version || echo "Python not setup properly!"
  which python

  # get pip via curl
  echo "Getting 'distribute' python package (pip dependency)"
  curl -O http://python-distribute.org/distribute_setup.py
  python distribute_setup.py
  curl -O http://raw.github.com/pypa/pip/master/contrib/get-pip.py
  python get-pip.py

  # upgrade pip
  pip install --upgrade pip || "pip is already current"

  # install virtualenv (don't activate) in case we ever use more than just conda testing
  pip install virtualenv

  # see what platform python we're running (should be 10.6)
  python -c 'from distutils.util import get_platform; print(get_platform())'

  # go back into the pyramid directory
  cd pyramid
fi
