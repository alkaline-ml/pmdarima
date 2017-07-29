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
  # depending on the python version, get a different one
  if [[ "$PYTHON_VERSION" == "2.7" ]]; then
    PYTHON_VERSION="2.7.13"
  else
    PYTHON_VERSION="3.5.4"
  fi

  # curl the URL
  PYTHON_NAME="Python-${PYTHON_VERSION}rc1-macosx10.6"
  curl https://www.python.org/ftp/python/${PYTHON_VERSION}/${PYTHON_NAME}.pkg > ${PYTHON_NAME}.pkg

  # unpack
  ls -la ${HOME}
  echo "Changing permissions to 777"
  sudo chmod 777 ${HOME}/${PYTHON_NAME}.pkg
  echo "Installing from pkg"
  sudo installer -pkg ${HOME}/${PYTHON_NAME}.pkg

  # add python to the path
  export PATH=${HOME}/bin/python:${PATH}
  python --version || echo "Python not setup properly!"

  # get pip via curl
  echo "Getting 'distribute' python package (pip dependency)"
  curl -O http://python-distribute.org/distribute_setup.py
  python distribute_setup.py
  curl -O http://raw.github.com/pypa/pip/master/contrib/get-pip.py
  python get-pip.py

  # install virtualenv (don't activate) in case we ever use more than just conda testing
  pip install virtualenv
fi
