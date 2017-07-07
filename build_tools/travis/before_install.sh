#!/usr/bin/env bash

set -e

# if it's a linux build, we need to apt-get update
if [[ "$TRAVIS_OS_NAME" == "linux" ]]; then
  echo "Updating apt-get for Linux build"
  sudo apt-get -qq update

# if it's a mac build, we need to brew update, and then we need to
# install python (it is not always native on older mac platforms)
else
  # homebrew should come with the xcode image
  sudo brew update

  # install some form of python and let miniconda take care of the rest...
  if [[ "$PYTHON_VERSION" == "2.7" ]]; then
    sudo brew install python
  else
    sudo brew install python3
  fi

  # get pip via curl
  curl -O http://python-distribute.org/distribute_setup.py
  python distribute_setup.py
  curl -O http://raw.github.com/pypa/pip/master/contrib/get-pip.py
  python get-pip.py

  # install virtualenv (don't activate) in case we ever use more than just conda testing
  pip install virtualenv

  # we SHOULD have git... but make sure brew has it installed so we can clone the branch
  brew install git
fi
