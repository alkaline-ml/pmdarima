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
  echo "Updating HomeBrew"
  brew update

  # install some form of python and let miniconda take care of the rest...
  echo "Installing Python via HomeBrew"
  # python 2.7 should already be there by default, but check...
  python --version || brew install python

  # get pip via curl
  echo "Getting 'distribute' python package (pip dependency)"
  curl -O http://python-distribute.org/distribute_setup.py
  python distribute_setup.py
  curl -O http://raw.github.com/pypa/pip/master/contrib/get-pip.py
  python get-pip.py

  # install virtualenv (don't activate) in case we ever use more than just conda testing
  pip install virtualenv

  # we SHOULD have git (I think the branch is cloned prior to before_install?)...
  # but make sure brew has it installed so we can clone the branch otherwise.
  # Again, this will fail if git already exists (why does homebrew not just warn for
  # these types of things?) so provide alternative
  git --version || brew install git
fi
