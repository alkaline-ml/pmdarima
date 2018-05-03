#!/usr/bin/env bash

set -e

# if it's a linux build, we need to apt-get update
if [[ "$TRAVIS_OS_NAME" == "linux" ]]; then
  echo "Updating apt-get for Linux build"
  sudo apt-get -qq update

# Workaround for https://github.com/travis-ci/travis-ci/issues/6307, which
# caused the following error on MacOS X workers:
#
# Warning, RVM 1.26.0 introduces signed releases and automated check of
# signatures when GPG software found.
# /Users/travis/build.sh: line 109: shell_session_update: command not found
elif [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
  echo "Updating Ruby for Mac OS build"
  # command curl -sSL https://rvm.io/mpapis.asc | gpg --import -;
  # rvm get stable
fi
