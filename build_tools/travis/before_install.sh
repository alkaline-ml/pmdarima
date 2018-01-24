#!/usr/bin/env bash

set -e

# if it's a linux build, we need to apt-get update
if [[ "$TRAVIS_OS_NAME" == "linux" ]]; then
  echo "Updating apt-get for Linux build"
  sudo apt-get -qq update
# Because screw you, Travis:
elif [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
  echo "Updating Ruby for Mac OS build"
  command curl -sSL https://rvm.io/mpapis.asc | gpg --import -;
  rvm get stable
fi
