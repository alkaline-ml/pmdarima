#!/usr/bin/env bash

set -e

# if it's a linux build, we need to apt-get update
if [[ "$TRAVIS_OS_NAME" == "linux" ]]; then
  echo "Updating apt-get for Linux build"
  sudo apt-get -qq update
# Because screw you, Travis:
# elif [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
#   rvm get stable
fi
