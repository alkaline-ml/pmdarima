#!/usr/bin/env bash

set -e

# if it's a linux build, we need to sleep before test:
if [[ "$TRAVIS_OS_NAME" == "linux" ]]; then
  export DISPLAY=:99.0
  sh -e /etc/init.d/xvfb start
  sleep 5 # give xvfb some time to start by sleeping for 5 seconds
fi
