#!/bin/bash

set -e
pip install pathlib

GIT_TAG=0.99.999 python ../get_tag.py

if [[ ! -f ~/pmdarima/pmdarima/VERSION ]]; then
    echo "Expected VERSION file"
    exit 4
fi
