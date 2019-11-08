#!/bin/bash

set -e
pip install pathlib

BUILD_SOURCEBRANCH=refs/tags/v0.99.999 python ${BUILD_SOURCESDIRECTORY}/pmdarima/build_tools/get_tag.py

if [[ ! -f ${BUILD_SOURCESDIRECTORY}/pmdarima/pmdarima/VERSION ]]; then
    echo "Expected VERSION file"
    exit 4
fi
