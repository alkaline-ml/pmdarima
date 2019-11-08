#!/bin/bash

set -e
pip install pathlib

BUILD_SOURCEBRANCH=refs/tags/v0.99.999 python ${BUILD_SOURCESDIRECTORY}/build_tools/get_tag.py

if [[ ! -f ${BUILD_SOURCESDIRECTORY}/pmdarima/VERSION ]]; then
    echo "Expected VERSION file"
    exit 4
fi
