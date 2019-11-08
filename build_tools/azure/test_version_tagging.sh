#!/bin/bash

set -e
pip install pathlib

BUILD_SOURCEBRANCH=refs/tags/0.99.999 python ${AGENT_SOURCESDIRECTORY}/pmdarima/build_tools/get_tag.py

if [[ ! -f ${AGENT_SOURCESDIRECTORY}/pmdarima/pmdarima/VERSION ]]; then
    echo "Expected VERSION file"
    exit 4
fi
