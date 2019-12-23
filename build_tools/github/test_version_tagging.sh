#!/bin/bash

set -e
pip install pathlib

GITHUB_REF=refs/tags/v0.99.999 python ${GITHUB_WORKSPACE}/build_tools/get_tag.py

if [[ ! -f ${GITHUB_WORKSPACE}/pmdarima/VERSION ]]; then
    echo "Expected VERSION file"
    exit 4
fi
