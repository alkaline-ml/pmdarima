#!/bin/bash

set -e
pip install pathlib

GIT_TAG=0.99.999 python ${AGENT_BUILDDIRECTORY}/pmdarima/build_tools/get_tag.py

if [[ ! -f ${AGENT_BUILDDIRECTORY}/pmdarima/pmdarima/VERSION ]]; then
    echo "Expected VERSION file"
    exit 4
fi
