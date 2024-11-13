#!/bin/bash

# EXECUTED IN A DOCKER CONTAINER

set -e

# Make sure we're in the root PMDARIMA dir (mounted at /io)
cd /io

pip install -r /io/build_tools/build_requirements.txt

make develop
make testing-requirements
make test-unit

# Upload coverage
codecov || echo "codecov upload failed"
