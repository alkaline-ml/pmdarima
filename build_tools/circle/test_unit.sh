#!/bin/bash

set -e -x

# Create base directory
pushd $(dirname $0) > /dev/null
rootdir=$(dirname $(dirname $(pwd -P))) # get one directory up from parent to get to root dir
popd > /dev/null

echo "Installing package from whl file"

# Construct docker image
pyver=$1
pythonimg="python:${pyver}"

# Mount root as a volume, execute installation + unit tests within the container
env > vars.env
docker run \
    --rm \
    -v `pwd`:/io \
    --env-file vars.env \
    ${pythonimg} \
    sh /io/build_tools/circle/dind/install_and_test.sh

status=$?
exit $status
