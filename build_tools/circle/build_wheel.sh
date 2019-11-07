#!/bin/bash

set -e -x

function build_wheel {
    local pyver=$1
    local arch=$2
    local ucs_setting=$3

    # https://www.python.org/dev/peps/pep-0513/#ucs-2-vs-ucs-4-builds
    ucs_tag="m"
    if [ "$ucs_setting" = "ucs4" ]; then
        ucs_tag="${ucs_tag}u"
    fi

    ML_PYTHON_VERSION=$(python -c \
        "print('cp{maj}{min}-cp{maj}{min}{ucs}'.format( \
               maj='${pyver}'.split('.')[0], \
	       min='${pyver}'.split('.')[1], \
	       ucs='${ucs_tag}'))")

    DOCKER_CONTAINER_NAME="wheel_builder"
    ML_IMAGE="quay.io/pypa/manylinux1_${arch}"

    docker pull "${ML_IMAGE}"
    docker run \
        --name "${DOCKER_CONTAINER_NAME}" \
        -v "${_root}":/io \
        -e "PYMODULE=pmdarima" \
        -e "PYTHON_VERSION=${ML_PYTHON_VERSION}" \
        "${ML_IMAGE}" "/io/build_tools/circle/build_manylinux_wheel.sh"
    sudo docker cp "${DOCKER_CONTAINER_NAME}:/io/dist/." "${_root}/dist/"
    docker rm $(docker ps -a -f status=exited -q)
}

# Guarantee we have the VERSION file before continuing
if [[ ! -f pmdarima/VERSION ]]; then
    echo "VERSION file was not created as expected"
    ls -la
    exit 2
fi

# Create base directory
pushd $(dirname $0) > /dev/null
_root=$(dirname $(dirname $(pwd -P))) # get one directory up from parent to get to root dir
popd > /dev/null

echo "Building LINUX OS wheels"

# Positional arg
pyver=$1

# We no longer explicitly set these... but in the past we did.
if [ -z "$UCS_SETTING" ] || [ "$UCS_SETTING" = "ucs2" ]; then
    build_wheel $pyver "x86_64" "ucs2"
elif [ "$UCS_SETTING" = "ucs4" ]; then
    build_wheel $pyver "x86_64" "ucs4"
else
    echo "Unrecognized UCS_SETTING: ${UCS_SETTING}"
fi
