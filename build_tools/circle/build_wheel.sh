#!/bin/bash

set -e -x

function build_wheel {
    local pyver=$1
    local arch=$2
    local ucs_setting=$3

    # https://www.python.org/dev/peps/pep-0513/#ucs-2-vs-ucs-4-builds
    ucs_tag=""
    if [ "$ucs_setting" = "ucs4" ]; then
        ucs_tag="${ucs_tag}u"
    fi

    distutils_version=""
    if [ "$pyver" = "3.12" ]; then
      distutils_version="local"
    else
      distutils_version="stdlib"
    fi

    ML_PYTHON_VERSION=$(python -c \
        "print('cp{maj}{min}-cp{maj}{min}{ucs}'.format( \
               maj='${pyver}'.split('.')[0], \
	       min='${pyver}'.split('.')[1], \
	       ucs='${ucs_tag}'))")

    DOCKER_CONTAINER_NAME=wheel_builder_$(uuidgen)

    ML_IMAGE="quay.io/pypa/manylinux_2_28_${arch}:2023-10-07-c1e05d1" # `latest` as of 2023-10-09
    PMDARIMA_VERSION=`cat ~/pmdarima/pmdarima/VERSION`

    docker pull "${ML_IMAGE}"
    # -v "${_root}":/io \
    docker run \
        --name "${DOCKER_CONTAINER_NAME}" \
        -v `pwd`:/io \
        -e "PYTHON_VERSION=${ML_PYTHON_VERSION}" \
        -e "PMDARIMA_VERSION=${PMDARIMA_VERSION}" \
        -e "SETUPTOOLS_USE_DISTUTILS=${distutils_version}" \
        "${ML_IMAGE}" "/io/build_tools/circle/dind/build_manylinux_wheel.sh"
    sudo docker cp "${DOCKER_CONTAINER_NAME}:/io/dist/." "${_root}/dist/"
    docker rm $(docker ps -a -f status=exited -q)
}

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
