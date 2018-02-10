#!/bin/bash

set -e -x

if [[ "${DEPLOY}" != true ]]; then
    # Not a release
    exit 0
fi

function build_wheel {
    local pyver=$1
    local arch=$2
    local ucs_setting=$3

    # https://www.python.org/dev/peps/pep-0513/#ucs-2-vs-ucs-4-builds
    ucs_tag="m"
    if [ "$ucs_setting" = "ucs4" ]; then
        ucs_tag="${ucs_tag}u"
    fi

    ML_PYTHON_VERSION=$(python3 -c \
        "print('cp{maj}{min}-cp{maj}{min}{ucs}'.format( \
               maj='${pyver}'.split('.')[0], \
	       min='${pyver}'.split('.')[1], \
	       ucs='${ucs_tag}'))")

    ML_IMAGE="quay.io/pypa/manylinux1_${arch}"
    docker pull "${ML_IMAGE}"
    docker run \
        --name "${DOCKER_CONTAINER_NAME}" \
        -v "${_root}":/io \
        -e "PYMODULE=${PYMODULE}" \
        -e "PYTHON_VERSION=${ML_PYTHON_VERSION}" \
        "${ML_IMAGE}" "/io/build_tools/travis/build_manywheels_linux.sh"
    sudo docker cp "${DOCKER_CONTAINER_NAME}:/io/dist/." "${_root}/dist/"
    docker rm $(docker ps -a -f status=exited -q)
}

# Create base directory
pushd $(dirname $0) > /dev/null
_root=$(dirname $(dirname $(pwd -P))) # get one directory up from parent to get to root dir
popd > /dev/null

if [ "${TRAVIS_OS_NAME}" == "linux" ]; then
    echo "Building LINUX OS wheels"

    for pyver in ${PYTHON_VERSION}; do
        if [ -z "$UCS_SETTING" ] || [ "$UCS_SETTING" = "ucs2" ]; then
            build_wheel $pyver "x86_64" "ucs2"
        elif [ "$UCS_SETTING" = "ucs4" ]; then
            build_wheel $pyver "x86_64" "ucs4"
        else
            echo "Unrecognized UCS_SETTING: ${UCS_SETTING}" 
        fi
    done
elif [ "${TRAVIS_OS_NAME}" == "osx" ]; then
    # this should be all that's required, right? We already removed the .egg-info
    # directory so no locally cached SOURCES.txt with absolute paths will blow things up
    python setup.py bdist_wheel
else
    echo "Cannot build on ${TRAVIS_OS_NAME}."
fi

# build the tar for all dists, but it will only be uploaded on one because of --skip-existing
sudo python setup.py sdist
