#!/bin/bash

set -e -x

if [[ "${DEPLOY}" != true ]]; then
    # Not a release
    exit 0
fi

# Create base directory
pushd $(dirname $0) > /dev/null
_root=$(dirname $(dirname $(pwd -P))) # get one directory up from parent to get to root dir
popd > /dev/null

if [ "${TRAVIS_OS_NAME}" == "linux" ]; then
    echo "Building LINUX OS wheels"

    for pyver in ${PYTHON_VERSION}; do
        ML_PYTHON_VERSION=$(python3 -c \
            "print('cp{maj}{min}-cp{maj}{min}m'.format( \
                   maj='${pyver}'.split('.')[0], \
                   min='${pyver}'.split('.')[1]))")

        for arch in x86_64; do
            ML_IMAGE="quay.io/pypa/manylinux1_${arch}"
            docker pull "${ML_IMAGE}"
            docker run \
                --name "${DOCKER_CONTAINER_NAME}" \
                -v "${_root}":/io \
                -e "PYMODULE=${PYMODULE}" \
                -e "PYTHON_VERSION=${ML_PYTHON_VERSION}" \
                "${ML_IMAGE}" "/io/build_tools/travis/build_manywheels_linux.sh"
            sudo docker cp "${DOCKER_CONTAINER_NAME}:/io/dist/" "${_root}/dist/"
            docker rm $(docker ps -a -f status=exited -q)
        done
    done
elif [ "${TRAVIS_OS_NAME}" == "osx" ]; then
    # this should be all that's required, right? We already removed the .egg-info
    # directory so no locally cached SOURCES.txt with absolute paths will blow things up
    python setup.py bdist_wheel
else
    echo "Cannot build on ${TRAVIS_OS_NAME}."
fi

# only one env will have us build the tar file for src dist
if [[ "$BUILD_TAR" ]]; then
    echo "Building .tar for source release to pypi"
    python setup.py sdist
fi
