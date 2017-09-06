#!/usr/bin/env bash

# fail out if errors
set -e

#########
# NOTES #
#########

# This script should be run immediately after a push
# has been made to the master branch to update the gh-pages branch.

# As part of those changes we need to rebuild the project
# to pull in the new updated version number.

# You will want to run this script from
# its current location (i.e. sh publish_gh_pages.sh)

if test $# -eq 0; then

    VERSION_NUMBER=`python -c 'import pyramid as p; '\
                   'from pkg_resources import parse_version as pv; '\
                   'print(str(pv(p.__version__)))'`

    exit 0

    COMMIT_MESSAGE="Update for version - $VERSION_NUMBER"
    COMMIT_TIME=`date "+%m/%d/%Y %H:%M"`
    CURRENT_BRANCH=`git rev-parse --abbrev-ref HEAD`
    echo "Current branch: ${CURRENT_BRANCH}"

    # checkout gh-pages
    echo "\n\nSee note A.1) git stash\n\n"
    cd ..
    git stash

    echo "\n\nSee note A.2) git checkout gh-pages\n\n"
    git fetch
    git checkout gh-pages

    echo "\n\nSee note A.3) git pull origin gh-pages\n\n"
    git pull origin gh-pages

    echo "\n\nSee note A.4) git merge --no-ff ${CURRENT_BRANCH}\n\n"
    git merge --no-ff ${CURRENT_BRANCH}

    echo "\n\nSee note A.5) make clean html\n\n"
    cd doc && make clean html

    echo "\n\nSee note A.6) on removing everything"
    # delete everything pretty much
    rm -rf ../*.yml

    echo "\n\nSee note A.7) cp _build/html/* ../\n\n"
    cp -R _build/html ../
    cd .. && cp -R html/* .

    echo "\n\nSee note A.8) git add -A\n\n"
    touch .nojekyll
    git add -A

    echo "\n\nSee note A.9) git commit -m '$COMMIT_MESSAGE'\n\n"
    git commit -m "[ci skip] $COMMIT_MESSAGE"

    echo "\n\nSee note A.10) git push origin gh-pages\n\n"
    git push origin gh-pages

    echo "\n\nSee note A.11) git checkout ${CURRENT_BRANCH}\n\n"
    git checkout master

else
    echo "\n>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<"
    echo ">>>>>>>Illegal number of arguments supplied!<<<<<<<"
    echo "Please invoke script like in the following example:"
    echo ">>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<\n"
    echo "sh publish_gh_pages.sh\n"
fi