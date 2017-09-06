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

#########
# STEPS #
#########

# This script will check if the number of input arguments are 0
#   A) If arguments == 0

#       1) cd doc && make clean html         # purge old docs and create new
#       2) cp -R ../ghpages/html ../         # copy html folder to base directory of master
#       3) git stash                         # delete unwanted/uncommited changes to master
#       4) git checkout gh-pages             # switch to gh-pages
#       5) git pull origin gh-pages          # pull in latest changes from gh-pages
#       6) cp -R ../html/* ..                # copy new html to base directory of gh-pages
#       7) rm -r ../html                     # remove html folder
#       8) git add -A                        # add the new and modified files
#       9) git commit -m "$COMMIT_MESSAGE"   # commit the added changes
#       10) git push origin gh-pages         # push the changes to gh-pages
#       11) git checkout master              # return to master
#
#   B) If arguments != 0
#       1) print messages on proper usage

#############
# EXECUTION #
#############

if test $# -eq 0; then

    VERSION_NUMBER=`python -c 'import pyramid as p; '\
                   'from pkg_resources import parse_version as pv; '\
                   'print(str(pv(p.__version__)))'`
    COMMIT_MESSAGE="Update for version - $VERSION_NUMBER"
    COMMIT_TIME=`date "+%m/%d/%Y %H:%M"`

    echo "\n\nSee note A.1) make clean html\n\n"
    make clean html

    echo "\n\nSee note A.2) ../gh-pages/html ../\n\n"
    cp -R ../gh-pages/html ../

    echo "\n\nSee note A.3) git stash\n\n"
    git stash

    echo "\n\nSee note A.4) git checkout gh-pages\n\n"
    git fetch
    git checkout gh-pages

    echo "\n\nSee note A.5) git pull origin gh-pages\n\n"
    git pull origin gh-pages

    echo "\n\nSee note A.6)cp -R ../html/* ..\n\n"
    cp -R ../html/* ..

    echo "\n\nSee note A.7)rm -r ../html"
    rm -r ../html

    echo "\n\nSee note A.6) git add -A\n\n"
    git add -A

    echo "\n\nSee note A.7) git commit -m '$COMMIT_MESSAGE'\n\n"
    git commit -m "$COMMIT_MESSAGE"

    echo "\n\nSee note A.8) git push origin gh-pages\n\n"
    git push origin gh-pages

    echo "\n\nSee note A.9) git checkout master\n\n"
    git checkout master

else
    echo "\n>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<"
    echo ">>>>>>>Illegal number of arguments supplied!<<<<<<<"
    echo "Please invoke script like in the following example:"
    echo ">>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<\n"
    echo "sh publish_gh_pages.sh\n"
fi