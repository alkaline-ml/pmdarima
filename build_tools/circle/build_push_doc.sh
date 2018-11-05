#!/bin/bash

set -e

# this is a hack, but we have to make sure we're only ever running this from
# the top level of the package and not in the subdirectory...
if [[ ! -d pmdarima/__check_build ]]; then
    echo "This must be run from the pmdarima project directory"
    exit 3
fi

# get the running branch
branch=$(git symbolic-ref --short HEAD)

# cd into docs, make them
cd doc
make clean html EXAMPLES_PATTERN=ex_*
cd ..

# move the docs to the top-level directory, stash for checkout
mv doc/_build/html ./

# html/ will stay there actually...
git stash

# checkout gh-pages, remove everything but .git, pop the stash
# switch into the gh-pages branch
git checkout gh-pages
git pull origin gh-pages

# Make sure to set the credentials!
git config --global user.email "$GH_EMAIL" > /dev/null 2>&1
git config --global user.name "$GH_NAME" > /dev/null 2>&1

# remove all files that are not in the .git dir
find . -not -name ".git/*" -type f -maxdepth 1 -delete

# Remove the remaining directories. Some of these are artifacts of the LAST
# gh-pages build, and others are remnants of the package itself
declare -a leftover=(".cache/"
                     ".idea/"
                     "build/"
                     "build_tools/"
                     "doc/"
                     "examples/"
                     "pmdarima/"
                     "pmdarima.egg-info/"
                     "_downloads/"
                     "_images/"
                     "_modules/"
                     "_sources/"
                     "_static/"
                     "auto_examples/"
                     "includes"
                     "modules/")

# check for each left over file/dir and remove it
for left in "${leftover[@]}"
do
    rm -r ${left} || echo "${left} does not exist; will not remove"
done

# we need this empty file for git not to try to build a jekyll project
touch .nojekyll
mv html/* ./
rm -r html/

# Add everything, get ready for commit. But only do it if we're on master
if [[ "${CIRCLE_BRANCH}" =~ ^master$|^[0-9]+\.[0-9]+\.X$ ]]; then
    git add --all
    git commit -m "[ci skip] publishing updated documentation..."

    # We have to re-add the origin with the GH_TOKEN credentials
    git remote rm origin
    git remote add origin https://${GH_NAME}:${GH_TOKEN}@github.com/${GH_NAME}/pyramid.git

    # NOW we should be able to push it
    git push origin gh-pages
else
    echo "Not on master, so won't push doc"
fi
