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

# we only really want to do this from master
if [[ ${branch} != "master" ]]; then
    echo "This must be run from the master branch"
    exit 5
fi

# make sure no untracked changes in git
if [[ -n $(git status -s) ]]; then
    echo "You have untracked changes in git"
    exit 7
fi

# setup the project
python setup.py install

# cd into docs, make them
cd doc
make clean html EXAMPLES_PATTERN=ex_*
cd ..

# move the docs to the top-level directory, stash for checkout
mv doc/_build/html ./

# We do NOT want to remove the .idea/ folder if it's there, because it contains
# user preferences for PyCharm. So we'll move it back one level, rename it, and
# then retrieve it after we switch back over to master
tmp_idea_dir="../.tmp_idea/"
if [[ -d .idea/ ]]; then
    echo "Found .idea/ directory. Moving it to ${tmp_idea_dir} for the push"
    mv .idea/ ${tmp_idea_dir}
fi

# html/ will stay there actually...
git stash

# checkout gh-pages, remove everything but .git, pop the stash
git checkout gh-pages
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

# add everything, get ready for commit
git add --all
git commit -m "[ci skip] publishing updated documentation..."
git push origin gh-pages

# switch back to master
git checkout master

# Check for the existing .tmp_idea/ and move it back into the directory
# if needed
if [[ -d ${tmp_idea_dir} ]]; then
    echo "Found stashed temporary .idea/ directory at ${tmp_idea_dir}"

    # if there is already an .idea dir, don't do anything
    if [[ ! -d ${tmp_idea_dir} ]]; then
        echo "Moving stashed temporary .idea/ back to git repo"
        mv ${tmp_idea_dir} .idea/
    else
        echo "Existing .idea/ found. Will not replace with ${tmp_idea_dir}"
    fi
fi
