#!/bin/bash

# TODO: what if it's a tag? Will it fail since the version dir already exists?

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
make clean html EXAMPLES_PATTERN=example_*
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


function deploy() {
  git add --all
  git commit -m "[ci skip] Publishing updated documentation for '${CIRCLE_BRANCH}' branch"

  # We have to re-add the origin with the GH_TOKEN credentials
  git remote rm origin
  git remote add origin https://${GH_NAME}:${GH_TOKEN}@github.com/${GH_NAME}/pmdarima.git

  # NOW we should be able to push it
  git push origin gh-pages
}


# If we're on master or develop, we'll end up deploying
echo "Branch name: ${CIRCLE_BRANCH}"

# Show the present files:
ls -la

# On both of these, we'll need to remove the artifacts from the package
# build itself
declare -a leftover=("benchmarks"
                     "build"
                     "dist"
                     "doc"
                     "pmdarima"
                     "pmdarima.egg-info")

# check for each left over file/dir and remove it
for left in "${leftover[@]}"
do
  echo "Removing ${left}"
  rm -rf ${left}
done

# If it's develop, we can simply rename the "html" directory as the
# "develop" directory
if [[ ${CIRCLE_BRANCH} == "develop" ]]; then

  # Remove the existing 'dev' folder (if it's there. might not be the
  # first time we do this)
  if [[ -d "develop" ]]; then
    rm -rf develop/
  fi

  # Rename the html dir
  mv html develop
  # That's it for dev

# Otherwise it's master or another branch, which is a bit more involved.
# Remove the artifacts from the previous deployment, move the new ones into the
# folder as well as into the versioned folder. This won't be deployed unless
# it's master
else

  # These are the web artifacts we want to remove from the base
  declare -a artifacts=("_downloads"
                        "_images"
                        "_modules"
                        "_sources"
                        "_static"
                        "auto_examples"
                        "includes"
                        "modules")

  for artifact in "${artifacts[@]}"
  do
    echo "Removing ${artifact}"
    rm -rf ${artifact}
  done

  # Make a copy of the html directory. We'll rename this as the versioned dir
  echo "Copying html directory"
  cp -a html html_copy

  # Move the HTML contents into the local dir
  mv html/* ./
  rm -r html/

  # Get the new version. This overwrites the old one.
  python -c "import pmdarima; print(pmdarima.__version__)" > VERSION

  # If the version already has a folder, we have to fail out. We don't
  # want to overwrite an existing version's documentation
  version=`cat VERSION`
  echo "New version: ${version}"
  if [[ -d ${version} ]]; then
    echo "Version ${version} already exists!! Will not overwrite. Failing job."
    exit 9
  fi

  # If we get here, we can simply rename the html_copy dir as the versioned
  # directory to be deployed.
  mv html_copy ${version}
fi

# we need this empty file for git not to try to build a jekyll project
touch .nojekyll
echo "Final directory contents:"
ls -la

# Finally, deploy the branch, but if it's a pull request or tag, don't!!
if [[ ! -z ${CIRCLE_PULL_REQUEST} ]]; then
  echo "Will not deploy doc on pull request (${CIRCLE_PULL_REQUEST})"
elif [[ ! -z ${CIRCLE_TAG} ]]; then
  # We do this since we deploy the documentation on Master anyways, and if it
  # encounters the already-existing versioned directory, it fails out (as coded
  # above with exit 9)
  echo "Will not re-deploy doc on tag (${CIRCLE_TAG})"
elif [[ ${CIRCLE_BRANCH} == "master" || ${CIRCLE_BRANCH} == "develop" ]]; then
  echo "Deploying documentation"
  deploy
else
  echo "Not on master or develop. Will not deploy doc"
fi
