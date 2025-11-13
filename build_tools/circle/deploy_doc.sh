#!/bin/bash

set -e

version=$(build_tools/get_tag.py)

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
  git remote add origin "https://${GH_NAME}:${GH_TOKEN}@github.com/alkaline-ml/pmdarima.git"

  # NOW we should be able to push it
  git push origin gh-pages
}


# If we're on master, we'll deploy to /develop (bleeding edge). If from a tag,
# we'll deploy to the version slug (e.g., /1.2.1)
echo "Branch name: ${CIRCLE_BRANCH}"

# Show the present files:
ls -la

# On both of these, we'll need to remove the artifacts from the package
# build itself. The _configtest files are added in the latest version of
# numpy...
declare -a leftover=("_configtest"
                     "_configtest.c"
                     "_configtest.o"
                     "benchmarks"
                     "build"
                     "dist"
                     "doc"
                     "pmdarima"
                     "pmdarima.egg-info")

# check for each left over file/dir and remove it
for left in "${leftover[@]}"
do
  echo "Removing ${left}"
  rm -rf "${left}" || echo "${left} does not exist"
done

# If it's master, we can simply rename the "html" directory as the
# "develop" directory
if [[ ${CIRCLE_BRANCH} == "master" ]]; then

  # Remove the existing 'dev' folder (if it's there. might not be the
  # first time we do this)
  if [[ -d "develop" ]]; then
    rm -rf develop/
  fi

  # Rename the html dir
  mv html develop
  # That's it for dev

# Otherwise it's a tag or another branch, which is a bit more involved.
# Remove the artifacts from the previous deployment, move the new ones into the
# folder as well as into the versioned folder. This won't be deployed unless
# it's a tag
else

  # These are the web artifacts we want to remove from the base
  declare -a artifacts=("_downloads"
                        "_images"
                        "_modules"
                        "_sources"
                        "_static"
                        "auto_examples"
                        "includes"
                        "modules"
                        "rfc"
                        "usecases")

  for artifact in "${artifacts[@]}"
  do
    echo "Removing ${artifact}"
    rm -rf "${artifact}"
  done

  # Make a copy of the html directory. We'll rename this as the versioned dir
  echo "Copying html directory"
  cp -a html html_copy

  # Move the HTML contents into the local dir
  mv html/* ./
  rm -r html/

  echo "${version}" > VERSION
  echo "New version: ${version}"

  # If the version already has a folder, we have to fail out. We don't
  # want to overwrite an existing version's documentation. This should no
  # longer happen on PRs since we don't push the 0.0.0 version
  if [[ -d ${version} ]]; then
    echo "Version ${version} already exists!! Will not overwrite. Failing job."
    exit 9
  fi

  # If we get here, we can simply rename the html_copy dir as the versioned
  # directory to be deployed.
  mv html_copy "${version}"
fi

# we need this empty file for git not to try to build a jekyll project
touch .nojekyll
echo "Final directory contents:"
ls -la

# Finally, deploy the branch, but if it's a pull request, don't!!
if [[ -n ${CIRCLE_PULL_REQUEST} ]]; then
  echo "Will not deploy doc on pull request (${CIRCLE_PULL_REQUEST})"
elif [[ ${CIRCLE_BRANCH} == "master" || ( ( -n ${CIRCLE_TAG} ) && ( ${CIRCLE_TAG} =~ ^v?[0-9]+\.[0-9]+\.?[0-9]*$ ) ) ]]; then
  echo "Deploying documentation"
  deploy
else
  echo "Not on master or production tag. Will not deploy doc"
fi
