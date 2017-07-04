#!/bin/bash
# The purpose of this script is to manage CD to pypi server. Since deployment
# depends on a tag being created, this script manages several things:
#   * Ensure we are on master branch
#   * Get the version number from pyramid/__init__.py
#   * Make sure there is no 'dev' in the version number
#   * Ensure the master branch has been pushed already
#   * Create and push the tag to git.

# current Git branch
branch=$(git symbolic-ref HEAD | sed -e 's,.*/\(.*\),\1,')

# if branch is NOT master, error out
if [[ "$branch" != "master" ]]; then
    echo "Not on master! Branch=$branch. Auto-tagging/release should only occur on master"
    exit -6
fi

# Get the version of pyramid - this is a major hack
VERSION_NUMBER=`cat pyramid/__init__.py | grep "version" | tr "=" "\n" | grep -v "version"`
# Strip out the quotes...
VERSION_NUMBER=`echo "$VERSION_NUMBER" | sed -e 's/^ "//' -e 's/"$//'`
# Append the 'v' prefix
versionLabel=v$VERSION_NUMBER

# If there is a 'dev' anything suffix in the versionLabel, fail out. Need to increment version
if [[ $versionLabel == *"dev"* ]]; then
    echo "Version number contains 'dev' suffix. Re-version then run again."
    exit -3
elif git diff-index --quiet HEAD --; then
    # no changes - master has been pushed
    echo "Tagging/release version $versionLabel"
else
    # changes need to be pushed
    echo "Local changes in git repo need pushing before tagging release."
fi

# Create the tag for this version
git tag -a $versionLabel -m "Release $versionLabel"

# Push the tag, which should trigger the release of the wheels.
git push origin $versionLabel
