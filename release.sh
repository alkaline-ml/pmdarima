#!/bin/bash
# The purpose of this script is to manage CD to pypi server. Since deployment
# depends on a tag being created, this script manages several things:
#   * Ensure we are on master branch
#   * Get the version number from pyramid/__init__.py
#   * Make sure there is no 'dev' in the version number
#   * Ensure the master branch has been pushed already
#   * Create and push the tag to git.
REPO=pyramid
OWNER=tgsmith61591

# current Git branch
branch=$(git symbolic-ref HEAD | sed -e 's,.*/\(.*\),\1,')

# function to bail. Abstract into single function (rather than multiple exit calls)
# so can pass for testing and make sure the whole script functions...
bail() {
    exit 12
}

# if branch is NOT master, error out
if [[ "$branch" != "master" ]]; then
    echo "Not on master! Branch=$branch. Auto-tagging/release should only occur on master"
    bail
fi

# Get the version of pyramid - this is a major hack
VERSION_NUMBER=`cat ${REPO}/__init__.py | grep "version" | tr "=" "\n" | grep -v "version"`
# Strip out the quotes...
VERSION_NUMBER=`echo "$VERSION_NUMBER" | sed -e 's/^ "//' -e 's/"$//'`
# Append the 'v' prefix
versionLabel=v$VERSION_NUMBER

# If there is a 'dev' anything suffix in the versionLabel, fail out. Need to increment version
if [[ $versionLabel == *"dev"* ]]; then
    echo "Version number contains 'dev' suffix. Re-version then run again."
    bail
elif git diff-index --quiet HEAD --; then
    # no changes - master has been pushed
    echo "Tagging/release version $versionLabel"
else
    # changes need to be pushed
    echo "Local changes in git repo need pushing before tagging release."
    bail
fi

# The release in pypi will reference an archive link. Need to ensure this link
# will not 404! Therefore, also automate the release to github from here. Build
# the JSON string here.. this will also create the tag on Git!
ACCESS_TOKEN=`cat ACCESS_TOKEN`
API_JSON=$(printf '{"tag_name": "%s", "target_commitish": "master", "name": "%s", "body": "Release of %s", "draft": false, "prerelease": false}' $versionLabel $versionLabel $versionLabel)
echo "Curling JSON: ${API_JSON}"
curl --data "$API_JSON" https://api.github.com/repos/${OWNER}/${REPO}/releases?access_token=${ACCESS_TOKEN}
