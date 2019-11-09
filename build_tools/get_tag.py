import os
from os.path import abspath, dirname

# This file assumes that our tags are always in this format: vX.X.X.
# In that case, we would only want to write X.X.X

TOP_LEVEL = abspath(dirname(dirname(__file__)))
OUT_FILE = os.path.join(TOP_LEVEL, 'pmdarima', 'VERSION')


def get_version_from_tag(tag):
    """Handles 1.5.0 or v1.5.0"""
    return tag[1:] if tag.startswith('v') else tag


# Circle is easy, since they give us the git tag
if os.getenv('CIRCLECI', False) and os.getenv('GIT_TAG', False):
    print('Tagged commit on Circle CI. Writing to {0}'.format(OUT_FILE))
    with open(OUT_FILE, 'w') as f:
        tag = get_version_from_tag(os.getenv('GIT_TAG'))
        f.write(tag)

# on Azure Pipelines, we have to look at the build branch, and apply this logic:
# refs/tags/vX.X.X -> ['refs', 'tags', 'vX.X.X'] -> 'vX.X.X'
elif os.getenv('BUILD_SOURCEBRANCH', False) and os.getenv('BUILD_SOURCEBRANCH').startswith('refs/tags/'):
    print('Tagged commit on Azure Pipelines. Writing to {0}'.format(OUT_FILE))
    with open(OUT_FILE, 'w') as f:
        tag = os.getenv('BUILD_SOURCEBRANCH').split('/')[-1]
        f.write(get_version_from_tag(tag))

# Local or non-tagged commit, so we don't generate a VERSION file
else:
    print('Not a tagged commit, or not on a CI/CD platform. Not writing VERSION file')
