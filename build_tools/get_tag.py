import os
from os.path import abspath, dirname

# This file assumes that our tags are always in this format: vX.X.X.
# In that case, we would only want to write X.X.X

TOP_LEVEL = abspath(dirname(dirname(__file__)))
OUT_FILE = os.path.join(TOP_LEVEL, 'pmdarima', 'VERSION')
DEFAULT_TAG = '0.0.0'


def get_version_from_tag(tag):
    """Handles 1.5.0 or v1.5.0"""
    return tag[1:] if tag.startswith('v') else tag


# Circle is easy, since they give us the git tag
if os.getenv('CIRCLECI', False) and os.getenv('CIRCLE_TAG', False):
    print('Tagged commit on Circle CI. Writing to {0}'.format(OUT_FILE))
    with open(OUT_FILE, 'w') as f:
        tag = get_version_from_tag(os.getenv('CIRCLE_TAG'))
        f.write(tag)

elif os.getenv('GITHUB_REF') and \
        os.getenv('GITHUB_REF').startswith('refs/tags/'):
    print('Tagged commit on Github Actions. Writing to {0}'.format(OUT_FILE))
    with open(OUT_FILE, 'w') as f:
        tag = os.getenv('GITHUB_REF').split('/')[-1]
        f.write(get_version_from_tag(tag))

# Local or non-tagged commit. setuptools requires a VERSION file, so just write a default one
else:
    print("Not a tagged commit or not on CI. Using default tag")
    with open(OUT_FILE, 'w') as f:
        f.write(DEFAULT_TAG)
