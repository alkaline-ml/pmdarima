#!/usr/bin/env python3
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

tag = DEFAULT_TAG

# Circle is easy, since they give us the git tag
if os.getenv('CIRCLECI', False) and os.getenv('CIRCLE_TAG', False):
    print(get_version_from_tag(os.getenv('CIRCLE_TAG')))

elif os.getenv('GITHUB_REF') and os.getenv('GITHUB_REF').startswith('refs/tags/'):
    tag = os.getenv('GITHUB_REF').split('/')[-1]
    print(get_version_from_tag(tag))

# Local or non-tagged commit. setuptools requires a VERSION file, so just write a default one
else:
    print(DEFAULT_TAG)
