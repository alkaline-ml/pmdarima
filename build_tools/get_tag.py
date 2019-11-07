from pathlib import Path
import os

# This file assumes that our tags are always in this format: vX.X.X.
# In that case, we would only want to write X.X.X

OUT_FILE = Path(__file__).parent / 'VERSION'

# Circle is easy, since they give us the git tag
if os.getenv('CIRCLECI', False) and os.getenv('GIT_TAG', False):
    print('Tagged commit on Circle CI. Writing to {0}'.format(OUT_FILE))
    tag = os.getenv('GIT_TAG')
    with open(OUT_FILE, 'w') as file:
        file.write(tag[1:] if tag.startswith('v') else tag)

# on Azure Pipelines, we have to look at the build branch, and apply this logic:
# refs/tags/vX.X.X -> ['refs', 'tags', 'vX.X.X'] -> 'vX.X.X'
elif os.getenv('BUILD_SOURCEBRANCH', False) and os.getenv('BUILD_SOURCEBRANCH').startswith('refs/tags/'):
    print('Tagged commit on Azure Pipelines. Writing to {0}'.format(OUT_FILE))
    tag = os.getenv('BUILD_SOURCEBRANCH').split('/')[-1]
    with open(OUT_FILE, 'w') as file:
        file.write(tag[1:] if tag.startswith('v') else tag)

# Local or non-tagged commit, so we don't generate a VERSION file
else:
    print('Not a tagged commit, or not on a CI/CD platform. Not writing VERSION file')
