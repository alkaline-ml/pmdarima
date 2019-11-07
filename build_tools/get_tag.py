from pathlib import Path
import os

# This file assumes that our tags are always in this format: vX.X.X.
# In that case, we would only want to write X.X.X

OUT_FILE = Path(__file__).parent / 'VERSION'

# Circle is easy, since they give us the git tag
if os.getenv('CIRCLECI', False) and os.getenv('GIT_TAG', False):
    print('Tagged commit on Circle CI. Writing to {0}'.format(OUT_FILE))
    with open(OUT_FILE, 'w') as file:
        file.write(os.getenv('GIT_TAG')[1:])

elif os.getenv('BUILD_SOURCEBRANCH', False):
    # This is for debugging on ADO right now
    for param in os.environ.keys():
        print("%20s %s" % (param, os.environ[param]))

    # print('Not a tagged commit, or not on a CI/CD platform. Not writing VERSION file')
