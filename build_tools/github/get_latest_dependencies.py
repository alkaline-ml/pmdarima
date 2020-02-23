import re
import os
from os.path import abspath, dirname

TOP_LEVEL = abspath(dirname(dirname(dirname(__file__))))
REQUIREMENTS = os.path.join(TOP_LEVEL, 'requirements.txt')

requirements = []
with open(REQUIREMENTS) as file:
    for line in file:
        requirement = line.strip()
        match = re.match(r'^([A-Za-z\-0-9]+)', requirement)
        requirements.append(match.group(0))

print(' '.join(requirements))
