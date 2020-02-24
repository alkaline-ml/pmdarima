"""This script reads our requirements.txt file and removes the pinned versions"""

import re
import os
from os.path import abspath, dirname

TOP_LEVEL = abspath(dirname(dirname(dirname(__file__))))
REQUIREMENTS = os.path.join(TOP_LEVEL, 'requirements.txt')
BUILD_REQUIREMENTS = os.path.join(TOP_LEVEL, 'build_tools', 'build_requirements.txt')

requirements = []  # We use a list instead of a set because we want to maintain order
with open(REQUIREMENTS) as file:
    for line in file:
        requirement = line.strip()
        if line.startswith('#'):
            continue
        match = re.match(r'^([A-Za-z\-0-9]+)', requirement)
        if match.group(0).lower() not in requirements:
            requirements.append(match.group(0).lower())

with open(BUILD_REQUIREMENTS) as file:
    for line in file:
        requirement = line.strip()
        if line.startswith('#'):
            continue
        match = re.match(r'^([A-Za-z\-0-9]+)', requirement)
        if match.group(0).lower() not in requirements:
            requirements.append(match.group(0).lower())

print(' '.join(list(requirements)))
