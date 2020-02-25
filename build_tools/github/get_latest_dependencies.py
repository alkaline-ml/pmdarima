"""This script reads our requirements.txt file and removes the pinned versions"""

import re
import os
from os.path import abspath, dirname

TOP_LEVEL = abspath(dirname(dirname(dirname(__file__))))
REQUIREMENTS = os.path.join(TOP_LEVEL, 'requirements.txt')
BUILD_REQUIREMENTS = os.path.join(TOP_LEVEL, 'build_tools', 'build_requirements.txt')


def find_latest_dependencies(*requirements_files):
    """Given one or more requirements.txt files, strip off any pinned versions

    Parameters
    ----------
    *requirements_files : str
        One or more paths to requirements.txt files to parse

    Returns
    -------
    requirements : list
        List of parsed dependencies without their pinned versions
    """
    requirements = []
    for requirements_file in requirements_files:
        with open(requirements_file) as file:
            for line in file:
                requirement = line.strip()
                if line.startswith('#'):
                    continue
                match = re.match(r'^([A-Za-z\-0-9]+)', requirement)
                if match.group(0).lower() not in requirements:
                    requirements.append(match.group(0).lower())

    return requirements


requirements = find_latest_dependencies(REQUIREMENTS, BUILD_REQUIREMENTS)
# We print because this is called from a bash script and we need to return a
# space-separated list
print(' '.join(requirements))
