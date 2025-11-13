"""This script reads pyproject.toml and removes the pinned versions"""

import re
import os
from os.path import abspath, dirname

TOP_LEVEL = abspath(dirname(dirname(dirname(__file__))))
PYPROJECT_TOML = os.path.join(TOP_LEVEL, 'pyproject.toml')


def parse_pyproject_dependencies(pyproject_path):
    """Parse dependencies from pyproject.toml

    Parameters
    ----------
    pyproject_path : str
        Path to pyproject.toml file

    Returns
    -------
    requirements : list
        List of parsed dependencies without their pinned versions
    """
    requirements = []
    in_dependencies = False
    in_optional_dependencies = False
    in_optional_section = False

    with open(pyproject_path) as file:
        for line in file:
            line = line.strip()

            # Start of main dependencies section
            if line.startswith('dependencies = ['):
                in_dependencies = True
                continue

            # Start of optional dependencies section
            if line.startswith('[project.optional-dependencies]'):
                in_optional_dependencies = True
                continue

            # End of dependencies section
            if in_dependencies and line == ']':
                in_dependencies = False
                continue

            # Start of a new optional dependency group (e.g., test = [...])
            if in_optional_dependencies and '= [' in line:
                in_optional_section = True
                continue

            # End of optional dependency group
            if in_optional_section and line == ']':
                in_optional_section = False
                continue

            # End of optional-dependencies section (next [section] starts)
            if in_optional_dependencies and line.startswith('[') and 'optional-dependencies' not in line:
                in_optional_dependencies = False
                continue

            # Parse dependency lines from main or optional dependencies
            if (in_dependencies or in_optional_section) and line.startswith('"'):
                # Skip comments
                if line.startswith('#'):
                    continue
                # Extract package name from quoted dependency string
                # e.g., "numpy>=1.21.6" -> numpy
                match = re.match(r'^"([A-Za-z\-0-9]+)', line)
                if match:
                    pkg_name = match.group(1).lower()
                    if pkg_name not in requirements:
                        requirements.append(pkg_name)

    return requirements


# Get all dependencies from pyproject.toml (main + optional)
requirements = parse_pyproject_dependencies(PYPROJECT_TOML)

# We print because this is called from a bash script and we need to return a
# space-separated list
print(' '.join(requirements))
