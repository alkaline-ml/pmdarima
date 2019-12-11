import os
import sys

from jinja2 import Environment, FileSystemLoader

# Constant file paths. Relative to this file, so we have to run this file from this directory
DIST_PATH = '../../dist'
VERSION_FILE = '../../VERSION'
REQUIREMENTS_FILE = '../../requirements.txt'
OUTPUT_DIR = '../../conda'
OUTPUT_FILE = '../../conda/meta.yaml'  # conda is weird about yml vs yaml, so we have to use yaml
TEMPLATE_PATH = '.'
TEMPLATE_ENVIRONMENT = Environment(
    autoescape=False,
    loader=FileSystemLoader(TEMPLATE_PATH),
    trim_blocks=False
)

# Find the version
try:
    VERSION = open(VERSION_FILE).readline().strip()
except FileNotFoundError:
    VERSION = '0.0.0'

# Find the requirements and versions
with open(REQUIREMENTS_FILE) as file:
    requirements = [line.strip() for line in file.readlines()]

# We build from source on windows, otherwise, we looks for a wheel
if sys.platform != 'win32':
    wheel = next(file for file in os.listdir(DIST_PATH) if file.endswith('.whl'))
else:
    wheel = None

# Numpy version is used for building
numpy_version = next(package for package in requirements if 'numpy' in package)

# Render and write the meta.yaml file to $ROOT/conda/meta.yaml
context = {
    'requirements': requirements,
    'numpy_version': numpy_version,
    'VERSION': VERSION,
    'wheel': wheel,
    'py_version': '{0.major}{0.minor}'.format(sys.version_info)
}

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(OUTPUT_FILE, 'w') as out:
    meta = TEMPLATE_ENVIRONMENT.get_template('meta_template.yml.j2').render(context)
    out.write(meta)
