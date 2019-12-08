import os

from jinja2 import Environment, FileSystemLoader

# pathlib fails on Github Actions using Python 3.5, so we have to use these
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

try:
    VERSION = open(VERSION_FILE).readline().strip()
except FileNotFoundError:
    VERSION = '0.0.0'

with open(REQUIREMENTS_FILE) as file:
    requirements = [line.strip() for line in file.readlines()]

wheel = next(file for file in os.listdir(DIST_PATH) if file.endswith('.whl'))
numpy_version = next(package for package in requirements if 'numpy' in package)

context = {
    'requirements': requirements,
    'numpy_version': numpy_version,
    'VERSION': VERSION,
    'wheel': wheel
}

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(OUTPUT_FILE, 'w') as out:
    meta = TEMPLATE_ENVIRONMENT.get_template('meta_template.yml').render(context)
    out.write(meta)
