import os
import re

from jinja2 import Environment, FileSystemLoader
from pathlib import Path

# Since conda is only on Azure Pipelines, we can use their env variables
ROOT_DIRECTORY = Path(os.getenv('BUILD_SOURCESDIRECTORY'))
DIST_PATH = ROOT_DIRECTORY / 'dist'
VERSION_FILE = ROOT_DIRECTORY / 'pmdarima' / 'VERSION'
REQUIREMENTS_FILE = ROOT_DIRECTORY / 'requirements.txt'

# conda is weird about yml vs yaml, so we have to use yaml
OUTPUT_DIR = ROOT_DIRECTORY / 'conda'
OUTPUT_FILE = OUTPUT_DIR / 'meta.yaml'

TEMPLATE_PATH = Path(__file__).parent
TEMPLATE_ENVIRONMENT = Environment(
    autoescape=False,
    loader=FileSystemLoader(str(TEMPLATE_PATH.resolve())),
    trim_blocks=False
)

# Find the version
try:
    VERSION = open(str(VERSION_FILE.resolve())).readline().strip()
except FileNotFoundError:
    VERSION = '0.0.0'

# Find the requirements and versions
# conda puts a space between packages and versions, so we have to match that
requirements = []
with open(str(REQUIREMENTS_FILE.resolve())) as file:
    for line in file:
        requirement = line.strip()
        match = re.match(r'^([A-Za-z\-0-9]+)', requirement)
        _, match_end = match.span()
        package = match.group(0)
        version = requirement[match_end:].replace('==', '')
        requirements.append(f'{package} {version}')

# Render and write the meta.yaml file to $ROOT/conda/meta.yaml
context = {
    'requirements': requirements,
    'VERSION': VERSION
}

# Ensure output directory exists
os.makedirs(str(OUTPUT_DIR.resolve()), exist_ok=True)

with open(str(OUTPUT_FILE.resolve()), 'w') as out:
    meta = TEMPLATE_ENVIRONMENT.get_template('meta_template.yml.j2').render(context)
    out.write(meta)
