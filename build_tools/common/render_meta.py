import os
import sys

from jinja2 import Environment, FileSystemLoader


DIST_PATH = '../../dist'  # Only used on Azure DevOps
REQUIREMENTS_FILE = '../../requirements.txt'
OUTPUT_FILE = '../../conda/meta.yaml'  # conda is weird about yml vs yaml, so we have to use yaml
TEMPLATE_PATH = '.'
TEMPLATE_ENVIRONMENT = Environment(
    autoescape=False,
    loader=FileSystemLoader(TEMPLATE_PATH),
    trim_blocks=False
)

with open(REQUIREMENTS_FILE) as file:
    requirements = [line.strip() for line in file.readlines()]

numpy_version = next(package for package in requirements if 'numpy' in package)

# If we are on Circle, we install from source. If we are on ADO, we install from a wheel
if os.environ.get('CIRCLECI', True) or sys.platform == 'win32':
    build_script = '{{ PYTHON }} -m pip install --no-deps --ignore-installed .'
else:
    wheel = next(file for file in os.listdir(DIST_PATH) if file.endswith('.whl'))
    # We need 4 braces here to render 2 braces (due to f-string)
    build_script = f'{{{{ PYTHON }}}} -m pip install dist/{wheel} --no-deps -vv'

context = {
    'requirements': requirements,
    'numpy_version': numpy_version,
    'build_script': build_script
}

with open(OUTPUT_FILE, 'w') as out:
    meta = TEMPLATE_ENVIRONMENT.get_template('meta_template.yml').render(context)
    out.write(meta)
