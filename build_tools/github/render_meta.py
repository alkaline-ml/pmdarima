from jinja2 import Environment, FileSystemLoader
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
REQUIREMENTS_FILE = ROOT / 'requirements.txt'
OUTPUT_FILE = ROOT / 'conda/meta.yaml'  # conda is weird about yml vs yaml, so we have to use yaml
TEMPLATE_ENVIRONMENT = Environment(
    autoescape=False,
    loader=FileSystemLoader(str(HERE.resolve())),
    trim_blocks=False
)
try:
    VERSION = (ROOT / 'VERSION').read_text()
except FileNotFoundError:
    VERSION = '0.0.0'

with open(REQUIREMENTS_FILE) as file:
    requirements = [line.strip() for line in file.readlines()]

numpy_version = next(package for package in requirements if 'numpy' in package)

context = {
    'requirements': requirements,
    'numpy_version': numpy_version,
    'VERSION': VERSION
}

with open(OUTPUT_FILE, 'w') as out:
    meta = TEMPLATE_ENVIRONMENT.get_template('meta_template.yml').render(context)
    out.write(meta)
