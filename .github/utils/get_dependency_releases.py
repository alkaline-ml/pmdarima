import sys

import requests

if len(sys.argv) < 2:
    print(f'Usage: python {sys.argv[0]} <list of packages>')
    sys.exit(1)

packages = sys.argv[1:]
session = requests.Session()

# Build up our table of release dates
releases = {}
for package in packages:
    pypi = session.get(f'https://pypi.org/pypi/{package}/json').json()
    latest_version = pypi['info']['version']
    latest_release_date = pypi['releases'][latest_version][0]['upload_time']
    releases[package] = latest_release_date.replace('T', ' ') + ' UTC'

session.close()
# Format as a slack message to return to GitHub Action
