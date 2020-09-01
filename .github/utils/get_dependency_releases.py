import sys

import requests
from tabulate import tabulate

if len(sys.argv) < 2:
    print(f'Usage: python {sys.argv[0]} <list of packages>')
    sys.exit(1)

packages = sys.argv[1:]
session = requests.Session()

releases = []
for package in packages:
    response = session.get(f'https://pypi.org/pypi/{package}/json')
    response.raise_for_status()
    pypi = response.json()
    latest_version = pypi['info']['version']
    latest_release_date = pypi['releases'][latest_version][0]['upload_time']
    releases.append([
        package, latest_version, latest_release_date.replace('T', ' ') + ' UTC']
    )

session.close()

table = tabulate(
    sorted(releases, key=lambda entry: entry[2], reverse=True),
    headers=['Package', 'Version', 'Release Date']
)
# Need repr so this is on one line for Slack
print(repr('```\n' + table + '\n```').replace("'", ''))
