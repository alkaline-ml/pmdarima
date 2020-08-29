import sys

import requests
from tabulate import tabulate

if len(sys.argv) < 2:
    print(f'Usage: python {sys.argv[0]} <list of packages>')
    sys.exit(1)

packages = sys.argv[1:]
session = requests.Session()

releases = []
for package in sorted(packages):
    pypi = session.get(f'https://pypi.org/pypi/{package}/json').json()
    latest_version = pypi['info']['version']
    latest_release_date = pypi['releases'][latest_version][0]['upload_time']
    releases.append([
        package, latest_version, latest_release_date.replace('T', ' ') + ' UTC']
    )

session.close()
print('```\n' + tabulate(releases, headers=['Package', 'Version', 'Release Date']) + '\n```')
