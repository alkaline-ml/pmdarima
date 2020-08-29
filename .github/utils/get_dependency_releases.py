import sys

import requests

if len(sys.argv) < 2:
    print(f'Usage: python {sys.argv[0]} <list of packages>')
    sys.exit(1)

packages = sys.argv[1:]
session = requests.Session()

# This is the format for a two-column table. We start with the headers, and
# build up the rest in the below loop. https://stackoverflow.com/a/55734396/10696164
releases = [
    {
        "type": "mrkdwn",
        "text": "*Package*"
    },
    {
        "type": "mrkdwn",
        "text": "*Release Date*"
    }
]
for package in packages:
    pypi = session.get(f'https://pypi.org/pypi/{package}/json').json()
    latest_version = pypi['info']['version']
    latest_release_date = pypi['releases'][latest_version][0]['upload_time']
    releases.append({
        'type': 'plain_text',
        'text': f'{package} {latest_version}'
    })
    releases.append({
        'type': 'plain_text',
        'text': latest_release_date.replace('T', ' ') + ' UTC'
    })

session.close()
print(releases) # this will be used in GitHub Actions
