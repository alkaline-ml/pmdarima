from datetime import datetime

import requests

session = requests.Session()
package = 'pmdarima'

info = session.get(f'https://pypi.org/pypi/{package}/json').json()
latest_release_date = info['releases'][-1]['upload_time']
print(latest_release_date)