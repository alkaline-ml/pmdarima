from datetime import date, timedelta
import json
import math
import os
import requests
from statistics import mean


def millify(n):
    """Abbreviate a number to nearest thousand, million, etc.

    Adapted from: https://stackoverflow.com/a/3155023/10696164

    Parameters
    ----------
    n : int
        The number to abbreviate

    Returns
    -------
    millified : str
        The number abbreviated to the nearest thousand, million, etc.
    """
    millnames = ['', 'k', 'M', 'B', 'T']
    n = float(n)
    millidx = max(
        0,
        min(
            len(millnames) - 1,
            int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))
        )
    )
    final_num = float(n / 10 ** (3 * millidx))
    one_decimal = round(final_num, 1)

    # If the number is in the millions, and has a decimal, we want to show one
    # decimal. I.e.:
    #  - 967123  -> 967k
    #  - 1000123 -> 1M
    #  - 1100123 -> 1.1M
    final_output = one_decimal if n > 1e6 and not one_decimal.is_integer() else int(round(final_num, 0))

    return f'{final_output}{millnames[millidx]}'


def downloads_per_day(downloads_per_version):
    """Find the total number of downloads for a given day

    Parameters
    ----------
    downloads_per_version: dict
        A dict of versions and downloads for that version

    Returns
    -------
    total : int
        The total number of downloads on a given day across all versions.
    """
    total = 0
    for download_stat in downloads_per_version.values():
        total += download_stat

    return total


def get_default_value(downloads):
    """Find the default value (one day's worth of downloads) for a given input

    Parameters
    ----------
    downloads : dict
        A dict of dates and downloads on that day

    Returns
    -------
    default_value : int
        The default value, which is the average of the last 7 days of downloads
        that are contained in the input dictionary.
    """
    last_7_keys = sorted(downloads.keys())[-7:]
    default_value = int(mean([downloads_per_day(downloads[key]) for key in last_7_keys]))
    return default_value


# Used to calculate downloads for the last week
today = date.today()
last_week = today - timedelta(days=7)
DATE_FORMAT = '%Y-%m-%d'

# Open a session to save time
session = requests.Session()

# Get the data for both the legacy namespace and our current one
pyramid_arima = json.loads(session.get('https://api.pepy.tech/api/v2/projects/pyramid-arima').text)
pmdarima = json.loads(session.get('https://api.pepy.tech/api/v2/projects/pmdarima').text)

# Sum up pmdarima and pyramid-arima downloads to the past week
pmdarima_downloads = 0
default_pmdarima_value = get_default_value(pmdarima['downloads'])
for i in range(7):
    new_downloads = pmdarima['downloads'].get((last_week + timedelta(days=i)).strftime(DATE_FORMAT))
    if new_downloads is not None:
        pmdarima_downloads += downloads_per_day(new_downloads)
    else:
        pmdarima_downloads += default_pmdarima_value

pyramid_arima_downloads = 0
default_pyramid_arima_value = get_default_value(pyramid_arima['downloads'])
for i in range(7):
    new_downloads = pyramid_arima['downloads'].get((last_week + timedelta(days=i)).strftime(DATE_FORMAT))
    if new_downloads is not None:
        pyramid_arima_downloads += downloads_per_day(new_downloads)
    else:
        pyramid_arima_downloads += default_pyramid_arima_value

# Millify the totals
total_downloads = millify(pyramid_arima['total_downloads'] + pmdarima['total_downloads'])
weekly_downloads = millify(pmdarima_downloads + pyramid_arima_downloads)

data = {
    'total': total_downloads,
    'weekly': weekly_downloads
}

request = session.post(
    url='https://store.zapier.com/api/records',
    headers={
        'X-Secret': os.getenv('ZAPIER_SHA')
    },
    data=json.dumps(data)
)
request.raise_for_status()

print(f"""
New total downloads: {data['total']}
New weekly downloads: {data['weekly']}
""")
