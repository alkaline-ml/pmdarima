"""
This code is meant to be run on Zapier

It is used to add the `downloads/week` badge to our README
"""
from datetime import date, timedelta
import json
import math
import os
import requests


# This function is duplicated from `zapier_total_downloads`. This is intentional,
# because Zapier only has access to one script at a time, so we cannot import
def millify(n):
    """Abreviate a number to nearest thousand, million, etc.

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
    final_output = one_decimal if n > 1e6 and not one_decimal.is_integer() else int(final_num)

    return f'{final_output}{millnames[millidx]}'


# Used to calculate downloads for the last week
today = date.today()
last_week = today - timedelta(days=7)
DATE_FORMAT = '%Y-%m-%d'

# Open a session to save time (only allowed 1 second on Zapier)
session = requests.Session()

# Get the data for both the legacy namespace and our current one
pyramid_arima = json.loads(session.get('https://api.pepy.tech/api/projects/pyramid-arima').text)
pmdarima = json.loads(session.get('https://api.pepy.tech/api/projects/pmdarima').text)

# Sum up pmdarima and pyramid-arima downloads to the past week
pmdarima_downloads = 0
for i in range(7):
    pmdarima_downloads += pmdarima['downloads'][(last_week + timedelta(days=i)).strftime(DATE_FORMAT)]

pyramid_arima_downloads = 0
for i in range(7):
    pyramid_arima_downloads += pyramid_arima['downloads'][(last_week + timedelta(days=i)).strftime(DATE_FORMAT)]

# Millify the total and save in a dict
total = millify(pmdarima_downloads + pyramid_arima_downloads)
data = {
    'total': total
}

# Write to our storage area
# StoreClient is Zapier-specific and automatically imported (hence no import)
store = StoreClient(os.environ.get('ZAPIER_SHA'))
store.set('data', data)
