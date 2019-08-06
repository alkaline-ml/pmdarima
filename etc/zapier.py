"""
This code is meant to be run on Zapier

It is used to add the `total downloads` badge to our README
"""
import json
import math
import os
import requests


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

    return '{:.0f}{}'.format(n / 10 ** (3 * millidx), millnames[millidx])


# Open a session to save time (only allowed 1 second on Zapier)
session = requests.Session()

# Get the data for both the legacy namespace and our current one
pyramid_arima = json.loads(session.get('https://api.pepy.tech/api/projects/pyramid-arima').text)
pmdarima = json.loads(session.get('https://api.pepy.tech/api/projects/pmdarima').text)

# Millify the total and save in a dict
total = millify(pyramid_arima['total_downloads'] + pmdarima['total_downloads'])
data = {
    'total': total
}

# Write to our storage area
# StoreClient is Zapier-specific and automatically imported (hence no import)
store = StoreClient(os.environ.get('ZAPIER_SHA'))
store.set('data', data)
