.. _datasets:

========================
Toy time-series datasets
========================

The datasets submodule provides an interface for loading various built-in toy
time-series datasets, some of which are datasets commonly used for benchmarking
time-series models or are pre-built in R.

All datasets share a common interface::

    load_<some_dataset>(as_series=False)

Where ``as_series=True`` will return a Pandas Series object with the appropriate index.

Lynx
----

The Lynx dataset records the number of skins of predators (lynx) that were
collected over many years by the Hudson's Bay Company (1821 - 1934). It's
commonly used for time-series benchmarking (Brockwell and Davis - 1991) and is
built into R. The dataset exhibits a clear 10-year cycle.

.. code-block:: python

    >>> load_lynx(True).head()
    1821     269
    1822     321
    1823     585
    1824     871
    1825    1475
    dtype: int64

Wineind
-------

This time-series records total wine sales by Australian wine makers in
bottles <= 1 litre between Jan 1980 -- Aug 1994. This dataset is found in the
R ``forecast`` package.

.. code-block:: python

    >>> load_wineind(True).head()
    Jan 1980    15136
    Feb 1980    16733
    Mar 1980    20016
    Apr 1980    17708
    May 1980    18019
    dtype: int64

Heartrate
---------

The heart rate data records sample of heartrate data borrowed from an
`MIT database <http://ecg.mit.edu/time-series/>`_. The sample consists
of 150 evenly spaced (0.5 seconds) heartrate measurements.

.. code-block:: python

    >>> load_heartrate(True).head()
    0    84.2697
    1    84.2697
    2    84.0619
    3    85.6542
    4    87.2093
    dtype: float64
