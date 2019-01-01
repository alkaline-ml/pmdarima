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

.. _austres:

Austres
-------

Numbers (in thousands) of Australian residents measured quarterly from
March 1971 to March 1994. The sample consists of 89 records on a quarterly basis.

.. code-block:: python

    >>> load_austres(True).head()
    0    13067.3
    1    13130.5
    2    13198.4
    3    13254.2
    4    13303.7
    dtype: float64

.. _heartrate:

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

.. _lynx:

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

.. _wineind:

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

.. _woolyrnq:

Woolyrnq
--------

A time-series that records the quarterly production (in tonnes) of woollen
yarn in Australia between Mar 1965 and Sep 1994. This dataset is found in the
R ``forecast`` package.

.. code-block:: python

    >>> load_woolyrnq(True).head()
    Q1 1965    6172
    Q2 1965    6709
    Q3 1965    6633
    Q4 1965    6660
    Q1 1966    6786
    dtype: int64
