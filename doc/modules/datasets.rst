.. _datasets:

========================
Toy time-series datasets
========================

The datasets submodule provides an interface for loading various built-in toy
time-series datasets, some of which are datasets commonly used for benchmarking
time-series models or are pre-built in R.

Lynx
----

The Lynx dataset records the number of skins of predators (lynx) that were
collected over many years by the Hudson's Bay Company (1821 - 1934). It's
commonly used for time-series benchmarking (Brockwell and Davis - 1991) and is
built into R. The dataset exhibits a clear 10-year cycle.

Wineind
-------

This time-series records total wine sales by Australian wine makers in
bottles <= 1 litre between Jan 1980 -- Aug 1994. This dataset is found in the
R ``forecast`` package.

Heartrate
---------

The heart rate data records sample of heartrate data borrowed from an
`MIT database <http://ecg.mit.edu/time-series/>`_. The sample consists
of 150 evenly spaced (0.5 seconds) heartrate measurements.
