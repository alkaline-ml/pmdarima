#!/bin/bash

# Should be 3.6.xxx. If 2.7, PyPy build was screwed up
python --version

export LOKY_MAX_CPU_COUNT="2"  # for joblib parallelization
python -m pip install -vv -e .

# Pytest is known to consume lots of memory for a large number of tests,
# and Circle 2.0 limits 4GB per container.
python -m pytest pmdarima/ -p no:logging --mpl --mpl-baseline-path=etc/pytest_images --benchmark-skip
