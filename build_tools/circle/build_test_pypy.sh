#!/bin/bash

# pypy container default python is python2.7, so need to source the virtualenv
# that is contained inside of the container
source /pypy-env/bin/activate

export LOKY_MAX_CPU_COUNT="2"  # for joblib parallelization
pip install -vv -e .

# Pytest is known to consume lots of memory for a large number of tests,
# and Circle 2.0 limits 4GB per container.
python -m pytest pmdarima/ -p no:logging --mpl --mpl-baseline-path=etc/pytest_images --benchmark-skip
