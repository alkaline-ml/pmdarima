# simple makefile to simplify repetitive build env management tasks under posix
# this is adopted from the sklearn Makefile

# caution: testing won't work on windows

PYTHON ?= python

.PHONY: clean develop test install bdist_wheel

clean:
	$(PYTHON) setup.py clean
	rm -rf dist
	rm -rf build

doc-requirements:
	$(PYTHON) -m pip install -r build_tools/doc/doc_requirements.txt

requirements:
	$(PYTHON) -m pip install -r requirements.txt

bdist_wheel: requirements
	$(PYTHON) setup.py bdist_wheel

develop: requirements
	$(PYTHON) setup.py develop

install: requirements
	$(PYTHON) setup.py install

test-requirements:
	$(PYTHON) -m pip install pytest flake8 matplotlib pytest-mpl

coverage-dependencies:
	$(PYTHON) -m pip install coverage pytest-cov codecov

test-lint: test-requirements
	$(PYTHON) -m flake8 pmdarima --filename='*.py' --ignore E803,F401,F403,W293,W504

test-unit: test-requirements coverage-dependencies
	$(PYTHON) -m pytest -v --durations=20 --mpl --mpl-baseline-path=pytest_images --cov-config .coveragerc --cov pmdarima -p no:logging

test: develop test-unit test-lint
	# Coverage creates all these random little artifacts we don't want
	rm .coverage.* || echo "No coverage artifacts to remove"

twine-check: bdist_wheel
	$(PYTHON) -m twine check dist/*
