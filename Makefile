# simple makefile to simplify repetitive build env management tasks under posix
# this is adopted from the sklearn Makefile

# caution: testing won't work on windows

PYTHON ?= python

.PHONY: clean develop test install bdist_wheel version

clean:
	$(PYTHON) setup.py clean
	rm -rf dist
	rm -rf build
	rm -rf .pytest_cache
	rm -rf pmdarima.egg-info
	rm -f pmdarima/VERSION
	rm -rf .coverage.*

deploy-requirements:
	$(PYTHON) -m pip install twine readme_renderer[md]

# Depends on an artifact existing in dist/, and two environment variables
deploy-twine-test: bdist_wheel deploy-requirements
	$(PYTHON) -m twine upload \
		--repository-url https://test.pypi.org/legacy/ dist/* \
		--username ${TWINE_USERNAME} \
		--password ${TWINE_PASSWORD}

doc-requirements:
	$(PYTHON) -m pip install -r build_tools/doc/doc_requirements.txt

documentation: doc-requirements version
	@make -C doc clean html EXAMPLES_PATTERN=example_*

requirements:
	$(PYTHON) -m pip install -r requirements.txt

bdist_wheel: version
	$(PYTHON) setup.py bdist_wheel

sdist: version
	$(PYTHON) setup.py sdist

develop: version
	$(PYTHON) setup.py develop

install: version
	$(PYTHON) setup.py install

test-requirements:
	$(PYTHON) -m pip install pytest flake8 matplotlib pytest-mpl pytest-benchmark

coverage-dependencies:
	$(PYTHON) -m pip install coverage pytest-cov codecov

test-lint: test-requirements
	$(PYTHON) -m flake8 pmdarima --filename='*.py' --ignore E803,F401,F403,W293,W504

test-unit: test-requirements coverage-dependencies
	$(PYTHON) -m pytest -v --durations=20 --cov-config .coveragerc --cov pmdarima -p no:logging --benchmark-skip

test-benchmark: test-requirements coverage-dependencies
	$(PYTHON) -m pytest -v --durations=12 --cov-config .coveragerc --cov pmdarima -p no:logging --benchmark-min-rounds=5 --benchmark-min-time=1 --benchmark-only

test: develop test-unit test-lint
	# Coverage creates all these random little artifacts we don't want
	rm .coverage.* || echo "No coverage artifacts to remove"

twine-check: bdist_wheel deploy-requirements
	# Check that twine will parse the README acceptably
	$(PYTHON) -m twine check dist/*

version: requirements
	@$(PYTHON) build_tools/get_tag.py
