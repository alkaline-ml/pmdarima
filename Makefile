# simple makefile to simplify repetitive build env management tasks under posix
# this is adopted from the sklearn Makefile

# caution: testing won't work on windows

PYTHON ?= python
DOCKER ?= docker
HERE = $(shell pwd)

.PHONY: clean develop test install bdist_wheel

clean:
	$(PYTHON) setup.py clean
	rm -rf dist
	rm -rf build
	rm -rf .pytest_cache
	rm -rf pmdarima.egg-info
	rm -f conda/meta.yaml
	rm -rf .coverage.*

deploy-requirements:
	$(PYTHON) -m pip install twine readme_renderer[md]

# Depends on an artifact existing in dist/, and two environment variables
deploy-twine-test: bdist_wheel deploy-requirements
	$(PYTHON) -m twine upload \
		--repository-url https://test.pypi.org/legacy/ dist/* \
		--username ${TWINE_USERNAME} \
		--password ${TWINE_PASSWORD}

documentation:
	$(DOCKER) run -v $(HERE):/pmdarima -w /pmdarima --rm alkalineml/pmdarima-doc-base:latest /bin/bash -c "make install docker-documentation"

# This one assumes we are in the docker container, so it can either be called from above (locally), or directly (on CI)
docker-documentation:
	@make -C doc clean html EXAMPLES_PATTERN=example_* PMDARIMA_VERSION=$(PMDARIMA_VERSION)

requirements:
	$(PYTHON) -m pip install ".[all]"

bdist_wheel:
	$(PYTHON) -m build --wheel

sdist:
	$(PYTHON) -m build --sdist

develop:
	$(PYTHON) -m pip install --editable . --no-build-isolation

install:
	$(PYTHON) -m pip install .

lint-requirements:
	$(PYTHON) -m pip install flake8

testing-requirements:
	$(PYTHON) -m pip install ".[test]" flake8 coverage pytest-cov codecov

test-lint:
	$(PYTHON) -m flake8 pmdarima --filename='*.py' --ignore F401,F403,W293,W504

test-unit:
	$(PYTHON) -m pytest -v --durations=20 --cov-config .coveragerc --cov pmdarima -p no:logging

test: test-unit test-lint
	# Coverage creates all these random little artifacts we don't want
	rm .coverage.* || echo "No coverage artifacts to remove"

twine-check: bdist_wheel deploy-requirements
	# Check that twine will parse the README acceptably
	$(PYTHON) -m twine check dist/*
