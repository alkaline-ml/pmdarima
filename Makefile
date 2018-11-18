# simple makefile to simplify repetitive build env management tasks under posix
# this is adopted from the sklearn Makefile

# caution: testing won't work on windows

PYTHON ?= python

.PHONY: clean develop test

clean:
	$(PYTHON) setup.py clean
	rm -rf dist
	rm -rf build

doc-requirements:
	$(PYTHON) -m pip install -r build_tools/doc/doc_requirements.txt

requirements:
	$(PYTHON) -m pip install -r requirements.txt

develop: requirements
	$(PYTHON) setup.py develop

install: requirements
	$(PYTHON) setup.py install

# test-sphinxext:
#	$(NOSETESTS) -s -v doc/sphinxext/
#test-doc:
#ifeq ($(BITS),64)
#	$(NOSETESTS) -s -v doc/*.rst doc/modules/ doc/datasets/ \
#	doc/developers doc/tutorial/basic doc/tutorial/statistical_inference \
#	doc/tutorial/text_analytics
#endif

test-requirements:
	$(PYTHON) -m pip install pytest pytest-cov flake8

test-lint: test-requirements
	$(PYTHON) -m flake8 pmdarima --filename='*.py' --ignore E803,F401,F403,W293,W504

test-unit: test-requirements
	$(PYTHON) -m pytest -v --durations=20 --cov-config .coveragerc --cov pmdarima

test: develop test-unit test-lint
