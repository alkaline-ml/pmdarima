# simple makefile to simplify repetitive build env management tasks under posix
# this is adopted from the sklearn Makefile

# caution: testing won't work on windows, see README

PYTHON ?= python
CYTHON ?= cython
NOSETESTS ?= nosetests
CTAGS ?= ctags

# skip doctests on 32bit python
BITS := $(shell python -c 'import struct; print(8 * struct.calcsize("P"))')

ifeq ($(BITS),32)
  NOSETESTS:=$(NOSETESTS) -c setup32.cfg
endif


all: clean inplace #test

clean-ctags:
	rm -f tags

clean: clean-ctags
	$(PYTHON) setup.py clean
	rm -rf dist

in: inplace # just a shortcut
inplace:
	$(PYTHON) setup.py build_ext -i

test-code: in
	$(NOSETESTS) -s -v pyramid

# test-sphinxext:
#	$(NOSETESTS) -s -v doc/sphinxext/
#test-doc:
#ifeq ($(BITS),64)
#	$(NOSETESTS) -s -v doc/*.rst doc/modules/ doc/datasets/ \
#	doc/developers doc/tutorial/basic doc/tutorial/statistical_inference \
#	doc/tutorial/text_analytics
#endif

test-coverage:
	rm -rf coverage .coverage
	$(NOSETESTS) -s -v --with-coverage pyramid

#test: test-code test-sphinxext test-doc

trailing-spaces:
	find python -name "*.py" -exec perl -pi -e 's/[ \t]*$$//' {} \;

cython:
	python setup.py build_src

ctags:
	# make tags for symbol based navigation in emacs and vim
	# Install with: sudo apt-get install exuberant-ctags
	$(CTAGS) --python-kinds=-i -R pyramid

#doc: inplace
#	$(MAKE) -C doc html

#doc-noplot: inplace
#	$(MAKE) -C doc html-noplot

code-analysis:
	flake8 pyramid | grep -v __init__ | grep -v external
	pylint -E -i y pyramid/ -d E1103,E0611,E1101

#flake8-diff:
#./build_tools/travis/flake8_diff.sh
