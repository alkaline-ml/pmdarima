.. _contrib:

=======================
Contributing to Pyramid
=======================

**Note: This document is a 'getting started' summary for contributing code,
documentation, testing, and filing issues.** Please read it carefully to help
make the code review process go as smoothly as possible and maximize the
likelihood of your contribution being merged.

How to contribute
-----------------

The preferred workflow for contributing to pyramid is to fork the
`main repository <https://github.com/tgsmith61591/pyramid>`_ on
Github, clone, and develop on a branch. Steps:

1. Fork the `project repository <https://github.com/tgsmith61591/pyramid>`_
   by clicking on the 'Fork' button near the top right of the page. This
   creates a copy of the code under your Github user account.

2. Clone your fork of the pyramid repo from your Github account to your
   local disk:

   .. code-block:: bash

       $ git clone https://github.com/tgsmith61591/pyramid.git
       $ cd pyramid

3. Create a ``feature`` branch to hold your development changes:

   .. code-block:: bash

       $ git checkout -b my-feature

   Always use a ``feature`` branch. It's good practice to never work on the ``master`` branch!

4. Develop the feature on your feature branch. Add changed files using ``git add`` and then ``git commit`` files:

   .. code-block:: bash

       $ git add modified_files
       $ git commit

   to record your changes in Git, then push the changes to your Github account with:

   .. code-block:: bash

       $ git push -u origin my-feature

5. Follow `these instructions <https://help.github.com/articles/creating-a-pull-request-from-a-fork>`_
to create a pull request from your fork. This will send an email to the committers.

Pull Request Checklist
----------------------

We recommended (and prefer that) that your contribution complies with the
following rules before you submit a pull request. Failure to adhere to the
rules may hinder the speed with which your contribution is merged:

-  Pyramid uses the `gitflow branching model <http://nvie.com/posts/a-successful-git-branching-model/>`_.
   That means all of your feature branches should be merged back to the `develop`
   branch, and *not* `master`!

-  Write detailed docstrings for all of public your functions. The preferred
   format for docstrings is the `numpy standard <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt#docstring-standard>`_.
   Also include usage examples where appropriate. See also the
   `Numpy guidelines for documenting your code  <https://numpydoc.readthedocs.io/en/latest/>`_

-  Use, when applicable, the validation tools and scripts in the
   `pyramid.utils` submodule.

-  Give your merge request a helpful title that summarizes what your
   contribution does. In some cases ``Fix <ISSUE TITLE>`` is enough.
   ``Fix #<ISSUE NUMBER>`` is not enough.

-  If your pull request references an issue, reference it in the body of your
   descriptive text using ``#<ISSUE NUMBER>``

-  Please prefix the title of your pull request with ``[MRG]`` (Ready for
   Merge), if the contribution is complete and ready for a detailed review.
   The core developers will then review your code and merge when approved.
   An incomplete contribution -- where you expect to do more work before
   receiving a full review -- should be prefixed ``[WIP]`` (to indicate a work
   in progress) and changed to ``[MRG]`` when it matures. WIPs may be useful
   to: indicate you are working on something to avoid duplicated work,
   request broad review of functionality or API, or seek collaborators.

-  All other tests pass when everything is rebuilt from scratch. Note that this
   will actually require a Spark distribution to work locally.
   On Unix-like systems, check with (from the toplevel source folder):

      .. code-block:: bash

          $ python setup.py develop
          $ pytest

   You may need to see the :ref:`setup` section for instructions on how
   to build the package. For instructions on how to test (using nose or pytest)
   see `Numpy's testing instructions <https://github.com/numpy/numpy/blob/master/doc/TESTS.rst.txt>`_.

Filing bugs
-----------
We use Github issues to track all bugs and feature requests; feel free to
open an issue if you have found a bug or wish to see a feature implemented.

It is recommended to check that your issue complies with the
following rules before submitting:

-  Verify that your issue is not being currently addressed by other
   `issues <https://github.com/tgsmith61591/pyramid/issues>`_
   or `pull requests <https://github.com/tgsmith61591/pyramid/pulls>`_.

-  If your issue references and pull request, reference it in the body of your
   descriptive text using ``!<PULL REQUEST NUMBER>``

-  Please include your operating system type and version number, as well
   as your Python, scikit-learn, numpy, scipy, pandas and pyramid versions. This
   information can be found by running the following code snippet:

  .. code-block:: python

      import platform; print(platform.platform())
      import sys; print("Python", sys.version)
      import numpy; print("NumPy", numpy.__version__)
      import scipy; print("SciPy", scipy.__version__)
      import sklearn; print("Scikit-Learn", sklearn.__version__)
      import pandas; print("Pandas", pandas.__version__)
      import statsmodels; print("Statsmodels", statsmodels.__version__)
      import pyramid; print("Pyramid", pyramid.__version__)
