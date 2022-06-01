.. _exog_to_X:

===========================
RFC: ``exogenous`` -> ``X``
===========================

This RFC proposes the renaming of the ``exogenous`` arg to ``X``. While this would
impact the public API, we would allow the current ``exogenous`` argument to persist
for several minor release cycles with a deprecation warning before completely removing it
in the next major release (2.0).

Why?
----

* **It's typo-prone**. We've received several issues lately with people asking why the ``exogenous``
  argument was not doing anything. Upon close inspection, it was evident they were misspelling the
  arg as "exogeneous", and the presence of ``**kwargs`` in the function signature allowed
  the argument through without raising a ``TypeError``.

* **It's clunky**. Typing ``exogenous`` when other APIs have simplified this to the ubiquitous
  ``X`` used in other scikit-style packages (scikit-learn, scikit-image, sktime) seems like
  a slightly annoying, arbitrary difference in signature definitions that keeps us from
  matching the signatures of other similar packages.

* **It can be confusing**. Not all of our user base is familiar with the classical statistics
  terminology and may not realize what this argument permits them. Conversely, nearly all
  users are familiar with the idea of what ``X`` allows them.

How?
----

For a while, we'd allow the ``exogenous`` argument to be passed in ``**kwargs``, and would simply
warn if it were present. For example:

.. code-block:: python

    def fit(self, y, X=None, **kwargs):
        if X is None:
            X = kwargs.pop("exogenous", None)
            if X is not None:
                warnings.warn("`exogenous` is deprecated and will raise an error "
                              "in version 2.0 - Use the `X` arg instead",
                              DeprecationWarning)

This would ensure backwards compatibility for several minor release cycles before the
change was made, and would give sufficient time to users to switch over to the new naming scheme.

Precedent
---------

Scikit-learn has made similar package naming decisions in the name of package consistency and ubiquity,
notably in migrating the ``cross_validation`` namespace to the ``model_selection`` namespace in version
0.18. This was preceded by several minor releases that warned on imports.
