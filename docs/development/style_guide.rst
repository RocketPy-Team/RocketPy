Style Guide
===========

This page describes code, documentation and git branching styles used throughout RocketPy's development.

Code Style
----------

We mostly follow the standard Python style conventions as described here: `Style Guide for Python Code <https://www.python.org/dev/peps/pep-0008/>`_

Naming Conventions
^^^^^^^^^^^^^^^^^^
RocketPy was, unfortunately, initially coded using `Camel Case <https://en.wikipedia.org/wiki/Camel_case>`_.
However, PEP8 specifies the following::

    Function names should be lowercase, with words separated by underscores as necessary to improve readability.
    Variable names follow the same convention as function names.

    mixedCase is allowed only in contexts where that is already the prevailing style (e.g. threading.py), to retain backwards compatibility.

Therefore, `Snake Case <https://en.wikipedia.org/wiki/Snake_case>`_ is preferred.
For this reason, RocketPy is being converted from Camel to Snake and any new contributions should strive to follow Snake case as well.

Furthermore, when it comes to naming new variables, functions, classes or anything in RocketPy, always try to use descriptive names.

As an example, instead of using `a` or `alpha` as a variable for a rocket's angle of attack, `angle_of_attack` is preferred.
Such descriptive names make the code significantly easier to understand, review and maintain.

In summary:

.. code-block:: python

    # Not acceptable
    a = 0.2 # angle of attack

    # Preferred
    angle_of_attack = 0.2 # in rad


Linting
^^^^^^^
As far as line wrapping, parentheses, calling chains and other code style related matter goes, RocketPy currently employs `Black style <https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html>`_.
Installing Black and running it before submitting a pull request is highly recommend.
Currently, there are pull request tests in place to enforce that Black style is used.


Documentation Style
-------------------

Every class, method, attribute and function added to RocketPy should be well documented using docstrings.
RocketPy follows NumPy style docstrings, which are very well explained here: `NumPyDoc Style Guide <https://numpydoc.readthedocs.io/en/latest/format.html>`_.

Here is a short example of the simplest acceptable docstring:


Git Style
---------

Branch Naming Conventions
^^^^^^^^^^^^^^^^^^^^^^^^^

Under construction.

Commit Messages
^^^^^^^^^^^^^^^

Commit messages should be clear and follow a few basic rules.  Example::

   ENH: add functionality X to rocketpy.<submodule>.

   The first line of the commit message starts with a capitalized acronym
   (options listed below) indicating what type of commit this is.  Then a blank
   line, then more text if needed.  Lines shouldn't be longer than 72
   characters.  If the commit is related to an issue, indicate that with
   "See #3456", "See issue 3456", "Closes #3456" or similar.

Describing the motivation for a change, the nature of a bug for bug fixes or
some details on what an enhancement does are also good to include in a commit
message.  Messages should be understandable without looking at the code
changes.  A commit message like ``MAINT: fixed another one`` is an example of
what not to do; the reader has to go look for context elsewhere.

Standard acronyms to start the commit message with are::

   BLD: change related to building RocketPy
   BUG: bug fix
   DEP: deprecate something, or remove a deprecated object
   DEV: development tool or utility
   DOC: documentation
   ENH: enhancement
   MAINT: maintenance commit (refactoring, typos, etc.)
   REV: revert an earlier commit
   STY: style fix (whitespace, PEP8)
   TST: addition or modification of tests
   REL: related to releasing RocketPy
