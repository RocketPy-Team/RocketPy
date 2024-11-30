Pro Tips
========

Last but not least, here are some pro tips that can help you to be more
productive when developing in the project (or any other project).


Makefile
--------

We have a ``Makefile`` in the repository that contains some useful commands to
help you with the development process.

Some examples of commands are:

* ``make black``: runs the black formatter on the code.
* ``make format``: runs the isort and black formatters on the code.
* ``make lint``: runs the flake8 and pylint tools.
* ``make build-docs``: This will build the documentation locally, so you can check if everything is working as expected.
* ``make pytest-slow``: runs only the slow tests (marked with the ``@pytest.mark.slow`` decorator).

These commands are meant to be system agnostic, so you can run them on any
Operational System (OS).

Knowing your editor
-------------------

Something we usually notice on beginners is that they don't know all the power
their editor can provide. Knowing your editor can save you a lot of time and
make you more productive.

As stated earlier, we recommend using VS Code as your editor, as most of the
RocketPy developers use it. Therefore it is easier to help you if you are using
the same editor.

Some of the features that can help you are:

1. **Code navigation**: Jump to definitions, find references, etc.
2. **Integrated tests**: Run tests directly from the editor.
3. **Python Debugger**: Debug your code directly from the editor.
4. **GitHub Pull Requests**: Open, review, comment, and merge pull requests directly from the editor

Extensions
----------

We have listed some recommended extensions in the ``.vscode/extensions.json`` file.
These are general recommendations based on what our developers usually install
in their VSCode.
Obviously, it is not mandatory to install them, but they can help you to be more
productive.


Code assistance
---------------

Artificial Intelligence (AI) assistance has becoming more and more common in
software development.
Some editors have AI assistance built-in.
Famous options are GitHub Copilot, JetBrains AI and TabNine.

At this repo, the use of AI tools is welcome, we don't have any restrictions
against it.

A few possible applications of AI tools are:

* Writing documentation.
* Writing tests.
* Getting explanations of code sections.

.. tip::

    As of today, November 2024, GitHub Copilot still provides free access for \
    university email addresses. We can't guarantee this will still be the case \
    when you are reading this, so check the GitHub Copilot website for more \
    information.


If you are against the use of AI tools, do not worry, you can still contribute
to the project without using them.


Engaging with the community
---------------------------

The most important part of contributing to an open-source project is engaging
with the community.
Our developers are frequently available on our server, and you can
ask any questions you may have there.
Either a question about the project, or a question about how to contribute, or
even a suggestion of a new feature.
Any kind of interaction is welcome.


.. tip::

    The official supported language in our server is English. \
    However, the RocketPy Team is lucky to have developers from all around the \
    world, so you may find people speaking other languages as well. \
    Don't let the language barrier keep you from engaging with the community.
