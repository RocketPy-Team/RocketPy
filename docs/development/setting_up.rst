Setting Up RocketPy as a Developer
==================================


Forking the repository
----------------------

The first step is to fork the RocketPy repository. You can do this by clicking
the **Fork** button on the top right corner of the repository page on GitHub.

By default, the fork will only have the `master` branch. You should also fork
the `develop` branch. You can do this by going to the `develop` branch on the
RocketPy repository before clicking the **Fork** button.

.. IMPORTANT: this part is usually not so clear.

.. important::

    Our `master` branch is the stable branch, and the `develop` branch is the \
    development branch. You should always create your PRs against the `develop` \
    branch, unless it is a really important hotfix.


Cloning the Repository
----------------------


Installation
------------


pip install -e .
pip install -r requirements-optional.txt
pip install -r requirements-tests.txt


Navigating through the project
------------------------------

git checkout ...


Running a flight simulation
---------------------------

Please see the other page... (link the other page)

Running the tests
-----------------

make pytest

make pytest-slow

make coverage-report

