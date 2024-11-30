Setting up RocketPy as a Developer
==================================

Your first step to contribute to RocketPy is to set up your development environment.
If you are an experienced developer, you may skip to the final sections of this page.

Forking the repository
----------------------

First you need to fork the RocketPy repository. You can do this by clicking
the **Fork** button on the top right corner of the repository page on GitHub.

By default, the fork will only have the ``master`` branch.
You should also fork the ``develop`` branch.
You can do this by **disabling** the "Copy the master branch only" option when
forking.

.. important::

    Our ``master`` branch is the stable branch, and the ``develop`` branch is the \
    development branch. You should always create your PRs against the ``develop`` \
    branch, unless it is a really important hotfix.

Alternatively, one may clone the repository directly from the RocketPy repository
and then add the fork as a remote.
However, this option is less frequently used.

Cloning the Repository
----------------------

Next step is to clone the repository to your local machine.
There are different ways to do this, but most of them will involve the following command:

.. code-block:: bash

    git clone https://github.com/<Your-GitHub-Account>/RocketPy.git


After cloning the repository, you will have a copy of the RocketPy repository on your \
local machine and, by default, you will be on the ``master`` branch.

.. tip::

    Cloning a repo means downloading the repository to your local machine and \
    creating a local copy of it. We call this local copy a "local repository", \
    and the original repository is called the "remote repository".

    When you clone a repository, you are creating a connection between your local \
    repository and the remote repository. This connection allows you to push and \
    pull changes between the two repositories.

Navigating through the project
------------------------------

In order to work on your local repository, you will need to open the directory where you \
cloned it. We recommend using VS Code as your editor, but you can use any editor you prefer.
If you are using VS Code, you can open the project by running the following command:

.. code-block:: console

    code <path/to/your/folder>

After opening the project for the first time, you will be at the ``master`` branch.
To switch to another branch, such as the ``develop`` branch, you will need to fetch \
it from the remote repository and then checkout to it.

.. code-block:: console

    git fetch origin develop
    git checkout develop


Installation
------------

We highly recommend creating a python virtual environment first.
However, we will not describe how to do this here, since this is a common task.
Run the commands below inside Rocketpy folder to install the library and
development requirements:

.. code-block:: console

    pip install -e .  # install the rocketpy lib in editable mode
    pip install -r requirements-optional.txt  # install optional requirements
    pip install -r requirements-tests.txt  # install test/dev requirements

.. tip::

    When installing the ``rocketpy`` library, the requirements listed in the \
    ``requirements.txt`` file will automatically be installed.

Running the tests
-----------------

One simple sanity check you can do after installing rocketpy is to run the unit and integration tests.
One may achieve this by running the following commands:

Running all tests:

.. code-block:: console

    make pytest

Running the slow tests only:

.. code-block:: console

    make pytest-slow

Creating a .html coverage report, where you could see the coverage of the tests:

.. code-block:: console

    make coverage-report


.. note::

    The slow tests are the tests marked with the ``@pytest.mark.slow`` decorator. \
    These tests are usually the ones that take longer to run, and therefore are \
    not run by default. More about the tests can be found in the \
    :ref:`testing_guidelines` section
