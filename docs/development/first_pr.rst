Your first Pull Request (PR)
============================

This page describes a few steps to help you get started with your first PR to RocketPy.
We are excited to have you here!

Picking an issue
----------------

Before you start coding, you should pick an issue to work on. You can find a
list of issues in the `Issues`_ tab on GitHub.
.. _Issues: https://github.com/RocketPy-Team/RocketPy/issues
If you are new hear, it is really recommended that you talk to maintainers
before starting to work on an issue.
That way you avoid working on something that is already being worked on or
something that is not in line with the project's goals.

If you want to develop a new feature that is not in the issues list, please open
a new issue to discuss it with the maintainers before submitting a PR.

Once the issue is assigned to you, you can start working on it.



Creating a new branch
---------------------

At your local machine, ...

Opening the PR
--------------

When you open a PR, you should:

* Use labels to help maintainers understand what the PR is about.
* Link any issue that may be closed when the PR is merged.


Remember, the PR is yours, not ours! You should keep track of it and update it as needed.


Continuous Integration (CI)
---------------------------

There are several automation on our repository to help us maintain the code quality.

Currently, our CI pipeline runs the following checks:

* **Linting**: we run `flake8`, `pylint`, `black` and `isort` on top of your latest commit in order to check for code style issues. To understand more about these tools, please read ...
* **Testing**: we run the tests defined in the `tests` folder to make sure your changes do not break any existing functionality. The tests will be executed 6 times, each time with a different Python version (the oldest and newest supported version) and with three different operating systems (Windows, Linux and MacOS).
* **Coverage**: based on the tests results, we also check the code coverage. There is an automation to check if the code coverage increased or decreased with your PR. It also points

Once you open your PR or commit and push to your branch, the CI will be initialized.
Please correct any issues that may arise from the CI checks.

.. note::

    All the commands we run in the CI pipeline can also be run locally. It is important \
    that you run the checks locally before pushing your changes to the repository.

The CHANGELOG file
------------------

We keep track of the changes in the `CHANGELOG.md` file. When you open a PR, you should add a new entry to the `Unreleased` section of the file. This entry should simply be the title of your PR.

.. note::

    In the future we would like to automate the CHANGELOG update, but for now \
    it is a manual process, unfortunately.

