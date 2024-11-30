Your first Pull Request (PR)
============================

This page describes a few steps to help you get started with your first PR to RocketPy.
We are excited to have you here!

Picking an issue
----------------

Before you start coding, you should pick an issue to work on. You can find a
list of issues in the ``Issues`` tab on GitHub (https://github.com/RocketPy-Team/RocketPy/issues).
If you are new here, it is really recommended that you talk to maintainers
before starting to work on an issue.
That way you avoid working on something that is already being worked on or
something that is not in line with the project's goals.

If you want to develop a new feature that is not in the issues list, please open
a new issue to discuss it with the maintainers before submitting a PR.

Once the issue is assigned to you, you can start working on it.

.. note::

    In order to open an issue at our repo, you must have a GitHub account. \
    In case you do not want to open an account yet, you can contact the maintainers \
    through our Discord server. But keep in mind that you you need an account to \
    open a Pull Request (PR).

Creating a new branch
---------------------

At your local machine, make sure you already have `git <https://git-scm.com/>`_ \
installed and set up.
In order to create a new branch, you should run the following command in your
preferred terminal:

.. code-block:: console

    git checkout -b <branch_name>

This command will create a new branch named ``branch_name`` from the current branch you are in.
Your branch name should follow the guidelines described in :doc:`/development/style_guide`.

.. tip::

    VS Code has a built-in integration with git, this allows you to run git commands \
    quite easily through the editor's interface. For example, you could search for \
    "git checkout" in the command palette and run the command from there.


Opening the PR
--------------

After you finish your work, you are more than welcome opening a PR to the RocketPy repository.
Here are some checks to do before opening a PR:

* Check if the test suite is passing.
* Format your code using ``black`` and ``isort``.

Check the :doc:`/development/pro_tips` section for more information on how to run these commands.

When you open a PR, you should:

* Provide a clear and concise title.
* Fill the PR description following the standard template.
* Use labels to help maintainers understand what the PR is about.
* Link any issue that may be closed when the PR is merged.

Remember, the PR is yours, not ours! You should keep track of it and update it as needed.

.. important::

    See the :doc:`/development/style_guide` for more information on how to name and \
    describe your PR.

Continuous Integration (CI)
---------------------------

There are several automation on our repository to help us maintain the code quality.
Currently, our CI pipeline runs the following checks:

* **Linting**: we run ``flake8``, ``pylint``, ``black`` and ``isort`` on top of your latest commit in order to check for code style issues. To understand more about these tools, please read the :doc:`/development/pro_tips` section.
* **Testing**: we run the tests defined in the ``tests`` folder to make sure your changes do not break any existing functionality. The tests will be executed 6 times, each time with a different Python version (the oldest and newest supported versions) and with three different operating systems (Windows, Linux and MacOS).
* **Coverage**: based on the tests results, we also check the code coverage results from our test suite. There is an automation to check if the code coverage increased or decreased with your PR. It also points to potential introduced lines of code that should be tested.

Once you open your PR or commit and push to your branch, the CI will be initialized.
Please correct any issues that may arise from the CI checks.

.. important::

    All the commands we run in the CI pipeline can also be run locally. It is \
    important that you run the checks locally before pushing your changes to \
    the repository.

The CHANGELOG file
------------------

We keep track of the changes in the ``CHANGELOG.md`` file.
When you open a PR, you should add a new entry to the "Unreleased" section of the file.
This entry should simply be the title of your PR.

.. note::

    In the future we would like to automate the CHANGELOG update, but for now \
    it is a manual process, unfortunately.


The review process
------------------

After you open a PR, the maintainers will review your code.
This review process is a way to ensure that the code is in line with the
project's goals and that it is well written and documented.

The maintainers may ask you to make changes to your code.
You should address these changes or explain why you think they are not necessary.
This is the best time to learn from the maintainers and improve your coding skills.

In case you do not address the comments in a timely manner, the maintainers may
either close the PR or make the changes themselves.


Merging the PR
--------------

There are 3 different ways of merging a PR:

1. **Create a merge commit**: this is the default option on GitHub.
2. **Squash and merge**: this option will squash all your commits into a single one. This is useful when you have many commits and you want to keep the history clean, therefore this is the recommended option.
3. **Rebase and merge**: this option will add your commits directly to the target branch, without creating a merge commit. This is useful to keep the history linear, however it also requires handling potential conflicts one at a time, which can be a bit more complex.

.. note::

    Overall, you will not have permission to merge your PR. The maintainers will \
    take care of that for you. This is here just for you to understand the process.

All in all, there is no right or wrong way to merge a PR.
The maintainers will decide which option is the best for the project.
What you should care though is about conflicting changes, let's talk about that next in :doc:`/development/conflicts`.
