Solving Git Conflicts
=====================

What Are Git Conflicts?
-----------------------

Git conflicts occur when Git is unable to automatically merge changes made to the \
same lines of code or when changes overlap in a file across branches.
This often occurs in RocketPy repository when two developers make changes to the same \
file and try to merge them.
In this scenario, Git pauses the operation and marks the conflicting files, \
requiring manual intervention.

Merge vs Rebase
---------------

**Merge**

The ``merge`` operation combines the changes from one branch into another. \
It creates a new commit that marks the merging of the two branches together.

* Retains the full history of both branches.
* Non-linear history shows how changes were combined.
* The history can become cluttered if there are many branches, due to the number of merge commits.

**Rebase**

The ``rebase`` operation integrates changes from one branch into another by reapplying \
the commits on top of the target branch. It results in a more linear history.

* Produces a cleaner, linear commit history.
* Easier to follow the sequence of changes.
* Can rewrite history, leading to potential issues when working on shared branches.

Example Commands
----------------

Let's consider a common scenario from which conflicts can arise:
updating a feature branch with the latest changes from the branch it was created.
Both ``merge`` and ``rebase`` can be used to update the feature branch.
However, the ``rebase`` option is highly recommended to keep a more linear history.

Merge
~~~~~

Suppose you have two branches, ``enh/feature`` that was branched off ``develop``.
It is likely that ``develop`` has been updated since you branched off ``enh/feature``, \
therefore before merging ``enh/feature`` into ``develop``, you should update ``enh/feature`` \
with the latest changes from ``develop``.

One way to do this is by merging ``develop`` into ``enh/feature`` as follows:

.. code-block:: console

    git checkout develop
    git pull
    git checkout enh/feature
    git merge develop

If there are conflicts, Git will pause the operation and mark the conflicting files. \
VS Code provides a visual interface to resolve these conflicts in the **Merge Editor**.
If you want to abort the merge, you can use ``git merge --abort``.

Rebase
~~~~~~

Similarly, another way to update ``enh/feature`` with the latest changes from ``develop`` \
is by rebasing ``enh/feature`` onto ``develop`` as follows:

.. code-block:: console

    git checkout develop
    git pull
    git checkout enh/feature
    git rebase develop

Differently from merge, if there are conflicts, they must be resolved commit by commit. \
After resolving each step conflicts, you can continue the rebase with ``git rebase --continue``. \
If you want to abort the rebase, you can use ``git rebase --abort``.

When to Use Merge or Rebase
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generally, the maintainers will inform you which operation to use when merging your PR. \
Notice that there is no wrong way to merge branches, but ``rebase`` is usually preferred \
to keep a more linear history.

Furthermore, it is important to avoid letting conflicts accumulate, since they can become \
harder to resolve. It is recommended to update your feature branch frequently with the latest \
changes from the branch it was branched off.

Solving Conflicts
-----------------

When conflicts arise, Git marks the conflicting files. The file will contain markers like:

.. code-block:: console

    <<<<<<< HEAD
    Current branch changes
    =======
    Incoming branch changes
    >>>>>>> branch-name

The ``HEAD`` section contains the changes from the current branch, while the ``branch-name`` section \
contains the changes from the incoming branch.
The ``=======`` line separates the two sections.
One can manually edit the file to resolve the conflict, removing the markers and keeping the desired changes, however \
for convenience it is recommended to use a visual tool like the *Merge Editor* in VS Code.

Resolving Conflicts in VS Code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a conflict occurs, VS Code will open the *Merge Editor* to help you resolve it.\

1. Open the conflicting file (marked with a ``!``).
2. The *Merge Editor* will show the conflicting sections side by side.
3. Click on the ``Accept Current Change`` or ``Accept Incoming Change`` buttons to keep the desired changes, sometimes both changes will be kept or even a manual edit will be necessary.

More details on VS Code interface and conflict solver can be found in `VS Code Docs <https://code.visualstudio.com/docs/sourcecontrol/overview#_3way-merge-editor>`_.
After resolving the conflicts, save the files, make sure all conflicts are resolved, and then \
commit the changes.

