RocketPy documentation
======================

RocketPy uses `Sphinx <https://www.sphinx-doc.org/>`_ to generate the
documentation.
Sphinx makes it easy to create documentation for Python projects and it is
widely used for Python projects, such as
`NumPy <https://numpy.org/doc/stable/>`_,
`Pandas <https://pandas.pydata.org/docs/>`_ and
`SciPy <https://docs.scipy.org/doc/scipy/>`_.


The `ReadTheDocs <https://about.readthedocs.com/?ref=readthedocs.com>`_ is used
to host the documentation. It is a free service that automatically builds
documentation from your sphinx source files and hosts them for free.

RocketPy official documentation is available at
`https://docs.rocketpy.org <https://docs.rocketpy.org/en/latest/index.html>`_.


How to build the documentation in your local machine
----------------------------------------------------

When you find yourself modifying the documentation files and trying to see the
results, you can build the documentation in your local machine.
This is important to check if the documentation is being generated correctly
before pushing the changes to the repository.

To build the documentation in your local machine, you need to install a set of
requirements that are needed to run the sphinx generator.
All these requirements are listed in the ``requirements.txt`` file inside the
``docs`` folder.

To install the requirements, navigate the terminal to the ``docs`` folder and
run the following command:

.. code-block:: bash
    
    pip install -r requirements.txt

After installing the requirements, you can build the documentation by running
the following command in your terminal:

.. code-block:: bash
    
    make html

The file named ``Makefile`` contains the commands to build the documentation.
The ``make html`` command will generate the documentation in the ``docs/_build/html``
folder.

To see the documentation, open the ``docs/_build/html/index.html`` file in your
browser.

.. note:: Watch out for any warnings or errors that may appear in the terminal
          when building the documentation. If you find any, fix them before
          pushing the changes to the repository or at least warn the team about
          them.

Sometimes you may face problems when building the documentation after several
times of building it.
This may happen because sphinx does not clean the ``docs/_build`` folder before
building the documentation again.
To clean the ``docs/_build`` folder, run the following command in your terminal:

.. code-block:: bash

    make clean

After cleaning the ``docs/_build`` folder, you can build the documentation again
by running the ``make html`` command.

If the error persists, it may be related to other files, such as the ``.rst``
files or the ``conf.py`` file.

.. danger::
    
    Do not modify the Makefile or the ``make.bat`` files. These files are \
    automatically generated by sphinx and any changes will be lost.


How to integrate the documentation with ReadTheDocs
---------------------------------------------------

The documentation is automatically built and hosted by `ReadTheDocs`.
Every time a commit is pushed to the repository, `ReadTheDocs` will build the
documentation and host it automatically. 
This includes other branches besides the master branch.
However, the documentation will only be available for the master branch, and you
need to configure the `ReadTheDocs` project to build the documentation for other
branches.

The connection between the GitHub repository and the `ReadTheDocs` project is
already configured and defined in the ``readthedocs.yml`` file, available at the
root of the repository.

.. note::
    
    You need admin permissions to configure the `ReadTheDocs` project. Ask \
    the team for help if you don't have admin permissions.

