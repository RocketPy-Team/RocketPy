Installation
============

Here you will learn all of the useful ways you can install RocketPy.


Quick Install Using ``pip``
---------------------------

To get a copy of RocketPy's latest stable version using ``pip``, just open up your terminal and run:

.. code-block:: shell

    pip install rocketpy

If you don't see any error messages, you are all set!

If you want to choose a specific version to guarantee compatibility, you may instead run:

.. code-block:: shell

    pip install rocketpy==0.11.0


Optional Installation Method: ``conda``
---------------------------------------

Alternatively, RocketPy can also be installed using ``conda`` and the `Conda-Forge <https://conda-forge.org/>`_ channel.
Just open your Anaconda terminal and run:

.. code-block:: shell

    conda install -c conda-forge rocketpy


Optional Installation Method: from source
-----------------------------------------

If you wish to downloaded RocketPy from source, you may do so either by:

- Downloading it from `RocketPy's GitHub <https://github.com/Projeto-Jupiter/RocketPy>`_ page and unzipping the downloaded folder.
- Or cloning it to a desired directory using git:

.. code-block:: shell

    git clone https://github.com/giovaniceotto/RocketPy.git

Once you are done downloading/cloning RocketPy's repository, you can install it by opening up a terminal inside the repository's folder on your computer and running:

.. code-block:: shell

    python setup.py install 


Development version
-------------------

Using ``pip``
^^^^^^^^^^^^^

RocketPy is being actively developed, which means we have stable versions and development versions.
All methods above will install a stable version to your computer.
If you want to get the latest development version, you also can!
And it's as simple as using ``pip``.
Just open up your terminal and run:

.. code-block:: shell

    pip install git+https://github.com/Projeto-Jupiter/RocketPy.git@develop


Cloning RocketPy's Repo
^^^^^^^^^^^^^^^^^^^^^^^

Alternatively, you can clone RocketPy's repository, check out the branch named ``develop`` and proceed with:

.. code-block:: shell

    python setup.py install 


