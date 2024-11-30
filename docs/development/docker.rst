RocketPy with docker
=====================

RocketPy does not provide an official docker image, but you can build one
yourself using the provided ``Dockerfile`` and ``docker-compose.yml`` files.

Benefits
--------

Docker allows you to run applications in containers. The main benefits of
using docker are:

1. **Isolation**: run RocketPy in a fresh environment, without
   worrying about dependencies.
2. **Portability**: run RocketPy on any operational system that supports
   docker, including the 3 main operational systems (Windows, Linux and Mac).
3. **Reproducibility**: ensure that our code is working regardless of the
   operational system.

Using docker will be specially important when you are not sure if the code
additions will still run on different operational systems.

Although we have a set of GitHub actions to test the code on different
operational systems every time a pull request is made, it is important to
submit a PR only after you are sure that the code will run flawlessly,
otherwise quota limits may be reached on GitHub.

Requirements
-------------

Before you start, you need to install on your machine:

1. `Docker <https://docs.docker.com/get-docker/>`__, to build and run the image.
2. `Docker Compose <https://docs.docker.com/compose/install/>`__, to compose multiple images at once.
3. Also, make sure you have cloned the RocketPy repository in your machine!

Build the image
----------------

To build the image, run the following command on your terminal:

.. code-block:: console

   docker build -t rocketpy-image -f Dockerfile .


This will build the image and tag it as ``rocketpy-image`` (you can apply another
name of your preference if you want).

An image is a read-only template with instructions for creating a Docker
container (see `Docker docs <https://docs.docker.com/get-started/overview/#docker-objects>`__).

This process may take a while, since it will create an image that could easily
be 1.5 GB in size.
But don't worry, you just need to build the image once.

Run the container
-----------------

Now that you have the image, you can run it as a container:

.. code-block:: console

   docker run -it --entrypoint /bin/bash rocketpy-image


This will run the container and open a bash terminal inside it.
If you are using VSCode, you can even integrate the running container into your
IDE, allowing you to code and test directly within the container environment.
This is particularly useful if you want to maintain your usual development setup
while ensuring consistency in the execution environment.
For more details on how to do this, refer to the
`VSCode docs <https://code.visualstudio.com/docs/remote/containers>`__
on developing inside a container.

Indeed, vscode offers a full support for docker, read the
`vscode docs <https://code.visualstudio.com/docs/containers/overview#_installation>`__
for more details


Run the unit tests
--------------------

You might have noticed that the container is running in an isolated environment
with no access to your machine's files, but the ``Dockerfile`` already copied the
RocketPy repository to the container.
This means that you can run tests (and simulations!) as if you were running
RocketPy on your machine.

As simple as that, you can run the unit tests:

.. code-block:: console

   pytest


To access a list of all available execution options, see the
`pytest docs <https://docs.pytest.org/en/latest/how-to/usage.html>`__.

Compose docker images
---------------------

We also made available a ``docker-compose.yml`` file that allows you to compose
multiple docker images at once.
Unfortunately, this file will not allow you to test the code on different
operational systems at once, since docker images inherits from the host
operational system.
However, it is still useful to run the unit tests on different python versions.

Currently, the ``docker-compose.yml`` file is configured to run the unit tests
on python 3.9 and 3.12.

To run the unit tests on both python versions, run the following command
**on your machine**:

.. code-block:: console

   docker-compose up

Also, you can check the logs of the containers by running:

.. code-block:: console

   docker-compose logs


This command is especially useful for debugging if any issues occur during the
build process or when running the containers.

After you're done testing, or if you wish to stop the containers and remove the
services, use the command:

.. code-block:: console

   docker-compose down


This will stop the running containers and remove the networks, volumes, and
images created by up.


Changing to other operational systems
-------------------------------------

The default image in the ``Dockerfile`` is based on a Linux distribution.
However, you can alter the base image to use different operating systems, though
the process may require additional steps depending on the OS's compatibility
with your project setup.
For instance, certain dependencies or scripts may behave differently or require
different installation procedures, so use it with caution.

To change the base image, you will need to modify the ``FROM`` statement in the
``Dockerfile``.
For example, to use a Windows-based image, you might change:

.. code-block:: Dockerfile

   FROM python:latest


to

.. code-block:: Dockerfile

   FROM mcr.microsoft.com/windows/servercore:ltsc2019


Please note, the above is just an example, and using a different OS may require
further adjustments in the ``Dockerfile``.
We recommend you to see the official Python images on the Docker Hub for
different OS options: `Docker Hub Python Tags <https://hub.docker.com/_/python/tags>`__.

Be aware that switching to a non-Linux image can lead to larger image sizes and
longer pull times.
