Exporting and saving data
=========================

An interesting feature is the ability to export the data from the environment to
a .json file, allowing you to import it later or share it with others.


Exporting the data 
------------------

You can use the :meth:`rocketpy.Environment.export_environment` method to export
the environment data to a .json file.

.. code-block:: python

    from rocketpy import Environment

    env = Environment()
    env.export_environment(filename='environment.json')

.. TODO: explain how to import the .json file back into the environment (takes longer to write)
