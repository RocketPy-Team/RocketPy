Standard Atmosphere
===================

By default, when initializing an Environment class, the International Standard
Atmosphere (ISA) as defined by ISO 2533 is initialized.

.. seealso::

    For more information on the International Standard Atmosphere, see
    `ISO 2533 <https://www.iso.org/obp/ui/en/#iso:std:iso:2533:ed-1:v1:en>`_.

Note that the International Standard Atmosphere only has temperature and pressure
profiles properly specified.
Other profiles can be derived from it, however, winds are automatically set to
0 m/s.

.. jupyter-execute::

    from rocketpy import Environment

    env = Environment() # already initializes the International Standard Atmosphere

    env.plots.atmospheric_model()

|

The International Standard Atmosphere can also be reset at any time by using the
:meth:`rocketpy.Environment.set_atmospheric_model` method. For example:

.. jupyter-execute::

    env.set_atmospheric_model(type="standard_atmosphere")
