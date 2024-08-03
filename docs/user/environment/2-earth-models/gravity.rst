Gravity
=======

RocketPy provides a flexible way to model the gravitational acceleration in 
simulations. The ``gravity`` parameter can be set to various input types to
define the gravity model used in the simulation. Below are the details on how
to set and use the gravity model in RocketPy.

Gravity Parameter
-----------------

The ``gravity`` parameter can be specified as an integer, float, callable,
string, or array. Positive values indicate downward acceleration. If ``None``,
the Somigliana formula is used. For more information, refer
to :meth:`Environment.set_gravity_model`.

Constant Gravity Acceleration
-----------------------------

To set up an ``Environment`` object with a constant gravity acceleration:

.. jupyter-execute::

    from rocketpy import Environment


    g_0 = 9.80665
    env = Environment(gravity=g_0)

    print("Gravity acceleration at 0m ASL: ", env.gravity(0)) 
    print("Gravity acceleration at 1000m ASL: ", env.gravity(1000))
    print("Gravity acceleration at 10000m ASL: ", env.gravity(10000))


Variable Gravity Acceleration
-----------------------------

To vary the gravity acceleration as a function of height:

.. jupyter-execute::

    from rocketpy import Environment


    def gravity(height):
        g_0 = 9.80665 # Gravity at sea level
        R_t = 6371000 # Earth radius
        return g_0 * (R_t / (R_t + height))**2 
    
    env = Environment(gravity=gravity)

    print("Gravity acceleration at 0m ASL: ", env.gravity(0)) 
    print("Gravity acceleration at 1000m ASL: ", env.gravity(1000))
    print("Gravity acceleration at 10000m ASL: ", env.gravity(10000))


Somigliana Gravity Formula
--------------------------

The Somigliana formula computes the gravity acceleration with an altitude
correction accurate for aviation heights. This formula is based on the WGS84
ellipsoid.

.. tip::

    The Somigliana formula is the default gravity model used in RocketPy. If \
    the ``gravity`` parameter is set to ``None``, the Somigliana formula is \
    used.

