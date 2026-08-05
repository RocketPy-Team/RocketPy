.. _controllers:

Controllers
===========

RocketPy can simulate active, in-flight control systems — such as air brakes or
other actuators — through a *controller function* that you attach to a
controllable object (for example, an :class:`rocketpy.AirBrakes` added with
``Rocket.add_air_brakes``). During the flight simulation, RocketPy calls your
controller function at the appropriate times, letting it read the current
simulation state and command the actuator.

This page describes the controller-function interface and how its call timing is
governed. For a complete, runnable example that puts these pieces together, see
the :doc:`Air Brakes Example <airbrakes>`.

The Controller Function
-----------------------

A controller function receives information about the simulation up to the
current time step and sets the state of the controlled object. It must take in
the following arguments, in this order:

1. ``time`` (float): The current simulation time in seconds.
2. ``sampling_rate`` (float or ``None``): The rate at which the controller
   function is called, measured in Hertz (Hz). It is ``None`` for continuous
   controllers, so guard any ``1 / sampling_rate`` computation against ``None``.
3. ``state`` (list): The state vector of the simulation. The state
   is a list containing the following values, in this order:

   - ``x``: The x position of the rocket, in meters.
   - ``y``: The y position of the rocket, in meters.
   - ``z``: The z position of the rocket, in meters.
   - ``v_x``: The x component of the velocity of the rocket, in meters per
     second.
   - ``v_y``: The y component of the velocity of the rocket, in meters per
     second.
   - ``v_z``: The z component of the velocity of the rocket, in meters per
     second.
   - ``e0``: The first component of the quaternion representing the rotation
     of the rocket.
   - ``e1``: The second component of the quaternion representing the rotation
     of the rocket.
   - ``e2``: The third component of the quaternion representing the rotation
     of the rocket.
   - ``e3``: The fourth component of the quaternion representing the rotation
     of the rocket.
   - ``w_x``: The x component of the angular velocity of the rocket, in
     radians per second.
   - ``w_y``: The y component of the angular velocity of the rocket, in
     radians per second.
   - ``w_z``: The z component of the angular velocity of the rocket, in
     radians per second.

4. ``state_history`` (list): A record of the rocket's state at each
   step throughout the simulation. The state_history is organized as
   a list of lists, with each sublist containing a state vector. The
   last item in the list always corresponds to the previous state
   vector, providing a chronological sequence of the rocket's
   evolving states.
5. ``observed_variables`` (list): A list containing the variables that
   the controller function returns. The return of each controller
   function call is appended to the observed_variables list. The
   initial value in the first step of the simulation of this list is
   provided by the ``initial_observed_variables`` argument.
6. The **controlled object** (e.g. ``air_brakes``): the instance being
   controlled, whose attributes the function sets (for example,
   ``air_brakes.deployment_level``).

.. note::

    - The controller function accepts 6, 7, or 8 parameters for backward
      compatibility:

      * **6 parameters** (original): ``time``, ``sampling_rate``, ``state``,
        ``state_history``, ``observed_variables``, and the controlled object.
      * **7 parameters** (with sensors): adds ``sensors`` as the 7th parameter.
      * **8 parameters** (with environment): adds ``sensors`` and ``environment``
        as the 7th and 8th parameters.

    - The **environment parameter** provides access to atmospheric conditions
      (wind, temperature, pressure, elevation) without relying on global
      variables. This enables proper serialization of rockets with controllers
      and improves code modularity. Available methods include
      ``environment.elevation``, ``environment.wind_velocity_x(altitude)``,
      ``environment.wind_velocity_y(altitude)``,
      ``environment.speed_of_sound(altitude)``, and others.

    - Anything can be returned by the controller function. The returned values
      are saved in the ``observed_variables`` list at every time step and can
      then be accessed by the controller function at the next time step. The
      saved values can also be accessed after the simulation is finished, which
      is useful for debugging and for plotting the results.

    - The controller function can also be defined in a separate file and
      imported into the simulation script. This includes importing ``c`` or
      ``cpp`` code into Python.

.. _discrete-vs-continuous-controllers:

Discrete vs. Continuous Controllers
-----------------------------------

The ``sampling_rate`` argument determines *when* the controller function is
called during the flight simulation:

- **Discrete controller** (``sampling_rate`` set to a number, e.g. ``10``):
  the controller function is called at fixed intervals of ``1 / sampling_rate``
  seconds. This mirrors a real flight computer that reads its sensors and
  updates its actuators at a fixed frequency, and is the recommended choice
  when you want the simulation to reflect the actual control-loop rate of your
  hardware.
- **Continuous controller** (``sampling_rate=None``): the controller function
  is called at *every* solver step of the numerical integrator. Use this when
  you want the control law to act as a continuous function of the state rather
  than a sampled one (for example, when validating a control model
  analytically).

.. warning::

    For continuous controllers, ``sampling_rate`` is passed to your controller
    function as ``None``. Any computation such as ``1 / sampling_rate`` (a
    common pattern for rate-limiting deployment) must guard against ``None`` to
    avoid a ``TypeError``.

.. note::

    Discrete controllers add their sampling instants as time nodes to the
    simulation, so remember to set ``time_overshoot=False`` in the ``Flight``
    to make the integrator stop exactly at those instants. Continuous
    controllers do not add time nodes; they are evaluated on the integrator's
    own steps.
