.. _sensorsusage:

Sensors
=======

RocketPy can simulate the data that onboard sensors would record during a
flight. This is useful to test avionics and control algorithms (such as air
brakes or active stabilization) against realistic, imperfect measurements
before flying the real hardware.

Four sensor types are available:

- :class:`rocketpy.Accelerometer` - measures linear acceleration (m/s^2) along
  three axes.
- :class:`rocketpy.Gyroscope` - measures angular velocity (rad/s) around three
  axes.
- :class:`rocketpy.Barometer` - measures the static pressure (Pa) at the sensor
  location.
- :class:`rocketpy.GnssReceiver` - measures the position of the rocket as
  latitude, longitude and altitude.

The :class:`rocketpy.Accelerometer` and :class:`rocketpy.Gyroscope` are
*inertial* sensors: their readings depend on the orientation and position of the
sensor in the rocket. The :class:`rocketpy.Barometer` is a *scalar* sensor and
the :class:`rocketpy.GnssReceiver` measures absolute position, so neither is
affected by the sensor orientation.

Every sensor degrades the true flight state with configurable imperfections
(misalignment, cross-axis coupling, quantization, noise, bias and temperature
drift). How each of these effects is modelled is specified in the
:ref:`sensors_technical` section at the end of this page; the sections before it
focus on *using* sensors and on *accessing their data through the Flight class*.

Setting Up the Rocket and Environment
-------------------------------------

The sensors below are attached to the same model used in the
:ref:`First Simulation <firstsimulation>` example. The environment, motor and
rocket setup is identical to that page, so its code is hidden here - refer to
:ref:`First Simulation <firstsimulation>` for a line-by-line explanation. The
``calisto`` rocket and ``env`` environment created below are used throughout
this page.

.. jupyter-execute::
    :hide-code:
    :hide-output:

    from rocketpy import Environment, SolidMotor, Rocket, Flight

    env = Environment(latitude=32.990254, longitude=-106.974998, elevation=1400)
    env.set_atmospheric_model(
        type="custom_atmosphere",
        wind_u=[(0, 3), (10000, 3)],
        wind_v=[(0, 5), (10000, -5)],
    )

    Pro75M1670 = SolidMotor(
        thrust_source="../data/motors/cesaroni/Cesaroni_M1670.eng",
        dry_mass=1.815,
        dry_inertia=(0.125, 0.125, 0.002),
        nozzle_radius=33 / 1000,
        grain_number=5,
        grain_density=1815,
        grain_outer_radius=33 / 1000,
        grain_initial_inner_radius=15 / 1000,
        grain_initial_height=120 / 1000,
        grain_separation=5 / 1000,
        grains_center_of_mass_position=0.397,
        center_of_dry_mass_position=0.317,
        nozzle_position=0,
        burn_time=3.9,
        throat_radius=11 / 1000,
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )

    calisto = Rocket(
        radius=127 / 2000,
        mass=14.426,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag="../data/rockets/calisto/powerOffDragCurve.csv",
        power_on_drag="../data/rockets/calisto/powerOnDragCurve.csv",
        center_of_mass_without_motor=0,
        coordinate_system_orientation="tail_to_nose",
    )

    calisto.set_rail_buttons(
        upper_button_position=0.0818,
        lower_button_position=-0.618,
        angular_position=45,
    )

    calisto.add_motor(Pro75M1670, position=-1.255)
    calisto.add_nose(length=0.55829, kind="vonKarman", position=1.278)
    calisto.add_trapezoidal_fins(
        n=4,
        root_chord=0.120,
        tip_chord=0.060,
        span=0.110,
        position=-1.04956,
        cant_angle=0.5,
    )
    calisto.add_tail(
        top_radius=0.0635, bottom_radius=0.0435, length=0.060, position=-1.194656
    )

Creating Sensors
----------------

Sensors are created independently from the rocket. The only required argument is
the ``sampling_rate`` (in Hz), which sets how often the sensor records a
measurement during the flight. All the other arguments describe the
imperfections of the sensor and default to an *ideal* sensor (no misalignment,
no noise, no bias, infinite range and resolution). The meaning of every
parameter is specified in the :ref:`sensors_technical` section.

Let's build one of each sensor type. The accelerometer is left aligned with the
rocket and ideal; the gyroscope is deliberately misaligned and given per-axis
noise; the barometer and GNSS receiver use their own noise parameters:

.. jupyter-execute::

    from rocketpy import Accelerometer, Gyroscope, Barometer, GnssReceiver

    accelerometer = Accelerometer(
        sampling_rate=100,
        measurement_range=70,
        resolution=0.4882,
        noise_density=0.05,
        random_walk_density=0.02,
        constant_bias=0.5,
        name="Accelerometer",
    )

    gyroscope = Gyroscope(
        sampling_rate=100,
        orientation=(60, 60, 60),  # deliberately misaligned, see Technical Reference
        resolution=0.0011,
        noise_density=[0, 0.03, 0.05],
        random_walk_density=[0, 0.01, 0.02],
        constant_bias=[0, 0.3, 0.5],
        acceleration_sensitivity=[0, 0.0008, 0.0017],
        name="Gyroscope",
    )

    barometer = Barometer(
        sampling_rate=50,
        measurement_range=100000,
        resolution=0.16,
        noise_density=19,
        noise_variance=19,
        random_walk_density=0.01,
        constant_bias=1,
        name="Barometer",
    )

    gnss = GnssReceiver(
        sampling_rate=1,
        position_accuracy=1,
        altitude_accuracy=1,
        name="GnssReceiver",
    )

Each sensor object can be inspected on its own with ``all_info()``, which prints
its configuration and (after a flight) plots its most recent run:

.. jupyter-execute::

    gyroscope.prints.all()

Adding Sensors to the Rocket
----------------------------

Sensors are attached with :meth:`rocketpy.Rocket.add_sensor`, which takes the
sensor object and its ``position``. The position can be a single value along the
rocket's centerline, or a ``(x, y, z)`` tuple in the rocket's user-defined
coordinate system. The position matters for inertial sensors: a sensor mounted
off the center of mass experiences extra acceleration due to the rocket's
rotation (see :ref:`sensors_technical`).

.. jupyter-execute::

    calisto.add_sensor(accelerometer, 1.278)
    calisto.add_sensor(gyroscope, -0.10482544178314143)
    calisto.add_sensor(barometer, (-0.10482544178314143, -127 / 2000, 0))
    calisto.add_sensor(gnss, (-0.10482544178314143, 127 / 2000, 0))

Visualizing sensor orientation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``calisto.draw()`` marks each sensor with a short line at its position and an
arrow pointing along the sensor's **normal vector** - the sensor frame's
z-axis, i.e. the direction the sensor's "up" axis points in the rocket. Scalar
sensors (barometer, GNSS) have no orientation, so they are drawn without an
arrow.

A single 2-D view only shows the *projection* of that 3-D arrow onto the drawing
plane. The ``plane`` argument selects which projection:

- ``plane="xz"`` shows the arrow's components along the rocket's **z** (long
  axis) and **x** axes.
- ``plane="yz"`` shows the arrow's components along the rocket's **z** and
  **y** axes.

Comparing the two views therefore reveals the full 3-D orientation. The
accelerometer is aligned with the rocket, so its arrow points straight along the
body axis in *both* planes. The gyroscope was misaligned with
``orientation=(60, 60, 60)``, so its arrow leans - and leans *differently* in
each plane, because its normal vector has distinct x and y components.

.. jupyter-execute::

    calisto.draw(plane="xz")

.. jupyter-execute::

    calisto.draw(plane="yz")

Using Sensors With a Controller (Air Brakes)
--------------------------------------------

The main motivation for simulating sensors is to feed their measurements into a
controller, such as an air brakes system, exactly as the real avionics would.
Inside a controller function, every sensor attached to the rocket is available
through the ``sensors`` keyword argument (a list, in the order they were added)
and through ``sensors_by_name`` (a dictionary keyed by the sensor ``name``).
Each sensor exposes its *latest* reading via the ``measurement`` attribute.

The controller below reads the accelerometer to detect motor burnout (measured
vertical acceleration turning negative) before it starts deploying the air
brakes:

.. jupyter-execute::

    def controller_function(**kwargs):
        time = kwargs["time"]
        sampling_rate = kwargs["sampling_rate"]
        state = kwargs["state"]
        air_brakes = kwargs["air_brakes"]
        sensors = kwargs["sensors"]

        # Read the sensor measurement instead of the true state.
        # Sensors can also be retrieved by name, e.g.
        # accelerometer = kwargs["sensors_by_name"]["Accelerometer"]
        accelerometer = sensors[0]

        # Do not deploy while the motor is still burning (measured az > 0)
        if accelerometer.measurement[2] > 0:
            return None

        altitude_AGL = kwargs["height_agl"]
        vz = state[5]

        # Below 1500 m AGL, keep the air brakes closed
        if altitude_AGL < 1500:
            air_brakes.deployment_level = 0
        else:
            new_deployment_level = air_brakes.deployment_level + 0.1 * vz
            # Limit the deployment rate to 0.2 per second
            max_change = 0.2 / sampling_rate
            lower_bound = air_brakes.deployment_level - max_change
            upper_bound = air_brakes.deployment_level + max_change
            new_deployment_level = min(
                max(new_deployment_level, lower_bound), upper_bound
            )
            air_brakes.deployment_level = new_deployment_level

        return time, air_brakes.deployment_level

.. note::

    The controller reads ``accelerometer.measurement`` rather than the true
    ``state`` vector. This is what makes the test realistic: the controller only
    sees the misaligned, noisy, quantized and biased data the physical sensor
    would actually provide.

Register the controller through the air brakes. See the
:ref:`Air Brakes <airbrakes>` example for a detailed explanation of these
arguments.

.. jupyter-execute::

    air_brakes = calisto.add_air_brakes(
        drag_coefficient_curve="../data/rockets/calisto/air_brakes_cd.csv",
        controller_function=controller_function,
        sampling_rate=10,
        reference_area=None,
        clamp=True,
        initial_observed_variables=[0, 0],
        override_rocket_drag=False,
        name="AirBrakes",
    )

Simulating the Flight
---------------------

Sensors record data automatically while the flight is simulated; no special
configuration is needed in the :class:`rocketpy.Flight` class.

.. jupyter-execute::

    test_flight = Flight(
        rocket=calisto,
        environment=env,
        rail_length=5.2,
        inclination=85,
        heading=0,
        terminate_on_apogee=True,
    )

.. _sensors_flight_data:

Accessing Sensor Data via the Flight Class
------------------------------------------

The recorded measurements are stored on the resulting flight, in the
``sensor_data`` attribute. This is the **canonical, per-flight record** of every
sensor's output and the recommended way to access sensor data.

.. important::

    A sensor object only retains the measurements of its *most recent* flight in
    ``sensor.measured_data`` - running another flight with the same rocket (for
    example in a Monte Carlo analysis, or when comparing two flights) overwrites
    it. ``flight.sensor_data`` is captured per flight, so always prefer it when a
    rocket or sensor is reused across simulations.

``flight.sensor_data`` is a dictionary keyed by the sensor object. Each value is
a list of ``(time, *measurement)`` tuples (or a list of such lists if the same
sensor object was added to the rocket in more than one position):

.. jupyter-execute::

    data = test_flight.sensor_data[accelerometer]
    print("Number of accelerometer samples:", len(data))
    print("First sample (t, ax, ay, az):", data[0])

Prints and plots
~~~~~~~~~~~~~~~~~

The flight's ``prints`` and ``plots`` objects expose sensor-aware helpers that
read directly from ``flight.sensor_data``. Use them to summarize and visualize
every sensor at once.

``flight.prints.sensors()`` prints a per-sensor summary (type, sampling rate,
number of samples and per-channel min/max/mean):

.. jupyter-execute::

    test_flight.prints.sensors()

``flight.plots.sensor_data()`` plots every channel of every sensor against time:

.. jupyter-execute::

    test_flight.plots.sensor_data()

Working with the raw data
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because ``flight.sensor_data`` holds plain tuples, the data can be unpacked and
post-processed directly. For example, to plot the measured vertical
acceleration:

.. jupyter-execute::

    import matplotlib.pyplot as plt

    time, ax, ay, az = zip(*test_flight.sensor_data[accelerometer])

    plt.plot(time, az, label="Measured az")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration az (m/s^2)")
    plt.legend()
    plt.grid()
    plt.show()

Exporting
~~~~~~~~~

The flight's data exporter writes every sensor's data (or a single sensor's, by
object or name) to a ``json`` file:

.. jupyter-execute::

    test_flight.exports.sensor_data("sensor_data.json")          # all sensors
    test_flight.exports.sensor_data("barometer.json", barometer)  # one sensor

.. _sensors_technical:

Technical Reference: Sensor Modelling
-------------------------------------

This section specifies how a sensor turns the true flight state into a recorded
measurement. For an inertial sensor the value passes through the following
chain, in order:

1. the true physical quantity is evaluated **at the sensor's location**;
2. it is rotated into the **sensor frame**, accounting for orientation and
   cross-axis sensitivity;
3. it is degraded by the **noise pipeline** (noise & bias, temperature drift,
   quantization).

Throughout, :math:`s_k` denotes the ideal value at sample :math:`k`,
:math:`m_k` the recorded measurement, :math:`f_s` the ``sampling_rate`` (Hz),
and :math:`T_0 = 298.15\ \text{K}` the reference temperature. For inertial
sensors every effect is applied independently per axis, so each parameter may be
a single value (applied to all three axes) or a list of three values (one per
axis).

Coordinate frames, position and orientation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The rocket's body frame is described in :ref:`rocket_axes`. A sensor defines its
own frame whose z-axis is the sensor's **normal vector** (the arrow shown by
``rocket.draw()``). The ``orientation`` argument is the rotation from the sensor
frame to the body frame and may be given as:

- a tuple/list of three Euler angles in degrees, or
- a 3x3 rotation matrix (list of lists) from the sensor frame to the body frame.

With ``orientation=(0, 0, 0)`` (the default) the sensor axes coincide with the
body axes, so the normal vector points along the rocket's longitudinal axis.

Scalar sensors (:class:`rocketpy.Barometer`, :class:`rocketpy.GnssReceiver`) are
not affected by orientation.

**Position / lever-arm.** For an :class:`rocketpy.Accelerometer` mounted at
position :math:`\mathbf{r}` relative to the rocket's center of mass, the
acceleration sensed is the center-of-mass acceleration plus the tangential and
centripetal terms produced by the body's angular velocity
:math:`\boldsymbol\omega` and angular acceleration :math:`\dot{\boldsymbol\omega}`:

.. math::

    \mathbf{a}_\text{sensor} = \mathbf{a}_\text{cm}
        + \dot{\boldsymbol\omega} \times \mathbf{r}
        + \boldsymbol\omega \times (\boldsymbol\omega \times \mathbf{r})

This is why the sensor ``position`` matters for accelerometers. By default the
accelerometer does **not** include gravity; set ``consider_gravity=True`` to add
the gravitational acceleration to :math:`\mathbf{a}_\text{cm}`.

Axis misalignment and cross-axis sensitivity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition to the (intentional) orientation above, real inertial axes are never
perfectly orthogonal. The ``cross_axis_sensitivity`` :math:`s` (in %) models this
by leaking a fraction of each axis into the others; the ideal sensor-frame
vector is pre-multiplied by:

.. math::

    \mathbf{C} =
    \begin{bmatrix}
        1 & s/100 & s/100 \\
        s/100 & 1 & s/100 \\
        s/100 & s/100 & 1
    \end{bmatrix}

Default ``0`` (perfectly orthogonal axes).

The noise pipeline
~~~~~~~~~~~~~~~~~~~

After the steps above, the (per-axis) value passes through three stages applied
to every sample.

**Stage 1 - Noise and bias.** Adds white noise :math:`w_k`, a random-walk drift
:math:`b_k` and a constant offset:

.. math::

    y_k = s_k + w_k + b_k + \text{constant\_bias}

.. math::

    w_k = \text{noise\_density}\,\sqrt{f_s}\; g_k,
    \qquad g_k \sim \mathcal{N}(0,\ \text{noise\_variance})

.. math::

    b_k = b_{k-1} + \frac{\text{random\_walk\_density}}{\sqrt{f_s}}\; h_k,
    \qquad h_k \sim \mathcal{N}(0,\ \text{random\_walk\_variance})

The random-walk drift (also called *bias instability* or *bias drift*) starts at
:math:`b_0 = 0` and accumulates over the flight.

- ``noise_density``: amplitude of the white noise, in units/√Hz. Sometimes
  called *velocity random walk* (accelerometers), *angular random walk*
  (gyroscopes) or *(rate) noise density*. Default ``0``.
- ``noise_variance``: variance of the unit draw :math:`g_k`. Default ``1``, so
  the white-noise standard deviation is :math:`\text{noise\_density}\,\sqrt{f_s}`.
- ``random_walk_density``: amplitude of the random-walk drift, in units/√Hz.
  Default ``0``.
- ``random_walk_variance``: variance of the unit increment :math:`h_k`. Default
  ``1``.
- ``constant_bias``: fixed offset added to every sample, in the sensor's units.
  Default ``0``.

**Stage 2 - Temperature drift.** Biases and scales the value according to how
far the sensor temperature :math:`T` (``operating_temperature``) is from
:math:`T_0`:

.. math::

    y_k' = \Big(y_k + (T - T_0)\,\text{temperature\_bias}\Big)
           \left(1 + \frac{T - T_0}{100}\,\text{temperature\_scale\_factor}\right)

At :math:`T = T_0` (25 °C) this stage is the identity.

- ``operating_temperature``: sensor temperature :math:`T`, in Kelvin. Default
  ``298.15``.
- ``temperature_bias``: additive bias per Kelvin of deviation from :math:`T_0`,
  in units/K. Default ``0``.
- ``temperature_scale_factor``: multiplicative (gain) error per Kelvin of
  deviation from :math:`T_0`, in %/K. Default ``0``.

**Stage 3 - Quantization.** Saturates the value to the measurement range, then
rounds it to the resolution. With ``measurement_range`` :math:`= (r_\min, r_\max)`:

.. math::

    y_k'' = \min\!\big(\max(y_k',\ r_\min),\ r_\max\big)

.. math::

    m_k = \text{resolution}\cdot
          \operatorname{round}\!\left(\frac{y_k''}{\text{resolution}}\right)

Rounding is only applied when ``resolution`` is non-zero.

- ``measurement_range``: saturation limits, in the sensor's units. A single
  value :math:`R` applies the symmetric range :math:`(-R, R)`; a tuple sets
  :math:`(r_\min, r_\max)` independently. Default ``np.inf`` (no saturation).
- ``resolution``: quantization step, in units/LSB. Default ``0`` (no
  quantization).

Gyroscope acceleration sensitivity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`rocketpy.Gyroscope` has one extra effect: a spurious angular rate
proportional, axis by axis, to the linear acceleration
:math:`\mathbf{a}_\text{sensor}` felt at its location:

.. math::

    \Delta\boldsymbol{\omega} =
    \text{acceleration\_sensitivity} \odot \mathbf{a}_\text{sensor}

where :math:`\odot` is the element-wise (Hadamard) product and
``acceleration_sensitivity`` is in rad/s/g. Default ``0``.

GNSS receiver model
~~~~~~~~~~~~~~~~~~~~~

The :class:`rocketpy.GnssReceiver` does not use the pipeline above. Its true
Cartesian position is perturbed by independent Gaussian noise and then converted
to latitude, longitude and altitude:

.. math::

    x \sim \mathcal{N}(x_\text{true},\ \text{position\_accuracy}), \quad
    y \sim \mathcal{N}(y_\text{true},\ \text{position\_accuracy}), \quad
    z \sim \mathcal{N}(z_\text{true},\ \text{altitude\_accuracy})

- ``position_accuracy``: standard deviation of the horizontal position error, in
  meters. Default ``0``.
- ``altitude_accuracy``: standard deviation of the altitude error, in meters.
  Default ``0``.
