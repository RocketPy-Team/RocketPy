Acceleration-Based Parachute Triggers
======================================

RocketPy lets parachute trigger functions access the state derivative
``u_dot`` (which holds the accelerations at indices ``[3:6]``) in addition to
pressure, height and the state vector. This enables avionics-style deployment
logic that mimics how real flight computers use accelerometer (IMU) data to
detect flight phases such as burnout, free-fall or liftoff.

Overview
--------

Built-in string and numeric triggers rely on altitude and vertical velocity.
By writing a **custom trigger function**, you can additionally use acceleration
to implement mission-specific logic, for example:

- Motor burnout detection (sudden drop in acceleration)
- Apogee detection combining near-zero velocity with downward acceleration
- Free-fall / ballistic coast detection (low total acceleration)
- Liftoff detection (high total acceleration)

For realistic noisy measurements, attach an :doc:`Accelerometer sensor
</reference/classes/sensors/index>` to the rocket and read it inside the
trigger instead of feeding the ideal ``u_dot`` directly.

Trigger function signatures
---------------------------

A custom trigger callable may take **3, 4, or 5** arguments. RocketPy detects
the signature automatically and only computes ``u_dot`` when a trigger asks for
it (so legacy triggers pay no performance cost):

- ``(pressure, height, state_vector)`` — the classic signature.
- ``(pressure, height, state_vector, u_dot)`` — name the 4th argument
  ``u_dot`` (or ``udot``/``acc``/``acceleration``) to receive the derivative;
  any other name receives the ``sensors`` list instead.
- ``(pressure, height, state_vector, sensors, u_dot)`` — receive both.

``state_vector`` is ``[x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]`` and
``u_dot`` is ``[vx, vy, vz, ax, ay, az, ...]``.

Built-in apogee trigger
-----------------------

Deploys when the rocket starts descending (vertical velocity becomes negative):

.. code-block:: python

    rocket.add_parachute(
        name="Main",
        cd_s=10.0,
        trigger="apogee",
        sampling_rate=100,
        lag=0.5,
    )

Numeric altitude trigger
------------------------

Pass a number to deploy at a fixed height above ground level while descending:

.. code-block:: python

    rocket.add_parachute(
        name="Main",
        cd_s=10.0,
        trigger=400,  # meters above ground level
        sampling_rate=100,
        lag=0.5,
    )

Custom trigger: motor burnout
-----------------------------

Burnout is highly mission-dependent, so it is best expressed as a custom
trigger with user-defined thresholds.

Logic: detect a drop in vertical or total acceleration once the rocket is above
a minimum height and still ascending.

.. code-block:: python

    def burnout_trigger_factory(
        min_height=5.0,
        min_vz=0.5,
        az_threshold=-8.0,
        total_acc_threshold=2.0,
    ):
        def burnout_trigger(_pressure, height, state_vector, u_dot):
            ax, ay, az = u_dot[3], u_dot[4], u_dot[5]
            total_acc = (ax**2 + ay**2 + az**2) ** 0.5
            vz = state_vector[5]
            if height < min_height or vz <= min_vz:
                return False
            return az < az_threshold or total_acc < total_acc_threshold

        return burnout_trigger

Attach it to a rocket:

.. code-block:: python

    rocket.add_parachute(
        name="Drogue",
        cd_s=1.0,
        trigger=burnout_trigger_factory(
            min_height=10.0,
            min_vz=2.0,
            az_threshold=-10.0,
            total_acc_threshold=3.0,
        ),
        sampling_rate=100,
        lag=1.5,
    )

Custom trigger: apogee by acceleration
--------------------------------------

Logic: near-zero vertical velocity together with downward acceleration.

.. code-block:: python

    def apogee_acc_trigger(_pressure, _height, state_vector, u_dot):
        vz = state_vector[5]
        az = u_dot[5]
        return abs(vz) < 1.0 and az < -0.1

.. code-block:: python

    rocket.add_parachute(
        name="Main",
        cd_s=10.0,
        trigger=apogee_acc_trigger,
        sampling_rate=100,
        lag=0.5,
    )

Custom trigger: free-fall
-------------------------

Logic: low total acceleration while descending above a small height.

.. code-block:: python

    def freefall_trigger(_pressure, height, state_vector, u_dot):
        ax, ay, az = u_dot[3], u_dot[4], u_dot[5]
        total_acc = (ax**2 + ay**2 + az**2) ** 0.5
        vz = state_vector[5]
        return height > 5.0 and vz < -0.2 and total_acc < 11.5

.. code-block:: python

    rocket.add_parachute(
        name="Drogue",
        cd_s=1.0,
        trigger=freefall_trigger,
        sampling_rate=100,
        lag=1.5,
    )

Custom trigger: liftoff
-----------------------

Logic: detect motor ignition by high total acceleration.

.. code-block:: python

    def liftoff_trigger(_pressure, _height, _state_vector, u_dot):
        ax, ay, az = u_dot[3], u_dot[4], u_dot[5]
        total_acc = (ax**2 + ay**2 + az**2) ** 0.5
        return total_acc > 15.0

.. code-block:: python

    rocket.add_parachute(
        name="Lift",
        cd_s=0.5,
        trigger=liftoff_trigger,
        sampling_rate=100,
        lag=0.1,
    )

Custom trigger: using sensor measurements
-----------------------------------------

A 5-argument trigger receives both the ``sensors`` list and ``u_dot``, so you
can cross-check a noisy accelerometer reading against the ideal derivative.

.. code-block:: python

    def advanced_trigger(_pressure, _height, _state_vector, sensors, u_dot):
        if not sensors:
            return False
        acc_reading = sensors[0].measurement
        if acc_reading is None or len(acc_reading) < 3:
            return False
        meas_az = acc_reading[2]
        az = u_dot[5]
        return az < -5.0 and meas_az < -5.0

.. code-block:: python

    rocket.add_parachute(
        name="Advanced",
        cd_s=1.5,
        trigger=advanced_trigger,
        sampling_rate=100,
    )

.. note::

    For realistic IMU behavior, attach a RocketPy sensor with its own noise
    model and read it inside the trigger via ``sensors``, instead of relying on
    the ideal ``u_dot``. See the :doc:`Sensor Classes
    </reference/classes/sensors/index>` for available sensors.

Full example: dual deployment
-----------------------------

In RocketPy only one parachute is active at a time, so a dual-deploy avionics
can be reproduced with two custom triggers — a drogue at burnout and a main at
a lower altitude:

.. code-block:: python

    from rocketpy import Rocket, Flight, Environment

    # Environment and rocket setup
    env = Environment(latitude=32.99, longitude=-106.97, elevation=1400)
    env.set_atmospheric_model(type="standard_atmosphere")

    rocket = Rocket(...)  # configure your rocket (motor, fins, etc.)

    # Drogue: deploy shortly after burnout (acceleration drop while ascending)
    def drogue_burnout_trigger(_pressure, height, state_vector, u_dot):
        az = u_dot[5]
        vz = state_vector[5]
        return height > 10 and vz > 1 and az < -8.0

    rocket.add_parachute(
        name="Drogue",
        cd_s=1.0,
        trigger=drogue_burnout_trigger,
        sampling_rate=100,
        lag=1.5,
        noise=(0, 8.3, 0.5),  # pressure-signal noise
    )

    # Main: deploy below 800 m while descending
    def main_deploy_trigger(_pressure, height, state_vector, u_dot):
        vz = state_vector[5]
        az = u_dot[5]
        return height < 800 and vz < -5 and az > -15

    rocket.add_parachute(
        name="Main",
        cd_s=10.0,
        trigger=main_deploy_trigger,
        sampling_rate=100,
        lag=0.5,
        noise=(0, 8.3, 0.5),
    )

    # Flight simulation
    flight = Flight(
        rocket=rocket,
        environment=env,
        rail_length=5.2,
        inclination=85,
        heading=0,
    )
    flight.all_info()

See Also
--------

- :doc:`Parachute Class Reference </reference/classes/Parachute>`
- :doc:`Flight Simulation </user/flight>`
- :doc:`Sensors </notebooks/sensors>`
