.. _threedofsimulation:

3-DOF Rocket Simulations
=========================

RocketPy supports simplified 3-DOF (3 Degrees of Freedom) trajectory simulations,
where the rocket is modeled as a point mass. This mode is useful for quick
analyses, educational purposes, or when rotational dynamics are negligible.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

In a 3-DOF simulation, the rocket's motion is described by three translational
degrees of freedom (x, y, z positions), ignoring all rotational dynamics. This
simplification:

- **Reduces computational complexity** - Faster simulations for initial design studies
- **Focuses on trajectory** - Ideal for apogee predictions and flight path analysis
- **Simplifies model setup** - Requires fewer input parameters than full 6-DOF

When to Use 3-DOF Simulations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

3-DOF simulations are appropriate when:

- You need quick trajectory estimates during preliminary design
- Rotational stability is not a concern (e.g., highly stable rockets)
- You're performing educational demonstrations
- You want to validate basic flight performance before detailed analysis

.. warning::

    3-DOF simulations **do not** account for:

    - Rocket rotation and attitude dynamics
    - Stability margin and center of pressure effects
    - Aerodynamic moments and angular motion
    - Fin effectiveness and control surfaces

    For complete flight analysis including stability, use standard 6-DOF simulations.

Setting Up a 3-DOF Simulation
------------------------------

A 3-DOF simulation requires three main components:

1. :class:`rocketpy.PointMassMotor` - Motor without rotational inertia
2. :class:`rocketpy.PointMassRocket` - Rocket without rotational properties
3. :class:`rocketpy.Flight` with ``simulation_mode="3 DOF"``

Step 1: Define the Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The environment setup is identical to standard simulations:

.. jupyter-execute::

    from rocketpy import Environment

    env = Environment(
        latitude=32.990254,
        longitude=-106.974998,
        elevation=1400
    )

    env.set_atmospheric_model(type="standard_atmosphere")

Step 2: Create a PointMassMotor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`rocketpy.PointMassMotor` class represents a motor as a point mass,
without rotational inertia or grain geometry:

.. jupyter-execute::

    from rocketpy import PointMassMotor

    # Using a thrust curve file
    motor = PointMassMotor(
        thrust_source="../data/motors/cesaroni/Cesaroni_M1670.eng",
        dry_mass=1.815,
        propellant_initial_mass=2.5,
    )

You can also define a constant thrust profile:

.. jupyter-execute::

    # Constant thrust of 250 N for 3 seconds
    motor_constant = PointMassMotor(
        thrust_source=250,
        dry_mass=1.0,
        propellant_initial_mass=0.5,
        burn_time=3.0,
    )

Or use a custom thrust function:

.. jupyter-execute::

    def custom_thrust(t):
        """Custom thrust profile: ramps up, plateaus, then ramps down"""
        if t < 0.5:
            return 500 * t / 0.5  # Ramp up
        elif t < 2.5:
            return 500  # Plateau
        elif t < 3.0:
            return 500 * (3.0 - t) / 0.5  # Ramp down
        else:
            return 0

    motor_custom = PointMassMotor(
        thrust_source=custom_thrust,
        dry_mass=1.2,
        propellant_initial_mass=0.6,
        burn_time=3.0,
    )

.. seealso::

    For detailed information about :class:`rocketpy.PointMassMotor` parameters,
    see the :class:`rocketpy.PointMassMotor` class documentation.

Step 3: Create a PointMassRocket
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`rocketpy.PointMassRocket` class represents a rocket as a point mass:

.. jupyter-execute::

    from rocketpy import PointMassRocket

    rocket = PointMassRocket(
        radius=0.0635,  # meters
        mass=5.0,  # kg (dry mass without motor)
        center_of_mass_without_motor=0.0,
        power_off_drag=0.5,  # Constant drag coefficient
        power_on_drag=0.5,
    )

    # Add the motor
    rocket.add_motor(motor, position=0)

You can also specify drag as a function of Mach number:

.. jupyter-execute::

    # Drag coefficient vs Mach number
    drag_curve = [
        [0.0, 0.50],
        [0.5, 0.48],
        [0.9, 0.52],
        [1.1, 0.65],
        [2.0, 0.55],
        [3.0, 0.50],
    ]

    rocket_with_drag_curve = PointMassRocket(
        radius=0.0635,
        mass=5.0,
        center_of_mass_without_motor=0.0,
        power_off_drag=drag_curve,
        power_on_drag=drag_curve,
    )

.. note::

    Unlike the standard :class:`rocketpy.Rocket` class, :class:`rocketpy.PointMassRocket`
    does **not** support:

    - Aerodynamic surfaces (fins, nose cones)
    - Inertia tensors
    - Center of pressure calculations
    - Stability margin analysis

Step 4: Run the Simulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a :class:`rocketpy.Flight` object with ``simulation_mode="3 DOF"``:

.. jupyter-execute::

    from rocketpy import Flight

    flight = Flight(
        rocket=rocket,
        environment=env,
        rail_length=5.2,
        inclination=85,  # degrees from horizontal
        heading=0,  # degrees (0 = North, 90 = East)
        simulation_mode="3 DOF",
        max_time=100,
        terminate_on_apogee=False,
    )

.. important::

    The ``simulation_mode="3 DOF"`` parameter is **required** to enable 3-DOF mode.
    Without it, RocketPy will attempt a full 6-DOF simulation and may fail with
    :class:`rocketpy.PointMassRocket`.

Analyzing Results
-----------------

Once the simulation is complete, you can access trajectory data and generate plots.

Trajectory Information
^^^^^^^^^^^^^^^^^^^^^^

View key flight metrics:

.. jupyter-execute::

    flight.info()

This will display:

- Apogee altitude and time
- Maximum velocity
- Flight time
- Landing position

Plotting Trajectory
^^^^^^^^^^^^^^^^^^^

Visualize the 3D flight path:

.. jupyter-execute::

    flight.plots.trajectory_3d()

.. note::

    In 3-DOF mode, the rocket maintains a fixed orientation (no pitch, yaw, or roll),
    so attitude plots are not meaningful.

Available Plots
^^^^^^^^^^^^^^^

The following plots are available for 3-DOF simulations:

.. jupyter-execute::

    # Altitude vs time
    flight.z.plot()

    # Velocity components
    flight.vx.plot()
    flight.vy.plot()
    flight.vz.plot()

    # Total velocity
    flight.speed.plot()

    # Acceleration
    flight.ax.plot()

Export Data
^^^^^^^^^^^

Export trajectory data to CSV:

.. jupyter-execute::

    from rocketpy.simulation import FlightDataExporter

    exporter = FlightDataExporter(flight)
    exporter.export_data(
        "trajectory_3dof.csv",
        "x",
        "y",
        "z",
        "vx",
        "vy",
        "vz",
    )


.. jupyter-execute::
    :hide-code:

    import os
    os.remove("trajectory_3dof.csv")


Complete Example
----------------

Here's a complete 3-DOF simulation from start to finish:

.. jupyter-execute::

    from rocketpy import Environment, PointMassMotor, PointMassRocket, Flight

    # 1. Environment
    env = Environment(
        latitude=39.3897,
        longitude=-8.2889,
        elevation=100
    )
    env.set_atmospheric_model(type="standard_atmosphere")

    # 2. Motor
    motor = PointMassMotor(
        thrust_source=1500,  # Constant 1500 N thrust
        dry_mass=2.0,
        propellant_initial_mass=3.0,
        burn_time=4.0,
    )

    # 3. Rocket
    rocket = PointMassRocket(
        radius=0.0635,
        mass=8.0,
        center_of_mass_without_motor=0.0,
        power_off_drag=0.45,
        power_on_drag=0.45,
    )
    rocket.add_motor(motor, position=0)

    # 4. Simulate
    flight = Flight(
        rocket=rocket,
        environment=env,
        rail_length=5.2,
        inclination=85,
        heading=0,
        simulation_mode="3 DOF",
        max_time=120,
    )

    # 5. Results
    print(f"Apogee: {flight.apogee:.2f} m")
    print(f"Max velocity: {flight.max_speed:.2f} m/s")
    print(f"Flight time: {flight.t_final:.2f} s")

    flight.plots.trajectory_3d()

Weathercocking Model
--------------------

RocketPy's 3-DOF simulation mode includes a weathercocking model that allows
the rocket's attitude to evolve during flight. This feature simulates how a
statically stable rocket naturally aligns with the relative wind direction.

Understanding Weathercocking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Weathercocking is the tendency of a rocket to align its body axis with the
direction of the relative wind. In reality, this occurs due to aerodynamic
restoring moments from fins and other stabilizing surfaces. The 3-DOF
weathercocking model provides a simplified representation of this behavior
without requiring full 6-DOF rotational dynamics.

The weathercocking coefficient (``weathercock_coeff``, often abbreviated
``wc``) represents the rate at which the rocket's body axis aligns with
the relative wind. This simplified model does not consider aerodynamic
surfaces (for example, fins) or compute aerodynamic torques. In a
full 6-DOF model, weathercocking depends on quantities such as the
static margin and the normal-force coefficient, which produce restoring
moments that turn the rocket into the wind. A 3-DOF point-mass
simulation cannot compute those moments, so the model enforces
alignment of the body axis toward the freestream with a proportional
law.

Treat ``weathercock_coeff`` as a tuning parameter that approximates the
combined effect of static stability and restoring moments. It has no
direct physical units; designers typically select values by trial and
error and validate them later against full 6-DOF simulations.

Sources:

- `Weathercocking (NASA Bottle Rocket tutorial) <https://www.grc.nasa.gov/www/k-12/VirtualAero/BottleRocket/airplane/rktcock.html>`_
- `Rocket weather-cocking (NASA beginners guide) <https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/rocket-weather-cocking/#new-flight-path>`_
 
The ``weathercock_coeff`` Parameter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The weathercocking behavior is controlled by the ``weathercock_coeff`` parameter
in the :class:`rocketpy.Flight` class:

.. jupyter-execute::

    from rocketpy import Environment, PointMassMotor, PointMassRocket, Flight

    env = Environment(
        latitude=32.990254,
        longitude=-106.974998,
        elevation=1400
    )
    env.set_atmospheric_model(type="standard_atmosphere")

    motor = PointMassMotor(
        thrust_source=1500,
        dry_mass=1.5,
        propellant_initial_mass=2.5,
        burn_time=3.5,
    )

    rocket = PointMassRocket(
        radius=0.078,
        mass=15.0,
        center_of_mass_without_motor=0.0,
        power_off_drag=0.43,
        power_on_drag=0.43,
    )
    rocket.add_motor(motor, position=0)

    # Flight with weathercocking enabled
    flight = Flight(
        rocket=rocket,
        environment=env,
        rail_length=4.2,
        inclination=85,
        heading=45,
        simulation_mode="3 DOF",
        weathercock_coeff=1.0,  # Example with weathercocking enabled
    )

    print(f"Apogee: {flight.apogee - env.elevation:.2f} m")

The ``weathercock_coeff`` parameter controls the rate at which the rocket
aligns with the relative wind:

- ``weathercock_coeff=0``: No weathercocking (original fixed-attitude behavior)
- ``weathercock_coeff=1.0``: Moderate alignment rate
- ``weathercock_coeff>1.0``: Faster alignment (more stable rocket)

Effect of Weathercocking Coefficient
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Higher values of ``weathercock_coeff`` result in faster alignment with the
relative wind. This affects the lateral motion and impact point:

.. list-table:: Weathercocking Coefficient Effects
   :header-rows: 1
   :widths: 25 25 50

   * - Coefficient
     - Alignment Speed
     - Typical Use Case
   * - 0
     - None (fixed attitude)
     - Original 3-DOF behavior
   * - 1.0
     - Moderate
     - General purpose
   * - 2.0-5.0
     - Fast
     - Highly stable rockets
   * - >5.0
     - Very fast
     - Rockets with large fins

3-DOF vs 6-DOF Comparison Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following example compares a 6-DOF simulation using the full Bella Lui rocket
with 3-DOF simulations using ``PointMassRocket`` and different weathercocking
coefficients. This demonstrates the trade-off between computational speed and
accuracy.

.. note::

    The thrust curve files used in this example (e.g., ``AeroTech_K828FJ.eng``)
    are included in the RocketPy repository under the ``data/motors/`` directory.
    If you are running this code outside of the repository, you can download the
    motor files from `RocketPy's data/motors folder on GitHub
    <https://github.com/RocketPy-Team/RocketPy/tree/master/data/motors>`_ or use
    your own thrust curve files.

**Setup the simulations:**

.. jupyter-execute::

    import numpy as np
    import time
    from rocketpy import Environment, Flight, Rocket, SolidMotor
    from rocketpy.rocket.point_mass_rocket import PointMassRocket
    from rocketpy.motors.point_mass_motor import PointMassMotor

    # Environment
    env = Environment(
        gravity=9.81,
        latitude=47.213476,
        longitude=9.003336,
        elevation=407,
    )
    env.set_atmospheric_model(type="standard_atmosphere")
    env.max_expected_height = 2000

    # Full 6-DOF Motor
    motor_6dof = SolidMotor(
        thrust_source="../data/motors/aerotech/AeroTech_K828FJ.eng",
        burn_time=2.43,
        dry_mass=1,
        dry_inertia=(0, 0, 0),
        center_of_dry_mass_position=0,
        grains_center_of_mass_position=-1,
        grain_number=3,
        grain_separation=0.003,
        grain_density=782.4,
        grain_outer_radius=0.042799,
        grain_initial_inner_radius=0.033147,
        grain_initial_height=0.1524,
        nozzle_radius=0.04445,
        throat_radius=0.0214376,
        nozzle_position=-1.1356,
    )

    # Full 6-DOF Rocket
    rocket_6dof = Rocket(
        radius=0.078,
        mass=17.227,
        inertia=(0.78267, 0.78267, 0.064244),
        power_off_drag=0.43,
        power_on_drag=0.43,
        center_of_mass_without_motor=0,
    )
    rocket_6dof.set_rail_buttons(0.1, -0.5)
    rocket_6dof.add_motor(motor_6dof, -1.1356)
    rocket_6dof.add_nose(length=0.242, kind="tangent", position=1.542)
    rocket_6dof.add_trapezoidal_fins(3, span=0.200, root_chord=0.280, tip_chord=0.125, position=-0.75)

    # Point Mass Motor for 3-DOF
    motor_3dof = PointMassMotor(
        thrust_source="../data/motors/aerotech/AeroTech_K828FJ.eng",
        dry_mass=1.0,
        propellant_initial_mass=1.373,
    )

    # Point Mass Rocket for 3-DOF
    rocket_3dof = PointMassRocket(
        radius=0.078,
        mass=17.227,
        center_of_mass_without_motor=0,
        power_off_drag=0.43,
        power_on_drag=0.43,
    )
    rocket_3dof.add_motor(motor_3dof, -1.1356)

**Run simulations and compare results:**

.. jupyter-execute::

    # 6-DOF Flight
    start = time.time()
    flight_6dof = Flight(
        rocket=rocket_6dof,
        environment=env,
        rail_length=4.2,
        inclination=89,
        heading=45,
        terminate_on_apogee=True,
    )
    time_6dof = time.time() - start

    # 3-DOF with no weathercocking
    start = time.time()
    flight_3dof_0 = Flight(
        rocket=rocket_3dof,
        environment=env,
        rail_length=4.2,
        inclination=89,
        heading=45,
        terminate_on_apogee=True,
        simulation_mode="3 DOF",
        weathercock_coeff=0.0,
    )
    time_3dof_0 = time.time() - start

    # 3-DOF with default weathercocking
    start = time.time()
    flight_3dof_1 = Flight(
        rocket=rocket_3dof,
        environment=env,
        rail_length=4.2,
        inclination=89,
        heading=45,
        terminate_on_apogee=True,
        simulation_mode="3 DOF",
        weathercock_coeff=1.0,
    )
    time_3dof_1 = time.time() - start

    # 3-DOF with high weathercocking
    start = time.time()
    flight_3dof_5 = Flight(
        rocket=rocket_3dof,
        environment=env,
        rail_length=4.2,
        inclination=89,
        heading=45,
        terminate_on_apogee=True,
        simulation_mode="3 DOF",
        weathercock_coeff=5.0,
    )
    time_3dof_5 = time.time() - start

    # Print comparison table
    print("=" * 80)
    print("SIMULATION RESULTS COMPARISON")
    print("=" * 80)
    print("\n{:<30} {:>12} {:>12} {:>12} {:>12}".format(
        "Parameter", "6-DOF", "3DOF(wc=0)", "3DOF(wc=1)", "3DOF(wc=5)"
    ))
    print("-" * 80)
    print("{:<30} {:>12.2f} {:>12.2f} {:>12.2f} {:>12.2f}".format(
        "Apogee (m AGL)",
        flight_6dof.apogee - env.elevation,
        flight_3dof_0.apogee - env.elevation,
        flight_3dof_1.apogee - env.elevation,
        flight_3dof_5.apogee - env.elevation,
    ))
    print("{:<30} {:>12.2f} {:>12.2f} {:>12.2f} {:>12.2f}".format(
        "Apogee Time (s)",
        flight_6dof.apogee_time,
        flight_3dof_0.apogee_time,
        flight_3dof_1.apogee_time,
        flight_3dof_5.apogee_time,
    ))
    print("{:<30} {:>12.2f} {:>12.2f} {:>12.2f} {:>12.2f}".format(
        "Max Speed (m/s)",
        flight_6dof.max_speed,
        flight_3dof_0.max_speed,
        flight_3dof_1.max_speed,
        flight_3dof_5.max_speed,
    ))
    print("{:<30} {:>12.3f} {:>12.3f} {:>12.3f} {:>12.3f}".format(
        "Runtime (s)",
        time_6dof,
        time_3dof_0,
        time_3dof_1,
        time_3dof_5,
    ))
    print("-" * 80)
    print("Speedup vs 6-DOF:             {:>12} {:>12.1f}x {:>12.1f}x {:>12.1f}x".format(
        "-",
        time_6dof / time_3dof_0 if time_3dof_0 > 0 else 0,
        time_6dof / time_3dof_1 if time_3dof_1 > 0 else 0,
        time_6dof / time_3dof_5 if time_3dof_5 > 0 else 0,
    ))

**3D Trajectory Comparison:**

.. jupyter-execute::

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot all trajectories
    ax.plot(flight_6dof.x[:, 1], flight_6dof.y[:, 1], flight_6dof.z[:, 1] - env.elevation,
            "b-", linewidth=2, label="6-DOF")
    ax.plot(flight_3dof_0.x[:, 1], flight_3dof_0.y[:, 1], flight_3dof_0.z[:, 1] - env.elevation,
            "r--", linewidth=2, label="3-DOF (wc=0)")
    ax.plot(flight_3dof_1.x[:, 1], flight_3dof_1.y[:, 1], flight_3dof_1.z[:, 1] - env.elevation,
            "g--", linewidth=2, label="3-DOF (wc=1)")
    ax.plot(flight_3dof_5.x[:, 1], flight_3dof_5.y[:, 1], flight_3dof_5.z[:, 1] - env.elevation,
            "m--", linewidth=2, label="3-DOF (wc=5)")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Altitude AGL (m)")
    ax.set_title("3-DOF vs 6-DOF Trajectory Comparison with Weathercocking")
    ax.legend()
    plt.tight_layout()
    plt.show()

The results show that:

- **3-DOF is 5-7x faster** than 6-DOF simulations
- **Apogee prediction** is within 1-3% of 6-DOF
- **Weathercocking** improves trajectory accuracy by aligning the rocket with relative wind
- **Higher weathercock_coeff** values result in trajectories closer to 6-DOF

Comparison: 3-DOF vs 6-DOF
---------------------------

Understanding the differences between simulation modes:

.. list-table:: 3-DOF vs 6-DOF Comparison
   :header-rows: 1
   :widths: 30 35 35

   * - Feature
     - 3-DOF
     - 6-DOF
   * - Computational Speed
     - 5-7x faster
     - Slower (more accurate)
   * - Rocket Orientation
     - Weathercocking model
     - Full attitude dynamics
   * - Stability Analysis
     - ❌ Not available
     - ✅ Full stability margin
   * - Aerodynamic Surfaces
     - ❌ Not modeled
     - ✅ Fins, nose, tail
   * - Center of Pressure
     - ❌ Not computed
     - ✅ Computed
   * - Moments of Inertia
     - ❌ Not needed
     - ✅ Required
   * - Use Cases
     - Quick estimates, Monte Carlo
     - Detailed design, stability
   * - Trajectory Accuracy
     - Good (~1.5% error)
     - Highly accurate

Best Practices
--------------

1. **Validate with 6-DOF**: After getting initial results with 3-DOF, validate
   critical designs with full 6-DOF simulations.

2. **Check Drag Coefficient**: Ensure your drag coefficient is realistic for
   your rocket's geometry. Use wind tunnel data or CFD if available.

3. **Use Realistic Launch Conditions**: Even in 3-DOF mode, wind conditions
   and rail length affect trajectory.

4. **Document Assumptions**: Clearly document that your analysis uses 3-DOF
   and its limitations.

Limitations and Warnings
------------------------

.. danger::

    **Critical Limitations:**

    - **No stability checking** - The simulation cannot detect unstable rockets
    - **No attitude control** - Air brakes and thrust vectoring are not supported
    - **Simplified weathercocking** - Uses proportional alignment model, not full dynamics

.. warning::

    3-DOF simulations should **not** be used for:

    - Final design verification
    - Stability margin analysis
    - Control system design
    - Fin sizing and optimization
    - Safety-critical trajectory predictions

See Also
--------

- :ref:`First Simulation <firstsimulation>` - Standard 6-DOF simulation tutorial
- :ref:`Rocket Class Usage <rocketusage>` - Full rocket modeling capabilities
- :ref:`Flight Class Usage <flightusage>` - Complete flight simulation options

Further Reading
---------------

For more information about point mass trajectory simulations:

- `Trajectory Optimization <https://en.wikipedia.org/wiki/Trajectory_optimization>`_
- `Equations of Motion <https://en.wikipedia.org/wiki/Equations_of_motion>`_
- `Point Mass Model <https://www.grc.nasa.gov/www/k-12/airplane/flteqs.html>`_