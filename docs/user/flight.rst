.. _flightusage:

Flight Class Usage
==================

The :class:`rocketpy.Flight` class is the heart of RocketPy's simulation engine. 
It takes a :class:`rocketpy.Rocket`, an :class:`rocketpy.Environment`, and 
launch parameters to simulate the complete flight trajectory of a rocket from 
launch to landing.

This page covers the comprehensive usage of the Flight class, including:

1. Creating a Flight simulation
2. Understanding Flight parameters
3. Accessing simulation results
4. Plotting flight data
5. Advanced features and options

.. seealso::

    For a complete example of Flight simulation, see the 
    :doc:`First Simulation </user/first_simulation>` guide.

.. contents:: Table of Contents
   :local:
   :depth: 2

Creating a Flight Simulation
----------------------------

Basic Flight Creation
~~~~~~~~~~~~~~~~~~~~~

The most basic way to create a Flight simulation requires three mandatory parameters:
a rocket, an environment, and a rail length.

.. jupyter-execute::

    from rocketpy import Environment, SolidMotor, Rocket, Flight

    # Assuming you have already defined env, motor, and rocket objects
    # (See Environment, Motor, and Rocket documentation for details)
    
    # Create a basic flight
    flight = Flight(
        rocket=rocket,           # Your Rocket object
        environment=env,         # Your Environment object
        rail_length=5.2,         # Length of launch rail in meters
    )

Once created, the Flight object automatically runs the simulation and stores 
all results internally.

Launch Parameters
~~~~~~~~~~~~~~~~~

You can customize the launch conditions by specifying additional parameters:

.. jupyter-execute::

    flight = Flight(
        rocket=rocket,
        environment=env,
        rail_length=5.2,
        inclination=85,          # Rail inclination angle (degrees from horizontal)
        heading=90,              # Launch direction (degrees from North)
    )

**Key Launch Parameters:**

- **inclination**: Rail inclination angle relative to ground (0° = horizontal, 90° = vertical)
- **heading**: Launch direction in degrees from North (0° = North, 90° = East, 180° = South, 270° = West)

Understanding Flight Parameters
-------------------------------

Coordinate Systems and Positioning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RocketPy uses a launch-centered coordinate system:

- **X-axis**: Points East (positive values = East direction)
- **Y-axis**: Points North (positive values = North direction)  
- **Z-axis**: Points upward (positive values = altitude above launch site)

The rocket's position is tracked relative to the launch site throughout the flight.

Initial Conditions
~~~~~~~~~~~~~~~~~~

The Flight class automatically calculates appropriate initial conditions based on:

- Rail inclination and heading angles
- Rocket orientation (affected by rail button positions if present)
- Environment conditions (wind, atmospheric properties)

You can also specify custom initial conditions by passing an ``initial_solution`` 
array or another Flight object to continue from a previous state.

Simulation Control Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Time Control:**

- ``max_time``: Maximum simulation duration (default: 600 seconds)
- ``max_time_step``: Maximum integration step size (default: infinity)
- ``min_time_step``: Minimum integration step size (default: 0)

**Accuracy Control:**

- ``rtol``: Relative tolerance for numerical integration (default: 1e-6)
- ``atol``: Absolute tolerance for numerical integration (default: auto-calculated)

**Simulation Behavior:**

- ``terminate_on_apogee``: Stop simulation at apogee (default: False)
- ``time_overshoot``: Allow time step overshoot for efficiency (default: True)

Accessing Simulation Results
----------------------------

Once a Flight simulation is complete, you can access a wealth of data about 
the rocket's trajectory and performance.

Trajectory Data
~~~~~~~~~~~~~~~

Basic position and velocity data:

.. jupyter-execute::

    # Position coordinates (as functions of time)
    x_trajectory = flight.x          # East coordinate (m)
    y_trajectory = flight.y          # North coordinate (m)
    altitude = flight.z              # Altitude above launch site (m)
    
    # Velocity components (as functions of time)
    vx = flight.vx                   # East velocity (m/s)
    vy = flight.vy                   # North velocity (m/s)
    vz = flight.vz                   # Vertical velocity (m/s)
    
    # Access specific values at given times
    altitude_at_10s = flight.z(10)   # Altitude at t=10 seconds
    max_altitude = flight.apogee     # Maximum altitude reached

Key Flight Events
~~~~~~~~~~~~~~~~~

Important events during the flight:

.. jupyter-execute::

    # Rail departure
    rail_departure_time = flight.out_of_rail_time
    rail_departure_velocity = flight.out_of_rail_velocity
    
    # Apogee
    apogee_time = flight.apogee_time
    apogee_altitude = flight.apogee
    apogee_coordinates = (flight.apogee_x, flight.apogee_y)
    
    # Landing/Impact
    impact_time = flight.t_final
    impact_velocity = flight.impact_velocity
    impact_coordinates = (flight.x_impact, flight.y_impact)

Forces and Accelerations
~~~~~~~~~~~~~~~~~~~~~~~~

The Flight object provides access to all forces and accelerations acting on the rocket:

.. jupyter-execute::

    # Linear accelerations in inertial frame (m/s²)
    ax = flight.ax                   # East acceleration
    ay = flight.ay                   # North acceleration  
    az = flight.az                   # Vertical acceleration
    
    # Aerodynamic forces in body frame (N)
    R1 = flight.R1                   # X-axis aerodynamic force
    R2 = flight.R2                   # Y-axis aerodynamic force
    R3 = flight.R3                   # Z-axis aerodynamic force (drag)
    
    # Aerodynamic moments in body frame (N⋅m)
    M1 = flight.M1                   # Roll moment
    M2 = flight.M2                   # Pitch moment
    M3 = flight.M3                   # Yaw moment

Attitude and Orientation
~~~~~~~~~~~~~~~~~~~~~~~~

Rocket orientation throughout the flight:

.. jupyter-execute::

    # Euler parameters (quaternions)
    e0, e1, e2, e3 = flight.e0, flight.e1, flight.e2, flight.e3
    
    # Angular velocities in body frame (rad/s)
    w1 = flight.w1                   # Roll rate
    w2 = flight.w2                   # Pitch rate  
    w3 = flight.w3                   # Yaw rate
    
    # Derived attitude angles
    attitude_angle = flight.attitude_angle
    path_angle = flight.path_angle

Performance Metrics
~~~~~~~~~~~~~~~~~~~

Key performance indicators:

.. jupyter-execute::

    # Velocity and speed
    total_speed = flight.speed
    mach_number = flight.mach_number
    
    # Stability indicators
    static_margin = flight.static_margin
    stability_margin = flight.stability_margin
    
    # Rail button forces (if present)
    max_rail_force_1 = flight.max_rail_button1_normal_force
    max_rail_force_2 = flight.max_rail_button2_normal_force

Plotting Flight Data
--------------------

The Flight class provides comprehensive plotting capabilities through the 
``plots`` attribute.

Trajectory Visualization
~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    # 3D trajectory plot
    flight.plots.trajectory_3d()
    
    # 2D trajectory (ground track)
    flight.plots.linear_kinematics_data()

Flight Data Plots
~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    # Velocity and acceleration plots
    flight.plots.linear_kinematics_data()
    
    # Attitude and angular motion
    flight.plots.attitude_data()
    flight.plots.angular_kinematics_data()
    
    # Flight path and orientation
    flight.plots.flight_path_angle_data()

Forces and Moments
~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    # Aerodynamic forces
    flight.plots.aerodynamic_forces()
    
    # Rail button forces (if applicable)
    flight.plots.rail_buttons_forces()

Energy Analysis
~~~~~~~~~~~~~~~

.. jupyter-execute::

    # Energy plots
    flight.plots.energy_data()
    
    # Stability analysis
    flight.plots.stability_and_control_data()

Comprehensive Analysis
~~~~~~~~~~~~~~~~~~~~~~

For a complete overview of all plots:

.. jupyter-execute::

    # Show all available plots
    flight.all_info()

Printing Flight Information
---------------------------

The Flight class also provides detailed text output through the ``prints`` attribute.

Flight Summary
~~~~~~~~~~~~~~

.. jupyter-execute::

    # Complete flight information
    flight.info()
    
    # All detailed information
    flight.all_info()

Specific Information Sections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    # Initial conditions
    flight.prints.initial_conditions()
    
    # Wind and environment conditions
    flight.prints.surface_wind_conditions()
    
    # Launch rail information
    flight.prints.launch_rail_conditions()
    
    # Rail departure conditions
    flight.prints.out_of_rail_conditions()
    
    # Motor burn out conditions
    flight.prints.burn_out_conditions()
    
    # Apogee conditions
    flight.prints.apogee_conditions()
    
    # Landing/impact conditions
    flight.prints.impact_conditions()
    
    # Maximum values during flight
    flight.prints.maximum_values()

Advanced Features
-----------------

Multi-Stage Rockets and Complex Simulations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For multi-stage rockets or complex scenarios, you can use the ``initial_solution`` 
parameter to chain simulations:

.. jupyter-execute::

    # First stage flight
    first_stage_flight = Flight(
        rocket=first_stage_rocket,
        environment=env,
        rail_length=5.2,
        max_time=60,
        terminate_on_apogee=False
    )
    
    # Second stage flight (continuing from first stage)
    second_stage_flight = Flight(
        rocket=second_stage_rocket,
        environment=env,
        rail_length=0,  # Already in flight
        initial_solution=first_stage_flight,  # Continue from previous state
        max_time=300
    )

Custom Equations of Motion
~~~~~~~~~~~~~~~~~~~~~~~~~~

RocketPy supports different sets of equations of motion:

.. jupyter-execute::

    # Standard 6-DOF equations (default)
    flight_6dof = Flight(
        rocket=rocket,
        environment=env,
        rail_length=5.2,
        equations_of_motion="standard"
    )
    
    # Simplified solid propulsion equations (legacy)
    flight_simple = Flight(
        rocket=rocket,
        environment=env,
        rail_length=5.2,
        equations_of_motion="solid_propulsion"
    )

Integration Method Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can choose different numerical integration methods:

.. jupyter-execute::

    # High-accuracy integration (default)
    flight_accurate = Flight(
        rocket=rocket,
        environment=env,
        rail_length=5.2,
        ode_solver="LSODA"
    )
    
    # Fast integration for quick simulations
    flight_fast = Flight(
        rocket=rocket,
        environment=env,
        rail_length=5.2,
        ode_solver="RK45"
    )

Data Export and Analysis
------------------------

Exporting Flight Data
~~~~~~~~~~~~~~~~~~~~~

You can export flight data for external analysis:

.. jupyter-execute::

    # Convert to dictionary format
    flight_data = flight.to_dict(include_outputs=True)
    # NOTE: RocketPy offers an unofficial json serializer, see rocketpy._encoders for details
    
    # Access specific data arrays
    time_array = flight.time
    altitude_array = flight.z.y_array
    velocity_array = flight.speed.y_array

Working with Function Objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most Flight data is stored as Function objects that can be evaluated at any time:

.. jupyter-execute::

    # Evaluate at specific times
    altitude_at_5s = flight.z(5.0)
    velocity_at_10s = flight.speed(10.0)
    
    # Get data arrays
    time_points = flight.z.x_array
    altitude_points = flight.z.y_array
    
    # Interpolate between points
    custom_times = [1, 5, 10, 20, 30]
    custom_altitudes = [flight.z(t) for t in custom_times]

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Integration Problems:**

- Reduce ``max_time_step`` for more accuracy
- Increase ``rtol`` for faster but less accurate simulations
- Check rocket and environment definitions for unrealistic values

**Missing Events:**

- Ensure ``max_time`` is sufficient for complete flight
- Verify parachute trigger conditions
- Check for premature termination conditions
- You can set ``verbose=True`` in the Flight constructor to get detailed logs during simulation

**Performance Issues:**

- Set ``time_overshoot=True`` for better performance
- Use simpler integration methods for quick runs
- Consider reducing the complexity of atmospheric models
