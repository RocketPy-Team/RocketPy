# pylint: disable=too-many-locals,too-many-statements
import numpy as np

from ...mathutils import Matrix, Vector


def _compute_drag_7d_inputs(
    flight,
    stream_velocity_body,
    stream_speed,
    stream_mach,
    density,
    dynamic_viscosity,
):
    """Compute drag-model inputs in the order expected by RocketPy drag functions.

    Parameters
    ----------
    flight : Flight
        Flight object providing rocket geometry.
    stream_velocity_body : Vector
        Freestream velocity expressed in the body frame.
    stream_speed : float
        Freestream speed magnitude in m/s.
    stream_mach : float
        Freestream Mach number.
    density : float
        Atmospheric density in kg/m^3.
    dynamic_viscosity : float
        Atmospheric dynamic viscosity in Pa·s.

    Returns
    -------
    tuple of float
        ``(alpha, beta, mach, reynolds)`` where ``alpha`` and ``beta`` are the
        aerodynamic angles, ``mach`` is the supplied Mach number, and ``reynolds``
        is the Reynolds number based on rocket diameter.
    """
    aerodynamic_stream_velocity = -stream_velocity_body
    alpha = np.arctan2(aerodynamic_stream_velocity[1], aerodynamic_stream_velocity[2])
    beta = np.arctan2(aerodynamic_stream_velocity[0], aerodynamic_stream_velocity[2])
    reynolds = (
        density * stream_speed * (2 * flight.rocket.radius) / dynamic_viscosity
        if dynamic_viscosity > 0
        else 0
    )
    return alpha, beta, stream_mach, reynolds


def udot_rail1(flight, t, u, post_processing=False):
    """Compute the 1-DOF rail-flight state derivative.

    The rail model advances translation along the launch rail while keeping the
    attitude and angular rates fixed.

    Parameters
    ----------
    flight : Flight
        Flight object containing the environment, rocket, and post-processing state.
    t : float
        Time in seconds.
    u : list
        State vector ``[x, y, z, vx, vy, vz, e0, e1, e2, e3, omega1, omega2, omega3]``.
    post_processing : bool, optional
        If ``True``, updates the flight post-processing buffer. Default is ``False``.

    Returns
    -------
    list
        State derivative ``[vx, vy, vz, ax, ay, az, e0dot, e1dot, e2dot, e3dot,
        alpha1, alpha2, alpha3]``.
    """
    # Retrieve integration data
    _, _, z, vx, vy, vz, e0, e1, e2, e3, _, _, _ = u

    # Retrieve important quantities
    # Mass
    total_mass_at_t = flight.rocket.total_mass.get_value_opt(t)

    # Get freestream speed
    free_stream_velocity = Vector(
        [
            flight.env.wind_velocity_x.get_value_opt(z) - vx,
            flight.env.wind_velocity_y.get_value_opt(z) - vy,
            -vz,
        ]
    )
    free_stream_speed = abs(free_stream_velocity)
    free_stream_mach = free_stream_speed / flight.env.speed_of_sound.get_value_opt(z)
    rho = flight.env.density.get_value_opt(z)
    stream_velocity_body = (
        Matrix.transformation([e0, e1, e2, e3]).transpose @ free_stream_velocity
    )
    dynamic_viscosity = flight.env.dynamic_viscosity.get_value_opt(z)
    alpha, beta, mach, reynolds = _compute_drag_7d_inputs(
        flight,
        stream_velocity_body,
        free_stream_speed,
        free_stream_mach,
        rho,
        dynamic_viscosity,
    )
    drag_coeff = flight.rocket.power_on_drag_7d(alpha, beta, mach, reynolds, 0, 0, 0)

    # Calculate Forces
    pressure = flight.env.pressure.get_value_opt(z)
    net_thrust = max(
        flight.rocket.motor.thrust.get_value_opt(t)
        + flight.rocket.motor.pressure_thrust(pressure),
        0,
    )
    R3 = -0.5 * rho * (free_stream_speed**2) * flight.rocket.area * (drag_coeff)

    # Calculate Linear acceleration
    a3 = (R3 + net_thrust) / total_mass_at_t - (
        e0**2 - e1**2 - e2**2 + e3**2
    ) * flight.env.gravity.get_value_opt(z)
    if a3 > 0:
        ax = 2 * (e1 * e3 + e0 * e2) * a3
        ay = 2 * (e2 * e3 - e0 * e1) * a3
        az = (1 - 2 * (e1**2 + e2**2)) * a3
    else:
        ax, ay, az = 0, 0, 0

    if post_processing:
        # Use u_dot post processing code for forces, moments and env data
        flight.u_dot_generalized(t, u, post_processing=True)
        # Save feasible accelerations
        flight._Flight__post_processed_variables[-1][1:7] = [ax, ay, az, 0, 0, 0]

    return [vx, vy, vz, ax, ay, az, 0, 0, 0, 0, 0, 0, 0]


def udot_rail2(flight, t, u, post_processing=False):  # pragma: no cover
    """Compute the rail-flight derivative through the generalized solver.

    This function is a placeholder for a dedicated 3DOF rail model. The current
    implementation simply delegates to :func:`u_dot_generalized`.

    Parameters
    ----------
    flight : Flight
        Flight object containing the environment, rocket, and post-processing state.
    t : float
        Time in seconds.
    u : list
        State vector ``[x, y, z, vx, vy, vz, e0, e1, e2, e3, omega1, omega2, omega3]``.
    post_processing : bool, optional
        If ``True``, updates the flight post-processing buffer. Default is ``False``.

    Returns
    -------
    list
        State derivative returned by :func:`u_dot_generalized`.
    """
    # Hey! We will finish this function later, now we just can use u_dot
    return flight.u_dot_generalized(t, u, post_processing=post_processing)


def u_dot(flight, t, u, post_processing=False):
    """Compute the simplified 6DOF free-flight derivative.

    This solver is used for powered ascent and coasting descent without parachutes.
    It uses a simplified rotational treatment relative to
    :func:`u_dot_generalized`.

    Parameters
    ----------
    flight : Flight
        Flight object containing the environment, rocket, and post-processing state.
    t : float
        Time in seconds.
    u : list
        State vector ``[x, y, z, vx, vy, vz, e0, e1, e2, e3, omega1, omega2, omega3]``.
    post_processing : bool, optional
        If ``True``, updates the flight post-processing buffer. Default is ``False``.

    Returns
    -------
    list
        State derivative ``[vx, vy, vz, ax, ay, az, e0dot, e1dot, e2dot, e3dot,
        alpha1, alpha2, alpha3]``.
    """

    # Retrieve integration data
    _, _, z, vx, vy, vz, e0, e1, e2, e3, omega1, omega2, omega3 = u
    # Determine lift force and moment
    omega1, omega2, omega3 = 0, 0, 0
    R1, R2, M1, M2, M3 = 0, 0, 0, 0, 0
    # Thrust correction parameters
    pressure = flight.env.pressure.get_value_opt(z)
    # Determine current behavior
    if flight.rocket.motor.burn_start_time < t < flight.rocket.motor.burn_out_time:
        # Motor burning
        # Retrieve important motor quantities
        # Inertias
        motor_I_33_at_t = flight.rocket.motor.I_33.get_value_opt(t)
        motor_I_11_at_t = flight.rocket.motor.I_11.get_value_opt(t)
        motor_I_33_derivative_at_t = flight.rocket.motor.I_33.differentiate(t, dx=1e-6)
        motor_I_11_derivative_at_t = flight.rocket.motor.I_11.differentiate(t, dx=1e-6)
        # Mass
        mass_flow_rate_at_t = flight.rocket.motor.mass_flow_rate.get_value_opt(t)
        propellant_mass_at_t = flight.rocket.motor.propellant_mass.get_value_opt(t)
        # Thrust

        net_thrust = max(
            flight.rocket.motor.thrust.get_value_opt(t)
            + flight.rocket.motor.pressure_thrust(pressure),
            0,
        )
        # Off center moment
        M1 += flight.rocket.thrust_eccentricity_y * net_thrust
        M2 -= flight.rocket.thrust_eccentricity_x * net_thrust
    else:
        # Motor stopped
        # Inertias
        (
            motor_I_33_at_t,
            motor_I_11_at_t,
            motor_I_33_derivative_at_t,
            motor_I_11_derivative_at_t,
        ) = (0, 0, 0, 0)
        # Mass
        mass_flow_rate_at_t, propellant_mass_at_t = 0, 0
        # thrust
        net_thrust = 0

    # Retrieve important quantities
    # Inertias
    rocket_dry_I_33 = flight.rocket.dry_I_33
    rocket_dry_I_11 = flight.rocket.dry_I_11
    # Mass
    rocket_dry_mass = flight.rocket.dry_mass  # already with motor's dry mass
    total_mass_at_t = propellant_mass_at_t + rocket_dry_mass
    mu = (propellant_mass_at_t * rocket_dry_mass) / (
        propellant_mass_at_t + rocket_dry_mass
    )
    # Geometry
    # b = -flight.rocket.distance_rocket_propellant
    b = (
        -(
            flight.rocket.center_of_propellant_position.get_value_opt(0)
            - flight.rocket.center_of_dry_mass_position
        )
        * flight.rocket._csys
    )
    c = flight.rocket.nozzle_to_cdm
    nozzle_radius = flight.rocket.motor.nozzle_radius
    # Prepare transformation matrix
    a11 = 1 - 2 * (e2**2 + e3**2)
    a12 = 2 * (e1 * e2 - e0 * e3)
    a13 = 2 * (e1 * e3 + e0 * e2)
    a21 = 2 * (e1 * e2 + e0 * e3)
    a22 = 1 - 2 * (e1**2 + e3**2)
    a23 = 2 * (e2 * e3 - e0 * e1)
    a31 = 2 * (e1 * e3 - e0 * e2)
    a32 = 2 * (e2 * e3 + e0 * e1)
    a33 = 1 - 2 * (e1**2 + e2**2)
    # Transformation matrix: (123) -> (XYZ)
    K = Matrix([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])
    Kt = K.transpose

    # Calculate Forces and Moments
    # Get freestream speed
    wind_velocity_x = flight.env.wind_velocity_x.get_value_opt(z)
    wind_velocity_y = flight.env.wind_velocity_y.get_value_opt(z)
    speed_of_sound = flight.env.speed_of_sound.get_value_opt(z)
    free_stream_velocity = Vector([wind_velocity_x - vx, wind_velocity_y - vy, -vz])
    free_stream_speed = abs(free_stream_velocity)
    free_stream_mach = free_stream_speed / speed_of_sound
    stream_velocity_body = Kt @ free_stream_velocity

    # Determine aerodynamics forces
    # Determine Drag Force
    rho = flight.env.density.get_value_opt(z)
    dynamic_viscosity = flight.env.dynamic_viscosity.get_value_opt(z)
    alpha, beta, mach, reynolds = _compute_drag_7d_inputs(
        flight,
        stream_velocity_body,
        free_stream_speed,
        free_stream_mach,
        rho,
        dynamic_viscosity,
    )
    if t < flight.rocket.motor.burn_out_time:
        drag_coeff = flight.rocket.power_on_drag_7d(
            alpha,
            beta,
            mach,
            reynolds,
            omega1,
            omega2,
            omega3,
        )
    else:
        drag_coeff = flight.rocket.power_off_drag_7d(
            alpha,
            beta,
            mach,
            reynolds,
            omega1,
            omega2,
            omega3,
        )
    R3 = -0.5 * rho * (free_stream_speed**2) * flight.rocket.area * drag_coeff
    for air_brakes in flight.rocket.air_brakes:
        if air_brakes.deployment_level > 0:
            air_brakes_cd = air_brakes.drag_coefficient.get_value_opt(
                air_brakes.deployment_level, free_stream_mach
            )
            air_brakes_force = (
                -0.5
                * rho
                * (free_stream_speed**2)
                * air_brakes.reference_area
                * air_brakes_cd
            )
            if air_brakes.override_rocket_drag:
                R3 = air_brakes_force  # Substitutes rocket drag coefficient
            else:
                R3 += air_brakes_force
    # Off center moment
    M1 += flight.rocket.cp_eccentricity_y * R3
    M2 -= flight.rocket.cp_eccentricity_x * R3
    # Get rocket velocity in body frame
    vx_b = a11 * vx + a21 * vy + a31 * vz
    vy_b = a12 * vx + a22 * vy + a32 * vz
    vz_b = a13 * vx + a23 * vy + a33 * vz
    # Calculate lift and moment for each component of the rocket
    velocity_in_body_frame = Vector([vx_b, vy_b, vz_b])
    w = Vector([omega1, omega2, omega3])
    for aero_surface, _ in flight.rocket.aerodynamic_surfaces:
        # Component cp relative to CDM in body frame
        comp_cp = flight.rocket.surfaces_cp_to_cdm[aero_surface]
        # Component absolute velocity in body frame
        comp_vb = velocity_in_body_frame + (w ^ comp_cp)
        # Wind velocity at component altitude
        comp_z = z + (K @ comp_cp).z
        comp_wind_vx = flight.env.wind_velocity_x.get_value_opt(comp_z)
        comp_wind_vy = flight.env.wind_velocity_y.get_value_opt(comp_z)
        # Component freestream velocity in body frame
        comp_wind_vb = Kt @ Vector([comp_wind_vx, comp_wind_vy, 0])
        comp_stream_velocity = comp_wind_vb - comp_vb
        comp_stream_speed = abs(comp_stream_velocity)
        comp_stream_mach = comp_stream_speed / speed_of_sound
        # Forces and moments
        X, Y, Z, M, N, L = aero_surface.compute_forces_and_moments(
            comp_stream_velocity,
            comp_stream_speed,
            comp_stream_mach,
            rho,
            comp_cp,
            w,
            flight.env.density,
            flight.env.dynamic_viscosity,
            comp_z,
        )
        R1 += X
        R2 += Y
        R3 += Z
        M1 += M
        M2 += N
        M3 += L
    # Off center moment
    M3 += flight.rocket.cp_eccentricity_x * R2 - flight.rocket.cp_eccentricity_y * R1

    # Calculate derivatives
    # Angular acceleration
    alpha1 = (
        M1
        - (
            omega2
            * omega3
            * (
                rocket_dry_I_33
                + motor_I_33_at_t
                - rocket_dry_I_11
                - motor_I_11_at_t
                - mu * b**2
            )
            + omega1
            * (
                (
                    motor_I_11_derivative_at_t
                    + mass_flow_rate_at_t
                    * (rocket_dry_mass - 1)
                    * (b / total_mass_at_t) ** 2
                )
                - mass_flow_rate_at_t
                * ((nozzle_radius / 2) ** 2 + (c - b * mu / rocket_dry_mass) ** 2)
            )
        )
    ) / (rocket_dry_I_11 + motor_I_11_at_t + mu * b**2)
    alpha2 = (
        M2
        - (
            omega1
            * omega3
            * (
                rocket_dry_I_11
                + motor_I_11_at_t
                + mu * b**2
                - rocket_dry_I_33
                - motor_I_33_at_t
            )
            + omega2
            * (
                (
                    motor_I_11_derivative_at_t
                    + mass_flow_rate_at_t
                    * (rocket_dry_mass - 1)
                    * (b / total_mass_at_t) ** 2
                )
                - mass_flow_rate_at_t
                * ((nozzle_radius / 2) ** 2 + (c - b * mu / rocket_dry_mass) ** 2)
            )
        )
    ) / (rocket_dry_I_11 + motor_I_11_at_t + mu * b**2)
    alpha3 = (
        M3
        - omega3
        * (motor_I_33_derivative_at_t - mass_flow_rate_at_t * (nozzle_radius**2) / 2)
    ) / (rocket_dry_I_33 + motor_I_33_at_t)
    # Euler parameters derivative
    e0dot = 0.5 * (-omega1 * e1 - omega2 * e2 - omega3 * e3)
    e1dot = 0.5 * (omega1 * e0 + omega3 * e2 - omega2 * e3)
    e2dot = 0.5 * (omega2 * e0 - omega3 * e1 + omega1 * e3)
    e3dot = 0.5 * (omega3 * e0 + omega2 * e1 - omega1 * e2)
    # Linear acceleration
    L = [
        (
            R1
            - b * propellant_mass_at_t * (omega2**2 + omega3**2)
            - 2 * c * mass_flow_rate_at_t * omega2
        )
        / total_mass_at_t,
        (
            R2
            + b * propellant_mass_at_t * (alpha3 + omega1 * omega2)
            + 2 * c * mass_flow_rate_at_t * omega1
        )
        / total_mass_at_t,
        (R3 - b * propellant_mass_at_t * (alpha2 - omega1 * omega3) + net_thrust)
        / total_mass_at_t,
    ]
    ax, ay, az = K @ Vector(L)
    az -= flight.env.gravity.get_value_opt(z)  # Include gravity

    # Coriolis acceleration
    _, w_earth_y, w_earth_z = flight.env.earth_rotation_vector
    ax -= 2 * (vz * w_earth_y - vy * w_earth_z)
    ay -= 2 * (vx * w_earth_z)
    az -= 2 * (-vx * w_earth_y)

    # Create u_dot
    u_dot = [
        vx,
        vy,
        vz,
        ax,
        ay,
        az,
        e0dot,
        e1dot,
        e2dot,
        e3dot,
        alpha1,
        alpha2,
        alpha3,
    ]

    if post_processing:
        flight._Flight__post_processed_variables.append(
            [
                t,
                ax,
                ay,
                az,
                alpha1,
                alpha2,
                alpha3,
                R1,
                R2,
                R3,
                M1,
                M2,
                M3,
                net_thrust,
            ]
        )
    return u_dot


def u_dot_generalized_3dof(flight, t, u, post_processing=False):
    """Compute the 3DOF free-flight derivative with weathercocking.

    The model advances translational motion with variable mass while using a kinematic
    alignment rule for the attitude evolution. It does not integrate full rigid-body
    rotational dynamics.

    Parameters
    ----------
    flight : Flight
        Flight object containing the environment, rocket, and post-processing state.
    t : float
        Time in seconds.
    u : list
        State vector ``[x, y, z, vx, vy, vz, e0, e1, e2, e3, omega1, omega2, omega3]``.
    post_processing : bool, optional
        If ``True``, updates the flight post-processing buffer. Default is ``False``.

    Returns
    -------
    list
        State derivative ``[vx, vy, vz, ax, ay, az, e0_dot, e1_dot, e2_dot, e3_dot,
        alpha1, alpha2, alpha3]``.
    """
    # Unpack state
    _, _, z, vx, vy, vz, e0, e1, e2, e3, omega1, omega2, omega3 = u

    # Define vectors
    v = Vector([vx, vy, vz])
    e = [e0, e1, e2, e3]
    w = Vector([omega1, omega2, omega3])

    # Mass and transformation
    total_mass = flight.rocket.total_mass.get_value_opt(t)
    K = Matrix.transformation(e)
    Kt = K.transpose

    # Atmospheric and wind data
    rho = flight.env.density.get_value_opt(z)
    wind_vx = flight.env.wind_velocity_x.get_value_opt(z)
    wind_vy = flight.env.wind_velocity_y.get_value_opt(z)
    wind_velocity = Vector([wind_vx, wind_vy, 0])

    free_stream_velocity = wind_velocity - v
    free_stream_speed = abs(free_stream_velocity)
    speed_of_sound = flight.env.speed_of_sound.get_value_opt(z)
    mach = free_stream_speed / speed_of_sound
    stream_velocity_body = Kt @ free_stream_velocity
    dynamic_viscosity = flight.env.dynamic_viscosity.get_value_opt(z)
    alpha, beta, mach, reynolds = _compute_drag_7d_inputs(
        flight,
        stream_velocity_body,
        free_stream_speed,
        mach,
        rho,
        dynamic_viscosity,
    )

    # Drag computation
    if t < flight.rocket.motor.burn_out_time:
        cd = flight.rocket.power_on_drag_7d(
            alpha, beta, mach, reynolds, omega1, omega2, omega3
        )
    else:
        cd = flight.rocket.power_off_drag_7d(
            alpha, beta, mach, reynolds, omega1, omega2, omega3
        )

    R1, R2 = 0, 0
    R3 = -0.5 * rho * free_stream_speed**2 * flight.rocket.area * cd

    for air_brake in flight.rocket.air_brakes:
        if air_brake.deployment_level > 0:
            ab_cd = air_brake.drag_coefficient.get_value_opt(
                air_brake.deployment_level, mach
            )
            ab_force = (
                -0.5 * rho * free_stream_speed**2 * air_brake.reference_area * ab_cd
            )
            if air_brake.override_rocket_drag:
                R3 = ab_force
            else:
                R3 += ab_force

    # Velocity in body frame
    vb_body = Kt @ v

    for surface, _ in flight.rocket.aerodynamic_surfaces:
        cp = flight.rocket.surfaces_cp_to_cdm[surface]
        vb_component = vb_body + (w ^ cp)

        comp_z = z + (K @ cp).z
        wind_cx = flight.env.wind_velocity_x.get_value_opt(comp_z)
        wind_cy = flight.env.wind_velocity_y.get_value_opt(comp_z)
        wind_body = Kt @ Vector([wind_cx, wind_cy, 0])

        rel_velocity = wind_body - vb_component
        rel_speed = abs(rel_velocity)
        rel_mach = rel_speed / speed_of_sound

        fx, fy, fz, *_ = surface.compute_forces_and_moments(
            rel_velocity,
            rel_speed,
            rel_mach,
            rho,
            cp,
            w,
            flight.env.density,
            flight.env.dynamic_viscosity,
            comp_z,
        )
        R1 += fx
        R2 += fy
        R3 += fz

    # Thrust and weight
    # Calculate net thrust including pressure thrust correction if motor is burning
    if flight.rocket.motor.burn_start_time < t < flight.rocket.motor.burn_out_time:
        pressure = flight.env.pressure.get_value_opt(z)
        net_thrust = max(
            flight.rocket.motor.thrust.get_value_opt(t)
            + flight.rocket.motor.pressure_thrust(pressure),
            0,
        )
    else:
        net_thrust = 0
    gravity = flight.env.gravity.get_value_opt(z)
    weight_body = Kt @ Vector([0, 0, -total_mass * gravity])

    total_force = Vector([0, 0, net_thrust]) + weight_body + Vector([R1, R2, R3])

    # Dynamics
    v_dot = K @ (total_force / total_mass)
    r_dot = [vx, vy, vz]
    # Weathercocking: evolve body axis direction toward relative wind
    # The body z-axis (attitude vector) should align with -freestream_velocity
    weathercock_coeff = getattr(flight.rocket, "weathercock_coeff", 0.0)
    if weathercock_coeff > 0 and free_stream_speed > 1e-6:
        # Current body z-axis in inertial frame (attitude vector)
        # From rotation matrix: column 3 gives the body z-axis in inertial frame
        body_z_inertial = Vector(
            [
                2 * (e1 * e3 + e0 * e2),
                2 * (e2 * e3 - e0 * e1),
                1 - 2 * (e1**2 + e2**2),
            ]
        )

        # Desired direction: opposite of freestream velocity (into the wind)
        # This is the direction the rocket nose should point
        # Division by free_stream_speed ensures the result is a unit vector
        desired_direction = -free_stream_velocity / free_stream_speed

        # Compute rotation axis (cross product of current and desired)
        rotation_axis = body_z_inertial ^ desired_direction
        rotation_axis_mag = abs(rotation_axis)

        # Determine omega_body based on alignment state
        omega_body = None

        if rotation_axis_mag > 1e-8:
            # Normal case: compute angular velocity from misalignment
            rotation_axis = rotation_axis / rotation_axis_mag

            # The magnitude of the cross product of two unit vectors equals
            # the sine of the angle between them
            sin_angle = min(1.0, max(-1.0, rotation_axis_mag))

            # Angular velocity magnitude proportional to misalignment angle
            omega_mag = weathercock_coeff * sin_angle

            # Angular velocity in inertial frame, then transform to body frame
            omega_body = Kt @ (rotation_axis * omega_mag)
        else:
            # Check if aligned or anti-aligned using dot product
            dot = body_z_inertial @ desired_direction
            if dot < -0.999:  # Anti-aligned
                # Choose an arbitrary perpendicular axis
                x_axis = Vector([1.0, 0.0, 0.0])
                perp_axis = body_z_inertial ^ x_axis
                if abs(perp_axis) < 1e-6:
                    y_axis = Vector([0.0, 1.0, 0.0])
                    perp_axis = body_z_inertial ^ y_axis
                    if abs(perp_axis) < 1e-6:
                        raise ValueError(
                            "Cannot determine a valid rotation axis: "
                            "body_z_inertial is parallel to both x and y axes."
                        )
                rotation_axis = perp_axis.unit_vector
                # 180 degree rotation: sin(angle) = 1
                omega_mag = weathercock_coeff * 1.0
                omega_body = Kt @ (rotation_axis * omega_mag)
            # else: aligned (dot > 0.999) - no rotation needed, omega_body stays None

        # Compute quaternion derivatives from omega_body
        if omega_body is not None:
            omega1_wc, omega2_wc, omega3_wc = (
                omega_body.x,
                omega_body.y,
                omega_body.z,
            )
            e0_dot = 0.5 * (-omega1_wc * e1 - omega2_wc * e2 - omega3_wc * e3)
            e1_dot = 0.5 * (omega1_wc * e0 + omega3_wc * e2 - omega2_wc * e3)
            e2_dot = 0.5 * (omega2_wc * e0 - omega3_wc * e1 + omega1_wc * e3)
            e3_dot = 0.5 * (omega3_wc * e0 + omega2_wc * e1 - omega1_wc * e2)
            e_dot = [e0_dot, e1_dot, e2_dot, e3_dot]
        else:
            e_dot = [0, 0, 0, 0]
        w_dot = [0, 0, 0]  # No angular acceleration in 3DOF model
    else:
        # No weathercocking or negligible freestream speed
        e_dot = [0, 0, 0, 0]
        w_dot = [0, 0, 0]

    u_dot = [*r_dot, *v_dot, *e_dot, *w_dot]

    if post_processing:
        flight._Flight__post_processed_variables.append(
            [t, *v_dot, *w_dot, R1, R2, R3, 0, 0, 0, net_thrust]
        )

    return u_dot


def u_dot_generalized(flight, t, u, post_processing=False):
    """Compute the full 6DOF generalized flight derivative.

    This is the highest-fidelity rigid-body flight solver in this module. It accounts
    for variable mass, changing inertia, aerodynamic surface loads, thrust eccentricity,
    and Earth-rotation effects.

    Parameters
    ----------
    flight : Flight
        Flight object containing the environment, rocket, and post-processing state.
    t : float
        Time in seconds.
    u : list
        State vector ``[x, y, z, vx, vy, vz, e0, e1, e2, e3, omega1, omega2, omega3]``.
    post_processing : bool, optional
        If ``True``, updates the flight post-processing buffer. Default is ``False``.

    Returns
    -------
    list
        State derivative ``[vx, vy, vz, ax, ay, az, e0_dot, e1_dot, e2_dot, e3_dot,
        alpha1, alpha2, alpha3]``.
    """
    # Retrieve integration data
    _, _, z, vx, vy, vz, e0, e1, e2, e3, omega1, omega2, omega3 = u

    # Create necessary vectors
    # r = Vector([x, y, z])  # CDM position vector
    v = Vector([vx, vy, vz])  # CDM velocity vector
    e = [e0, e1, e2, e3]  # Euler parameters/quaternions
    w = Vector([omega1, omega2, omega3])  # Angular velocity vector

    # Retrieve necessary quantities
    ## Rocket mass
    total_mass = flight.rocket.total_mass.get_value_opt(t)
    total_mass_dot = flight.rocket.total_mass_flow_rate.get_value_opt(t)
    total_mass_ddot = flight.rocket.total_mass_flow_rate.differentiate_complex_step(t)
    ## CM position vector and time derivatives relative to CDM in body frame
    r_CM_z = flight.rocket.com_to_cdm_function
    r_CM_t = r_CM_z.get_value_opt(t)
    r_CM = Vector([0, 0, r_CM_t])
    r_CM_dot = Vector([0, 0, r_CM_z.differentiate_complex_step(t)])
    r_CM_ddot = Vector([0, 0, r_CM_z.differentiate(t, order=2)])
    ## Nozzle position vector
    r_NOZ = Vector([0, 0, flight.rocket.nozzle_to_cdm])
    ## Nozzle gyration tensor
    S_nozzle = flight.rocket.nozzle_gyration_tensor
    ## Inertia tensor
    inertia_tensor = flight.rocket.get_inertia_tensor_at_time(t)
    ## Inertia tensor time derivative in the body frame
    I_dot = flight.rocket.get_inertia_tensor_derivative_at_time(t)

    # Calculate the Inertia tensor relative to CM
    H = (r_CM.cross_matrix @ -r_CM.cross_matrix) * total_mass
    I_CM = inertia_tensor - H

    # Prepare transformation matrices
    K = Matrix.transformation(e)
    Kt = K.transpose

    # Compute aerodynamic forces and moments
    R1, R2, R3, M1, M2, M3 = 0, 0, 0, 0, 0, 0

    ## Drag force
    rho = flight.env.density.get_value_opt(z)
    wind_velocity_x = flight.env.wind_velocity_x.get_value_opt(z)
    wind_velocity_y = flight.env.wind_velocity_y.get_value_opt(z)
    wind_velocity = Vector([wind_velocity_x, wind_velocity_y, 0])
    free_stream_velocity = wind_velocity - v
    free_stream_speed = abs(free_stream_velocity)
    speed_of_sound = flight.env.speed_of_sound.get_value_opt(z)
    free_stream_mach = free_stream_speed / speed_of_sound
    stream_velocity_body = Kt @ free_stream_velocity
    dynamic_viscosity = flight.env.dynamic_viscosity.get_value_opt(z)
    alpha, beta, mach, reynolds = _compute_drag_7d_inputs(
        flight,
        stream_velocity_body,
        free_stream_speed,
        free_stream_mach,
        rho,
        dynamic_viscosity,
    )

    if flight.rocket.motor.burn_start_time < t < flight.rocket.motor.burn_out_time:
        pressure = flight.env.pressure.get_value_opt(z)
        net_thrust = max(
            flight.rocket.motor.thrust.get_value_opt(t)
            + flight.rocket.motor.pressure_thrust(pressure),
            0,
        )
        drag_coeff = flight.rocket.power_on_drag_7d(
            alpha,
            beta,
            mach,
            reynolds,
            omega1,
            omega2,
            omega3,
        )
    else:
        net_thrust = 0
        drag_coeff = flight.rocket.power_off_drag_7d(
            alpha,
            beta,
            mach,
            reynolds,
            omega1,
            omega2,
            omega3,
        )
    R3 += -0.5 * rho * (free_stream_speed**2) * flight.rocket.area * drag_coeff
    for air_brakes in flight.rocket.air_brakes:
        if air_brakes.deployment_level > 0:
            air_brakes_cd = air_brakes.drag_coefficient.get_value_opt(
                air_brakes.deployment_level, free_stream_mach
            )
            air_brakes_force = (
                -0.5
                * rho
                * (free_stream_speed**2)
                * air_brakes.reference_area
                * air_brakes_cd
            )
            if air_brakes.override_rocket_drag:
                R3 = air_brakes_force  # Substitutes rocket drag coefficient
            else:
                R3 += air_brakes_force
    # Get rocket velocity in body frame
    velocity_in_body_frame = Kt @ v
    # Calculate lift and moment for each component of the rocket
    for aero_surface, _ in flight.rocket.aerodynamic_surfaces:
        # Component cp relative to CDM in body frame
        comp_cp = flight.rocket.surfaces_cp_to_cdm[aero_surface]
        # Component absolute velocity in body frame
        comp_vb = velocity_in_body_frame + (w ^ comp_cp)
        # Wind velocity at component altitude
        comp_z = z + (K @ comp_cp).z
        comp_wind_vx = flight.env.wind_velocity_x.get_value_opt(comp_z)
        comp_wind_vy = flight.env.wind_velocity_y.get_value_opt(comp_z)
        # Component freestream velocity in body frame
        comp_wind_vb = Kt @ Vector([comp_wind_vx, comp_wind_vy, 0])
        comp_stream_velocity = comp_wind_vb - comp_vb
        comp_stream_speed = abs(comp_stream_velocity)
        comp_stream_mach = comp_stream_speed / speed_of_sound
        # Forces and moments
        X, Y, Z, M, N, L = aero_surface.compute_forces_and_moments(
            comp_stream_velocity,
            comp_stream_speed,
            comp_stream_mach,
            rho,
            comp_cp,
            w,
            flight.env.density,
            flight.env.dynamic_viscosity,
            comp_z,
        )
        R1 += X
        R2 += Y
        R3 += Z
        M1 += M
        M2 += N
        M3 += L

    # Off center moment
    M1 += (
        flight.rocket.cp_eccentricity_y * R3
        + flight.rocket.thrust_eccentricity_y * net_thrust
    )
    M2 -= (
        flight.rocket.cp_eccentricity_x * R3
        + flight.rocket.thrust_eccentricity_x * net_thrust
    )
    M3 += flight.rocket.cp_eccentricity_x * R2 - flight.rocket.cp_eccentricity_y * R1

    weight_in_body_frame = Kt @ Vector(
        [0, 0, -total_mass * flight.env.gravity.get_value_opt(z)]
    )

    T00 = total_mass * r_CM
    T03 = 2 * total_mass_dot * (r_NOZ - r_CM) - 2 * total_mass * r_CM_dot
    T04 = (
        Vector([0, 0, net_thrust])
        - total_mass * r_CM_ddot
        - 2 * total_mass_dot * r_CM_dot
        + total_mass_ddot * (r_NOZ - r_CM)
    )
    T05 = total_mass_dot * S_nozzle - I_dot

    T20 = (
        ((w ^ T00) ^ w) + (w ^ T03) + T04 + weight_in_body_frame + Vector([R1, R2, R3])
    )

    T21 = (
        ((inertia_tensor @ w) ^ w)
        + T05 @ w
        - (weight_in_body_frame ^ r_CM)
        + Vector([M1, M2, M3])
    )

    # Angular velocity derivative
    w_dot = I_CM.inverse @ (T21 + (T20 ^ r_CM))

    # Euler parameters derivative
    e_dot = [
        0.5 * (-omega1 * e1 - omega2 * e2 - omega3 * e3),
        0.5 * (omega1 * e0 + omega3 * e2 - omega2 * e3),
        0.5 * (omega2 * e0 - omega3 * e1 + omega1 * e3),
        0.5 * (omega3 * e0 + omega2 * e1 - omega1 * e2),
    ]

    # Velocity vector derivative + Coriolis acceleration
    w_earth = Vector(flight.env.earth_rotation_vector)
    v_dot = K @ (T20 / total_mass - (r_CM ^ w_dot)) - 2 * (w_earth ^ v)

    # Position vector derivative
    r_dot = [vx, vy, vz]

    # Create u_dot
    u_dot = [*r_dot, *v_dot, *e_dot, *w_dot]

    if post_processing:
        flight._Flight__post_processed_variables.append(
            [t, *v_dot, *w_dot, R1, R2, R3, M1, M2, M3, net_thrust]
        )

    return u_dot


def u_dot_parachute(flight, t, u, post_processing=False):
    """Compute the parachute descent derivative.

    The parachute model is a 3DOF translational approximation with drag and added-mass
    effects. Angular motion is not integrated.

    Parameters
    ----------
    flight : Flight
        Flight object containing the environment, rocket, and active parachute.
    t : float
        Time in seconds.
    u : list
        State vector ``[x, y, z, vx, vy, vz, e0, e1, e2, e3, omega1, omega2, omega3]``.
    post_processing : bool, optional
        If ``True``, updates the flight post-processing buffer. Default is ``False``.

    Returns
    -------
    list
        State derivative ``[vx, vy, vz, ax, ay, az, e0dot, e1dot, e2dot, e3dot,
        alpha1, alpha2, alpha3]``.
    """
    # Get relevant state data
    z, vx, vy, vz = u[2:6]

    # Get atmospheric data
    rho = flight.env.density.get_value_opt(z)
    wind_velocity_x = flight.env.wind_velocity_x.get_value_opt(z)
    wind_velocity_y = flight.env.wind_velocity_y.get_value_opt(z)

    # Get the mass of the rocket
    mp = flight.rocket.dry_mass

    # to = 1.2
    # eta = 1
    # Rdot = (6 * R * (1 - eta) / (1.2**6)) * (
    #     (1 - eta) * t**5 + eta * (to**3) * (t**2)
    # )
    # Rdot = 0

    # tf = 8 * nominal diameter / velocity at line stretch

    # Calculate added mass
    ma = (
        flight._active_parachute.added_mass_coefficient
        * rho
        * (2 / 3)
        * np.pi
        * flight._active_parachute.radius**2
        * flight._active_parachute.height
    )

    # Calculate freestream speed
    freestream_x = vx - wind_velocity_x
    freestream_y = vy - wind_velocity_y
    freestream_z = vz
    free_stream_speed = (freestream_x**2 + freestream_y**2 + freestream_z**2) ** 0.5

    # Determine drag force
    pseudo_drag = -0.5 * rho * flight._active_parachute.cd_s * free_stream_speed
    # pseudo_drag = pseudo_drag - ka * rho * 4 * np.pi * (R**2) * Rdot
    Dx = pseudo_drag * freestream_x  # add eta efficiency for wake
    Dy = pseudo_drag * freestream_y
    Dz = pseudo_drag * freestream_z
    ax = Dx / (mp + ma)
    ay = Dy / (mp + ma)
    az = (Dz - mp * flight.env.gravity.get_value_opt(z)) / (mp + ma)

    # Add coriolis acceleration
    _, w_earth_y, w_earth_z = flight.env.earth_rotation_vector
    ax -= 2 * (vz * w_earth_y - vy * w_earth_z)
    ay -= 2 * (vx * w_earth_z)
    az -= 2 * (-vx * w_earth_y)

    if post_processing:
        flight._Flight__post_processed_variables.append(
            [t, ax, ay, az, 0, 0, 0, Dx, Dy, Dz, 0, 0, 0, 0]
        )

    return [vx, vy, vz, ax, ay, az, 0, 0, 0, 0, 0, 0, 0]
