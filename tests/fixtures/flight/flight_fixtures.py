import numpy as np
import pytest

from rocketpy import Flight, Function, Rocket


@pytest.fixture
def flight_calisto(calisto, example_plain_env):  # old name: flight
    """A rocketpy.Flight object of the Calisto rocket. This uses the calisto
    without the aerodynamic surfaces and parachutes. The environment is the
    simplest possible, with no parameters set.

    Parameters
    ----------
    calisto : rocketpy.Rocket
        An object of the Rocket class. This is a pytest fixture too.
    example_plain_env : rocketpy.Environment
        An object of the Environment class. This is a pytest fixture too.

    Returns
    -------
    rocketpy.Flight
        A rocketpy.Flight object of the Calisto rocket in the simplest possible
        conditions.
    """
    return Flight(
        environment=example_plain_env,
        rocket=calisto,
        rail_length=5.2,
        inclination=85,
        heading=0,
        terminate_on_apogee=False,
    )


@pytest.fixture
def flight_calisto_nose_to_tail(calisto_nose_to_tail, example_plain_env):
    """A rocketpy.Flight object of the Calisto rocket. This uses the calisto
    with "nose_to_tail" coordinate system orientation, just as described in the
    calisto_nose_to_tail fixture.

    Parameters
    ----------
    calisto_nose_to_tail : rocketpy.Rocket
        An object of the Rocket class. This is a pytest fixture too.
    example_plain_env : rocketpy.Environment
        An object of the Environment class. This is a pytest fixture too.

    Returns
    -------
    rocketpy.Flight
        The Calisto rocket with the coordinate system orientation set to
        "nose_to_tail".
    """
    return Flight(
        environment=example_plain_env,
        rocket=calisto_nose_to_tail,
        rail_length=5.2,
        inclination=85,
        heading=0,
        terminate_on_apogee=False,
    )


@pytest.fixture
def flight_calisto_robust(calisto_robust, example_spaceport_env):
    """A rocketpy.Flight object of the Calisto rocket. This uses the calisto
    with the aerodynamic surfaces and parachutes. The environment is a bit more
    complex than the one in the flight_calisto fixture. This time the latitude,
    longitude and elevation are set, as well as the datum and the date. The
    location refers to the Spaceport America Cup launch site, while the date is
    set to tomorrow at noon.

    Parameters
    ----------
    calisto_robust : rocketpy.Rocket
        An object of the Rocket class. This is a pytest fixture too.
    example_spaceport_env : rocketpy.Environment
        An object of the Environment class. This is a pytest fixture too.

    Returns
    -------
    rocketpy.Flight
        A rocketpy.Flight object of the Calisto rocket in a more complex
        condition.
    """
    return Flight(
        environment=example_spaceport_env,
        rocket=calisto_robust,
        rail_length=5.2,
        inclination=85,
        heading=0,
        terminate_on_apogee=False,
    )


@pytest.fixture
def flight_calisto_nose_to_tail_robust(
    calisto_nose_to_tail_robust, example_spaceport_env
):
    return Flight(
        environment=example_spaceport_env,
        rocket=calisto_nose_to_tail_robust,
        rail_length=5.2,
        inclination=85,
        heading=0,
        terminate_on_apogee=False,
    )


@pytest.fixture
def flight_calisto_robust_solid_eom(calisto_robust, example_spaceport_env):
    """Similar to flight_calisto_robust, but with the equations of motion set to
    "solid_propulsion".
    """
    return Flight(
        environment=example_spaceport_env,
        rocket=calisto_robust,
        rail_length=5.2,
        inclination=85,
        heading=0,
        terminate_on_apogee=False,
        equations_of_motion="solid_propulsion",
    )


@pytest.fixture
def flight_calisto_liquid_modded(calisto_liquid_modded, example_plain_env):
    """A rocketpy.Flight object of the Calisto rocket modded for a liquid
    motor. The environment is the simplest possible, with no parameters set.

    Parameters
    ----------
    calisto_liquid_modded : rocketpy.Rocket
        An object of the Rocket class. This is a pytest fixture too.
    example_plain_env : rocketpy.Environment
        An object of the Environment class. This is a pytest fixture too.

    Returns
    -------
    rocketpy.Flight
        A rocketpy.Flight object.
    """
    return Flight(
        rocket=calisto_liquid_modded,
        environment=example_plain_env,
        rail_length=5,
        inclination=85,
        heading=0,
        max_time_step=0.25,
    )


@pytest.fixture
def flight_calisto_hybrid_modded(calisto_hybrid_modded, example_plain_env):
    """A rocketpy.Flight object of the Calisto rocket modded for a hybrid
    motor. The environment is the simplest possible, with no parameters set.

    Parameters
    ----------
    calisto_hybrid_modded : rocketpy.Rocket
        An object of the Rocket class. This is a pytest fixture too.
    example_plain_env : rocketpy.Environment
        An object of the Environment class. This is a pytest fixture too.

    Returns
    -------
    rocketpy.Flight
        A rocketpy.Flight object.
    """
    return Flight(
        rocket=calisto_hybrid_modded,
        environment=example_plain_env,
        rail_length=5.2,
        inclination=85,
        heading=0,
        time_overshoot=False,
        terminate_on_apogee=True,
    )


@pytest.fixture
def flight_calisto_custom_wind(calisto_robust, example_spaceport_env):
    """A rocketpy.Flight object of the Calisto rocket. This uses the calisto
    with the aerodynamic surfaces and parachutes. The environment is a bit more
    complex than the one in the flight_calisto_robust fixture. Now the wind is
    set to 5m/s (x direction) and 2m/s (y direction), constant with altitude.

    Parameters
    ----------
    calisto_robust : rocketpy.Rocket
        An object of the Rocket class. This is a pytest fixture too.
    example_spaceport_env : rocketpy.Environment
        An object of the Environment class. This is a pytest fixture too.

    Returns
    -------
    rocketpy.Flight

    """
    env = example_spaceport_env
    env.set_atmospheric_model(
        type="custom_atmosphere",
        temperature=300,
        wind_u=[(0, 5), (4000, 5)],
        wind_v=[(0, 2), (4000, 2)],
    )
    return Flight(
        environment=env,
        rocket=calisto_robust,
        rail_length=5.2,
        inclination=85,
        heading=0,
        terminate_on_apogee=False,
    )


@pytest.fixture
def flight_calisto_air_brakes(calisto_air_brakes_clamp_on, example_plain_env):
    """A rocketpy.Flight object of the Calisto rocket. This uses the calisto
    with the aerodynamic surfaces and air brakes. The environment is the
    simplest possible, with no parameters set. The air brakes are set to clamp
    the deployment level.

    Parameters
    ----------
    calisto_air_brakes_clamp_on : rocketpy.Rocket
        An object of the Rocket class.
    example_plain_env : rocketpy.Environment
        An object of the Environment class.

    Returns
    -------
    rocketpy.Flight
        A rocketpy.Flight object of the Calisto rocket in a more complex
        condition.
    """
    return Flight(
        rocket=calisto_air_brakes_clamp_on,
        environment=example_plain_env,
        rail_length=5.2,
        inclination=85,
        heading=0,
        time_overshoot=False,
        terminate_on_apogee=True,
    )


@pytest.fixture
def flight_calisto_with_sensors(calisto_with_sensors, example_plain_env):
    """A rocketpy.Flight object of the Calisto rocket. This uses the calisto
    with a set of ideal sensors. The environment is the simplest possible, with
    no parameters set.

    Parameters
    ----------
    calisto_with_sensors : rocketpy.Rocket
        An object of the Rocket class.
    example_plain_env : rocketpy.Environment
        An object of the Environment class.

    Returns
    -------
    rocketpy.Flight
        A rocketpy.Flight object of the Calisto rocket in a more complex
        condition.
    """
    return Flight(
        rocket=calisto_with_sensors,
        environment=example_plain_env,
        rail_length=5.2,
        inclination=85,
        heading=0,
        time_overshoot=False,
        terminate_on_apogee=True,
    )


@pytest.fixture
def flight_alpha(example_plain_env, cesaroni_m1670):
    """Fixture that returns a Flight using an alpha-dependent 3D Cd function."""
    # Create grid data
    mach = np.array([0.0, 0.5, 1.0, 1.5])
    reynolds = np.array([1e5, 1e6])
    alpha = np.array([0.0, 5.0, 10.0, 15.0])
    M, _, A = np.meshgrid(mach, reynolds, alpha, indexing="ij")
    cd_data = 0.3 + 0.05 * M + 0.03 * A
    cd_data = np.clip(cd_data, 0.2, 2.0)

    drag_func = Function.from_grid(
        cd_data,
        [mach, reynolds, alpha],
        inputs=["Mach", "Reynolds", "Alpha"],
        outputs="Cd",
    )

    env = example_plain_env
    env.set_atmospheric_model(type="standard_atmosphere")

    # Build rocket and flight
    rocket = Rocket(
        radius=0.0635,
        mass=16.24,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag=drag_func,
        power_on_drag=drag_func,
        center_of_mass_without_motor=0,
        coordinate_system_orientation="tail_to_nose",
    )
    rocket.set_rail_buttons(0.2, -0.5, 30)
    rocket.add_motor(cesaroni_m1670, position=-1.255)

    return Flight(
        rocket=rocket,
        environment=env,
        rail_length=5.2,
        inclination=85,
        heading=0,
    )


@pytest.fixture
def flight_flat(example_plain_env, cesaroni_m1670):
    """Fixture that returns a Flight using an alpha-averaged (flat) Cd function."""
    # Create grid data
    mach = np.array([0.0, 0.5, 1.0, 1.5])
    reynolds = np.array([1e5, 1e6])
    alpha = np.array([0.0, 5.0, 10.0, 15.0])
    M, _, A = np.meshgrid(mach, reynolds, alpha, indexing="ij")
    cd_data = 0.3 + 0.05 * M + 0.03 * A
    cd_data = np.clip(cd_data, 0.2, 2.0)

    cd_flat = cd_data.mean(axis=2)
    drag_flat = Function.from_grid(
        cd_flat,
        [mach, reynolds],
        inputs=["Mach", "Reynolds"],
        outputs="Cd",
    )

    env = example_plain_env
    env.set_atmospheric_model(type="standard_atmosphere")

    # Build rocket and flight
    rocket = Rocket(
        radius=0.0635,
        mass=16.24,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag=drag_flat,
        power_on_drag=drag_flat,
        center_of_mass_without_motor=0,
        coordinate_system_orientation="tail_to_nose",
    )
    rocket.set_rail_buttons(0.2, -0.5, 30)
    rocket.add_motor(cesaroni_m1670, position=-1.255)

    return Flight(
        rocket=rocket,
        environment=env,
        rail_length=5.2,
        inclination=85,
        heading=0,
    )
