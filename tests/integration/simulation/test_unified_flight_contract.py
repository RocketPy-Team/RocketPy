"""Integration coverage for the unified launch/orbital Flight contract."""

from datetime import UTC, datetime
from inspect import signature

import numpy as np
import pytest

from rocketpy import (
    Atmosphere,
    Earth,
    Environment,
    Epoch,
    Flight,
    FlightState,
    GenericMotor,
    Mission,
    Parachute,
    PointMassRocket,
    ReferenceFrame,
    Rocket,
    SimpleDatum,
    Space,
    Spacecraft,
    Vehicle,
    ZeroAtmosphereLayer,
)


def _earth():
    return Earth(geopotential="point_mass", atmosphere=ZeroAtmosphereLayer())


def _low_orbit_state(earth, altitude=500e3):
    epoch = Epoch.from_datetime(datetime(2026, 1, 1, tzinfo=UTC))
    radius = earth.datum.semi_major_axis + altitude
    speed = np.sqrt(earth.datum.gravitational_parameter / radius)
    return FlightState.cartesian(
        epoch=epoch,
        position=(radius, 0, 0),
        velocity=(0, speed, 0),
        frame="gcrf",
    )


def test_vehicle_hierarchy_and_flight_modes_are_independent():
    assert issubclass(Rocket, Vehicle)
    assert issubclass(Spacecraft, Vehicle)

    rocket = PointMassRocket(0.05, 2, 0, 0.5, 0.5, inertia=(1, 1, 1))
    flight = Flight.from_launch(
        rocket,
        Environment(),
        1,
        simulation_mode="6 DOF",
        max_time=0.05,
        max_time_step=0.01,
    )
    assert flight.simulation_mode == "6DOF"

    parameters = signature(Flight).parameters
    assert {
        "reference_frame",
        "initial_state",
        "force_models",
        "additional_forces",
        "dynamics",
    }.isdisjoint(parameters)
    assert "space" in parameters
    assert "forces" in parameters


def test_earth_launch_uses_one_inertial_6dof_flight_from_the_rail(calisto):
    """A normal Rocket can leave a rotating-Earth rail without changing solvers."""

    def zero_guidance(epoch, state, vehicle, earth):
        del epoch, state, vehicle, earth
        return np.zeros(3)

    launch_environment = Environment(
        date=datetime(2026, 1, 1, tzinfo=UTC),
        latitude=32.990254,
        longitude=-106.974998,
        elevation=1400,
    )
    earth = Earth(
        geopotential="point_mass",
        atmosphere=Atmosphere(
            {-np.inf: launch_environment},
            automatic_fall_environment=True,
        ),
        datum=SimpleDatum(),
    )

    flight = Flight.from_launch(
        calisto,
        earth,
        rail_length=5.2,
        inclination=85,
        heading=0,
        max_time=8,
        max_time_step=0.05,
        simulation_mode="6DOF",
        ode_solver="DOP853",
        forces=[zero_guidance],
    )

    assert flight.reference_frame is ReferenceFrame.GCRF
    assert flight.out_of_rail_time > 0
    assert flight.out_of_rail_time < flight.t_final
    assert flight.altitude(flight.t_final) > 0
    assert flight.position_local()[0] == pytest.approx(np.zeros(3), abs=1e-8)
    assert flight.velocity_local()[0] == pytest.approx(np.zeros(3), abs=1e-8)
    assert flight.speed(0) == pytest.approx(0, abs=1e-8)
    assert flight.attitude_local()[0, :, 2] == pytest.approx(
        flight._launch_fixed_to_enu @ flight._launch_direction_fixed,
        abs=1e-10,
    )
    assert flight.plots.is_high_altitude_flight is False
    assert flight.plots.low_altitude_end_time == pytest.approx(flight.t_final)
    assert flight.info() is None
    assert flight.initial_derivative is flight.udot_rail_earth_centered
    assert flight.u_dot_generalized is flight.u_dot_earth_centered
    assert flight.forces == [zero_guidance]
    fall_event = next(
        event for event in flight.events if event.name == "Automatic Fall Environment"
    )
    assert fall_event.triggered_times == []


def test_earth_centered_point_mass_uses_passive_gravity_turn():
    """GCRF 3-DOF evolves the thrust axis without adding a Flight force."""

    def vehicle(weathercock_coeff):
        motor = GenericMotor(
            thrust_source=20_000,
            burn_time=20,
            chamber_radius=0.2,
            chamber_height=2,
            chamber_position=-1,
            propellant_initial_mass=500,
            nozzle_radius=0.1,
            dry_mass=50,
            center_of_dry_mass_position=-1,
            dry_inertia=(20, 20, 2),
            nozzle_position=-2,
        )
        rocket = PointMassRocket(
            radius=0.3,
            mass=150,
            center_of_mass_without_motor=0,
            power_off_drag=0.15,
            power_on_drag=0.15,
            weathercock_coeff=weathercock_coeff,
            inertia=(30, 30, 3),
        )
        rocket.add_motor(motor, position=-1)
        return rocket

    launch_environment = Environment(
        date=datetime(2026, 1, 1, tzinfo=UTC),
        latitude=0,
        longitude=0,
    )
    earth = Earth(
        datum=SimpleDatum(),
        geopotential="point_mass",
        atmosphere=Atmosphere({-np.inf: launch_environment}),
    )
    fixed_axis = Flight.from_launch(
        vehicle(0),
        earth,
        rail_length=5,
        inclination=80,
        heading=90,
        max_time=8,
        max_time_step=0.1,
        simulation_mode="3DOF",
        ode_solver="DOP853",
    )
    passive_turn = Flight.from_launch(
        vehicle(1),
        earth,
        rail_length=5,
        inclination=80,
        heading=90,
        max_time=8,
        max_time_step=0.1,
        simulation_mode="3DOF",
        ode_solver="DOP853",
    )

    fixed_rotation = np.linalg.norm(
        fixed_axis.solution_array[-1, 7:11] - fixed_axis.solution_array[0, 7:11]
    )
    passive_rotation = np.linalg.norm(
        passive_turn.solution_array[-1, 7:11] - passive_turn.solution_array[0, 7:11]
    )
    assert passive_turn.reference_frame is ReferenceFrame.GCRF
    assert passive_turn.forces == []
    assert passive_rotation > 10 * fixed_rotation


def test_motor_thrust_vector_preserves_scalar_thrust_compatibility():
    motor = GenericMotor(
        thrust_source=1_000,
        burn_time=10,
        chamber_radius=0.1,
        chamber_height=1,
        chamber_position=0,
        propellant_initial_mass=5,
        nozzle_radius=0.05,
        dry_mass=1,
        center_of_dry_mass_position=0,
        dry_inertia=(1, 1, 1),
        nozzle_position=0,
    )

    assert motor.thrust(1) == pytest.approx(1_000)
    assert motor.thrust_vector(1) == pytest.approx((0, 0, 1_000))

    motor.set_thrust_direction((0, 1, 1))
    assert np.linalg.norm(motor.thrust_vector(1)) == pytest.approx(motor.thrust(1))
    assert motor.thrust_vector(1)[1:] == pytest.approx(
        (1_000 / np.sqrt(2), 1_000 / np.sqrt(2))
    )
    assert motor.net_thrust_vector(11) == pytest.approx(np.zeros(3))


def test_composed_ownership_and_mission_concatenation():
    earth = _earth()
    space = Space()
    spacecraft = Spacecraft(100, area=1, inertia=(10, 10, 10))
    first = Flight.from_orbit(
        spacecraft,
        earth,
        _low_orbit_state(earth),
        space=space,
        duration=2,
        max_time_step=1,
        simulation_mode="6DOF",
        ode_solver="DOP853",
    )
    assert first.forces == []

    mission = Mission(first)
    second = mission.concatenate(
        Spacecraft(20),
        duration=1,
        max_time_step=0.5,
        simulation_mode="3 DOF",
        ode_solver="DOP853",
    )
    assert second.state_at_time(0).elapsed_time == pytest.approx(2)
    assert mission.time[-1] == pytest.approx(3)


def test_automatic_fall_environment_is_created_at_reentry_coordinates():
    epoch = Epoch.from_datetime(datetime(2026, 1, 1, tzinfo=UTC))
    source_environment = Environment(
        date=epoch.to_datetime(),
        latitude=0,
        longitude=0,
    )
    atmosphere = Atmosphere(
        {-np.inf: source_environment, 80_000: ZeroAtmosphereLayer()},
        automatic_fall_environment=True,
    )
    earth = Earth(geopotential="point_mass", atmosphere=atmosphere)
    state = FlightState.geodetic(
        epoch=epoch,
        latitude=10,
        longitude=20,
        altitude=21_000,
        velocity_enu=(0, 0, -500),
        datum=earth.datum,
    )
    vehicle = Spacecraft(10, inertia=(2, 3, 4))
    vehicle.parachutes.append(
        Parachute("reentry", cd_s=5, trigger=19_000.0, sampling_rate=20)
    )
    source_identity = id(earth)
    flight = Flight.from_state(
        vehicle,
        earth,
        state,
        duration=8,
        max_time_step=0.1,
        simulation_mode="6DOF",
        ode_solver="DOP853",
    )
    event = next(
        event for event in flight.events if event.name == "Automatic Fall Environment"
    )
    assert len(event.triggered_times) == 1
    assert event.callback_log[0]["altitude"] == pytest.approx(20_000, abs=1e-4)
    assert flight.env.latitude == pytest.approx(10, abs=1e-3)
    assert flight.env.longitude == pytest.approx(20, abs=1e-3)
    assert id(flight.env) == source_identity
    assert earth.local_environment is event.callback_log[0]["regional_environment"]
    assert type(earth.local_environment) is type(source_environment)
    assert (
        earth.local_environment.atmospheric_model_type
        == source_environment.atmospheric_model_type
    )
    assert flight.simulation_mode == "6DOF"
    assert len(flight.parachute_events) == 1
    assert flight.parachute_events[0][0] > event.triggered_times[0]
    continuation = Mission(flight).concatenate(
        Spacecraft(5),
        at=event,
        duration=0.1,
        max_time_step=0.1,
        simulation_mode="3DOF",
        ode_solver="DOP853",
    )
    assert all(
        candidate.name != "Automatic Fall Environment"
        for candidate in continuation.events
    )


def test_automatic_fall_environment_is_ignored_without_environment_layer():
    earth = Earth(
        geopotential="point_mass",
        atmosphere=Atmosphere(
            {0.0: ZeroAtmosphereLayer()},
            automatic_fall_environment=True,
        ),
    )
    flight = Flight.from_orbit(
        Spacecraft(10),
        earth,
        _low_orbit_state(earth),
        duration=0.1,
        max_time_step=0.1,
        simulation_mode="3DOF",
        ode_solver="DOP853",
    )

    assert all(event.name != "Automatic Fall Environment" for event in flight.events)


def test_somigliana_component_rejects_rotating_frame():
    from rocketpy import DefaultGravity

    environment = Environment()
    gravity = DefaultGravity(environment.gravity)
    with pytest.raises(ValueError, match="flat-Earth"):
        gravity.local.acceleration(
            environment.epoch,
            np.ones(3),
            frame=ReferenceFrame.GCRF,
        )
