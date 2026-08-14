import pytest

from rocketpy import Flight
from rocketpy.mathutils.vector_matrix import Matrix, Vector
from rocketpy.motors.point_mass_motor import PointMassMotor
from rocketpy.rocket.multistage import MultiStageRocket, Stage
from rocketpy.rocket.rocket import Rocket
from rocketpy.simulation.mission import Mission


def _two_stage_vehicle(
    booster_separation=0.5,
    sustainer_ignition_delay=0.0,
    booster_separation_delta_v=0.0,
):
    """Same lightweight two-stage rig as test_multistage.py's
    _two_stage_vehicle, plus separation/ignition timing so Mission can
    orchestrate a full two-stage flight.
    """
    booster_rocket = Rocket(
        radius=0.1,
        mass=10.0,
        inertia=(1.0, 1.0, 0.01),
        power_off_drag=0.5,
        power_on_drag=0.6,
        center_of_mass_without_motor=0.0,
    )
    booster_rocket.add_motor(
        PointMassMotor(
            thrust_source=400, dry_mass=1.0, propellant_initial_mass=2.0,
            burn_time=1.0,
        ),
        position=0.0,
    )
    booster = Stage(
        name="booster",
        rocket=booster_rocket,
        separation=booster_separation,
        separation_delta_v=booster_separation_delta_v,
    )

    sustainer_rocket = Rocket(
        radius=0.08,
        mass=5.0,
        inertia=(0.5, 0.5, 0.005),
        power_off_drag=0.3,
        power_on_drag=0.4,
        center_of_mass_without_motor=2.0,
    )
    sustainer_rocket.add_motor(
        PointMassMotor(
            thrust_source=200, dry_mass=0.5, propellant_initial_mass=1.0,
            burn_time=1.0,
        ),
        position=2.0,
    )
    sustainer = Stage(
        name="sustainer", rocket=sustainer_rocket,
        ignition_delay=sustainer_ignition_delay,
    )

    return booster, sustainer


def test_mission_degenerates_to_a_single_flight_for_a_plain_rocket(
    calisto, example_plain_env
):
    mission = Mission(
        vehicle=calisto,
        environment=example_plain_env,
        rail_length=5.2,
        inclination=85,
        heading=0,
    )

    assert list(mission.flights.keys()) == ["stage_1"]
    assert len(mission.flights["stage_1"]) == 1
    assert isinstance(mission.flights["stage_1"][0], Flight)


def test_mission_timeline_has_ignition_liftoff_and_impact_only(
    calisto, example_plain_env
):
    mission = Mission(
        vehicle=calisto,
        environment=example_plain_env,
        rail_length=5.2,
        inclination=85,
        heading=0,
    )

    event_names = [name for _, name in mission.timeline]

    assert "liftoff" in event_names
    assert any(name.startswith("ignition:") for name in event_names)
    assert any(name.startswith("impact:") for name in event_names)
    assert not any("separation" in name or "ejection" in name for name in event_names)


def test_mission_uses_the_stage_name_as_the_flights_key(calisto, example_plain_env):
    stage = Stage(name="first_stage", rocket=calisto)
    vehicle = MultiStageRocket(stages=[stage])

    mission = Mission(
        vehicle=vehicle,
        environment=example_plain_env,
        rail_length=5.2,
        inclination=85,
        heading=0,
    )

    assert list(mission.flights.keys()) == ["first_stage"]


def test_two_stage_mission_separates_at_burnout_plus_delay(example_plain_env):
    booster, sustainer = _two_stage_vehicle(booster_separation=0.5)
    vehicle = MultiStageRocket(stages=[booster, sustainer])

    mission = Mission(
        vehicle=vehicle,
        environment=example_plain_env,
        rail_length=1.0,
        inclination=90,
        heading=0,
        max_time=5,
    )

    assert "booster" in mission.flights
    assert "sustainer" in mission.flights
    # The full stack's flight appears under every body aboard it.
    assert mission.flights["sustainer"][0] is mission.flights["booster"][0]
    assert len(mission.flights["booster"]) == 2  # stack, then booster alone
    assert len(mission.flights["sustainer"]) == 2  # stack, then sustainer alone

    event_times = dict((name, t) for t, name in mission.timeline)
    assert event_times["separation:booster"] == pytest.approx(1.5)
    assert event_times["ignition:sustainer"] == pytest.approx(1.5)


def test_two_stage_mission_handoff_matches_hand_computed_kinematics(
    example_plain_env,
):
    booster, sustainer = _two_stage_vehicle(
        booster_separation=0.5, booster_separation_delta_v=0.0
    )
    vehicle = MultiStageRocket(stages=[booster, sustainer])

    mission = Mission(
        vehicle=vehicle,
        environment=example_plain_env,
        rail_length=1.0,
        inclination=90,
        heading=0,
        max_time=5,
    )

    stack_flight = mission.flights["booster"][0]
    sustainer_flight = mission.flights["sustainer"][-1]
    ending_t, x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3 = (
        stack_flight.solution[-1]
    )

    # Independent re-derivation of the handoff formula from
    # mission_multistage_design.md's _handoff_state, using the actual
    # ending state and the actual Rocket objects Mission built - not a
    # call into Mission's own code path.
    stack_rocket = vehicle.flight_rocket(active_stages=(booster, sustainer))
    offset = (
        sustainer_flight.rocket.center_of_dry_mass_position
        - stack_rocket.center_of_dry_mass_position
    )
    rotation = Matrix.transformation((e0, e1, e2, e3))
    d = Vector([0, 0, offset])
    omega = Vector([w1, w2, w3])

    expected_position = Vector([x, y, z]) + rotation @ d
    expected_velocity = Vector([vx, vy, vz]) + rotation @ omega.cross(d)

    handoff_state = sustainer_flight.solution[0]
    assert handoff_state[0] == pytest.approx(ending_t)
    assert handoff_state[1:4] == pytest.approx(list(expected_position))
    assert handoff_state[4:7] == pytest.approx(list(expected_velocity))
    assert handoff_state[7:11] == pytest.approx([e0, e1, e2, e3])
    assert handoff_state[11:14] == pytest.approx([w1, w2, w3])


def test_two_stage_mission_splits_separation_delta_v_by_momentum_conservation(
    example_plain_env,
):
    delta_v = 2.0
    booster, sustainer = _two_stage_vehicle(
        booster_separation=0.5, booster_separation_delta_v=delta_v
    )
    vehicle = MultiStageRocket(stages=[booster, sustainer])

    mission = Mission(
        vehicle=vehicle,
        environment=example_plain_env,
        rail_length=1.0,
        inclination=90,
        heading=0,
        max_time=5,
    )

    # Hand-computed momentum-conserving split, independent of Mission's
    # own code path: booster is spent (dry_mass) and sustainer is full
    # (total_mass at its own t=0, not yet ignited) at the separation
    # instant.
    booster_mass_after = booster.rocket.dry_mass
    sustainer_mass_after = sustainer.rocket.total_mass(0)
    total_mass_after = booster_mass_after + sustainer_mass_after
    expected_booster_delta_v = -(sustainer_mass_after / total_mass_after) * delta_v
    expected_sustainer_delta_v = (booster_mass_after / total_mass_after) * delta_v

    stack_flight = mission.flights["booster"][0]
    ending_vz = stack_flight.solution[-1][6]

    booster_flight = mission.flights["booster"][-1]
    sustainer_flight = mission.flights["sustainer"][-1]

    # inclination=90, heading=0 -> identity rotation, so the body-frame
    # axial delta_v maps directly onto the inertial vz component.
    assert booster_flight.solution[0][6] == pytest.approx(
        ending_vz + expected_booster_delta_v
    )
    assert sustainer_flight.solution[0][6] == pytest.approx(
        ending_vz + expected_sustainer_delta_v
    )
