import math

import matplotlib.pyplot as plt
import pytest

from rocketpy import Flight
from rocketpy.mathutils.vector_matrix import Matrix, Vector
from rocketpy.motors.point_mass_motor import PointMassMotor
from rocketpy.plots.compare.compare_flights import CompareFlights
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


def test_mission_timeline_has_no_separation_or_ejection_for_a_plain_rocket(
    calisto, example_plain_env
):
    # The degenerate single-stage, no-deployables case: every event type
    # Mission can ever record (ignition, liftoff, rail_departure,
    # burnout, apogee, impact) EXCEPT separation/ejection, which only
    # apply to multi-stage vehicles and deployables respectively.
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
    assert any(name.startswith("burnout:") for name in event_names)
    assert any(name.startswith("apogee:") for name in event_names)
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


def test_mission_timeline_includes_burnout_for_every_stage(example_plain_env):
    # Useful on its own (a natural "what happened when" event, same
    # status as ignition/separation/impact) and needed to plot it -
    # burnout isn't derivable from the other timeline entries alone.
    booster, sustainer = _two_stage_vehicle(booster_separation=0.5)
    vehicle = MultiStageRocket(stages=[booster, sustainer])

    mission = Mission(
        vehicle=vehicle, environment=example_plain_env, rail_length=1.0,
        inclination=90, heading=0, max_time=5,
    )

    event_times = dict((name, t) for t, name in mission.timeline)
    # Hand-computed: booster is the root stage (never shifted), so its
    # own burn_out_time is already absolute. sustainer ignites at
    # separation (1.5s, independently verified by the test above) and
    # its own local burn_time=1.0s is shifted by that same amount.
    assert event_times["burnout:booster"] == pytest.approx(
        booster.rocket.motor.burn_out_time
    )
    assert event_times["burnout:sustainer"] == pytest.approx(1.5 + 1.0)


def test_mission_timeline_records_each_stages_burnout_exactly_once(
    example_plain_env,
):
    # A stage riding through an ejection (deployable leaving, stage
    # unchanged) re-enters _walk with the SAME bottom stage - burnout
    # must not be recorded a second time for it.
    vehicle, _stage, _deployable = _single_stage_with_deployable_vehicle()

    mission = Mission(
        vehicle=vehicle, environment=example_plain_env, rail_length=1.0,
        inclination=90, heading=0, max_time=20,
    )

    burnout_entries = [
        (t, name) for t, name in mission.timeline if name == "burnout:carrier"
    ]
    assert len(burnout_entries) == 1


def test_mission_timeline_includes_apogee_for_every_flight_that_reaches_one(
    example_plain_env,
):
    # Not literally every flight - see
    # test_mission_timeline_omits_apogee_for_a_flight_truncated_before_reaching_it
    # for the case of a flight that ends (via separation) before ever
    # reaching its own apogee, which must NOT get an entry.
    booster, sustainer = _two_stage_vehicle(booster_separation=0.5)
    vehicle = MultiStageRocket(stages=[booster, sustainer])

    mission = Mission(
        vehicle=vehicle, environment=example_plain_env, rail_length=1.0,
        inclination=90, heading=0, max_time=5,
    )

    event_times = dict((name, t) for t, name in mission.timeline)
    for flight in mission.all_flights:
        start_time = flight.solution[0][0]
        end_time = flight.solution[-1][0]
        if start_time < flight.apogee_time < end_time:
            assert event_times[f"apogee:{flight.name}"] == pytest.approx(
                flight.apogee_time
            )
        else:
            assert f"apogee:{flight.name}" not in event_times


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


def _single_stage_with_deployable_vehicle(deployable_delta_v=0.0):
    """A carrier stage with a payload that ejects at apogee, via a fully
    built free_rocket (no add_surface() aerodynamics).
    """
    stage_rocket = Rocket(
        radius=0.1,
        mass=10.0,
        inertia=(1.0, 1.0, 0.01),
        power_off_drag=0.5,
        power_on_drag=0.6,
        center_of_mass_without_motor=0.0,
    )
    stage_rocket.add_motor(
        PointMassMotor(
            thrust_source=400, dry_mass=1.0, propellant_initial_mass=2.0,
            burn_time=1.0,
        ),
        position=0.0,
    )
    stage = Stage(name="carrier", rocket=stage_rocket)

    payload_rocket = Rocket(
        radius=0.02,
        mass=1.0,
        inertia=(0.001, 0.001, 0.0001),
        power_off_drag=0.5,
        power_on_drag=0.5,
        center_of_mass_without_motor=0.0,
    )

    vehicle = MultiStageRocket(stages=[stage])
    deployable = vehicle.add_deployable(
        name="payload",
        mass=1.0,
        inertia=(0.001, 0.001, 0.0001),
        position=1.0,
        free_rocket=payload_rocket,
        ejection="apogee",
        separation_delta_v=deployable_delta_v,
    )

    return vehicle, stage, deployable


def test_deployable_ejects_at_apogee(example_plain_env):
    vehicle, _, _ = _single_stage_with_deployable_vehicle()

    mission = Mission(
        vehicle=vehicle,
        environment=example_plain_env,
        rail_length=1.0,
        inclination=90,
        heading=0,
        max_time=20,
    )

    assert "carrier" in mission.flights
    assert "payload" in mission.flights
    # The carrier's flight while the payload is aboard appears under both.
    assert mission.flights["payload"][0] is mission.flights["carrier"][0]
    assert len(mission.flights["carrier"]) == 2  # carrier+payload, then carrier alone
    assert len(mission.flights["payload"]) == 2  # carrier+payload, then payload alone

    carrier_flight = mission.flights["carrier"][0]
    event_times = dict((name, t) for t, name in mission.timeline)
    assert event_times["ejection:payload"] == pytest.approx(carrier_flight.apogee_time)
    assert not any("separation" in name for _, name in mission.timeline)


def test_deployable_handoff_matches_hand_computed_kinematics(example_plain_env):
    vehicle, stage, deployable = _single_stage_with_deployable_vehicle()

    mission = Mission(
        vehicle=vehicle,
        environment=example_plain_env,
        rail_length=1.0,
        inclination=90,
        heading=0,
        max_time=20,
    )

    carrier_flight = mission.flights["carrier"][0]
    payload_flight = mission.flights["payload"][-1]
    ending_t, x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3 = (
        carrier_flight.solution[-1]
    )

    # Independent re-derivation of the handoff formula, using the actual
    # apogee state and the actual Rocket objects Mission built - not a
    # call into Mission's own code path.
    carrier_rocket = vehicle.flight_rocket(
        active_stages=(stage,), carried_deployables=(deployable,)
    )
    offset = (
        payload_flight.rocket.center_of_dry_mass_position
        - carrier_rocket.center_of_dry_mass_position
    )
    rotation = Matrix.transformation((e0, e1, e2, e3))
    d = Vector([0, 0, offset])
    omega = Vector([w1, w2, w3])

    expected_position = Vector([x, y, z]) + rotation @ d
    expected_velocity = Vector([vx, vy, vz]) + rotation @ omega.cross(d)

    handoff_state = payload_flight.solution[0]
    assert handoff_state[0] == pytest.approx(ending_t)
    assert handoff_state[1:4] == pytest.approx(list(expected_position))
    assert handoff_state[4:7] == pytest.approx(list(expected_velocity))


def test_deployable_handoff_matches_hand_computed_kinematics_off_vertical(
    example_plain_env,
):
    # Same gap as the two-stage off-vertical test above: this handoff was
    # only ever exercised at inclination=90 (Identity rotation).
    vehicle, stage, deployable = _single_stage_with_deployable_vehicle()

    mission = Mission(
        vehicle=vehicle,
        environment=example_plain_env,
        rail_length=1.0,
        inclination=84,
        heading=30,
        max_time=20,
    )

    carrier_flight = mission.flights["carrier"][0]
    payload_flight = mission.flights["payload"][-1]
    ending_t, x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3 = (
        carrier_flight.solution[-1]
    )

    rotation = Matrix.transformation((e0, e1, e2, e3))
    assert rotation != Matrix.identity()
    assert x != pytest.approx(0.0)

    carrier_rocket = vehicle.flight_rocket(
        active_stages=(stage,), carried_deployables=(deployable,)
    )
    offset = (
        payload_flight.rocket.center_of_dry_mass_position
        - carrier_rocket.center_of_dry_mass_position
    )
    d = Vector([0, 0, offset])
    omega = Vector([w1, w2, w3])

    expected_position = Vector([x, y, z]) + rotation @ d
    expected_velocity = Vector([vx, vy, vz]) + rotation @ omega.cross(d)

    handoff_state = payload_flight.solution[0]
    assert handoff_state[0] == pytest.approx(ending_t)
    assert handoff_state[1:4] == pytest.approx(list(expected_position))
    assert handoff_state[4:7] == pytest.approx(list(expected_velocity))


def test_deployable_splits_separation_delta_v_by_momentum_conservation(
    example_plain_env,
):
    delta_v = 3.0
    vehicle, stage, deployable = _single_stage_with_deployable_vehicle(
        deployable_delta_v=delta_v
    )

    mission = Mission(
        vehicle=vehicle,
        environment=example_plain_env,
        rail_length=1.0,
        inclination=90,
        heading=0,
        max_time=20,
    )

    carrier_flight = mission.flights["carrier"][0]
    ending_vz = carrier_flight.solution[-1][6]

    # Hand-computed momentum-conserving split, independent of Mission's
    # own code path.
    stage_mass_after = stage.rocket.total_mass(carrier_flight.apogee_time)
    payload_mass_after = deployable.free_rocket.total_mass(0)
    total_mass_after = stage_mass_after + payload_mass_after
    expected_stage_delta_v = -(payload_mass_after / total_mass_after) * delta_v
    expected_payload_delta_v = (stage_mass_after / total_mass_after) * delta_v

    stage_flight = mission.flights["carrier"][-1]
    payload_flight = mission.flights["payload"][-1]

    # inclination=90, heading=0 -> identity rotation, so the body-frame
    # axial delta_v maps directly onto the inertial vz component.
    assert stage_flight.solution[0][6] == pytest.approx(
        ending_vz + expected_stage_delta_v
    )
    assert payload_flight.solution[0][6] == pytest.approx(
        ending_vz + expected_payload_delta_v
    )


def test_all_flights_is_the_single_flight_for_the_degenerate_case(
    calisto, example_plain_env
):
    mission = Mission(
        vehicle=calisto,
        environment=example_plain_env,
        rail_length=5.2,
        inclination=85,
        heading=0,
    )

    assert mission.all_flights == [mission.flights["stage_1"][0]]


def test_all_flights_has_no_duplicate_for_the_shared_stack_flight(example_plain_env):
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

    assert len(mission.all_flights) == 3
    assert len(set(id(flight) for flight in mission.all_flights)) == 3
    assert mission.all_flights == [
        mission.flights["booster"][0],
        mission.flights["booster"][1],
        mission.flights["sustainer"][1],
    ]


def test_all_flights_feeds_compare_flights(example_plain_env):
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

    assert CompareFlights(mission.all_flights).trajectories_3d(filename=None) is None


def test_flight_names_distinguish_each_body_and_phase(example_plain_env):
    # CompareFlights labels each line in a plot legend using flight.name;
    # every Flight Mission creates must have a name that distinguishes it
    # (Flight's own default "Flight" would make every legend entry
    # identical and useless).
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

    names = [flight.name for flight in mission.all_flights]
    assert len(names) == len(set(names))
    assert all(name != "Flight" for name in names)


def _three_stage_vehicle():
    """Booster -> sustainer -> kick stage, each separating in turn (delay
    0.5s after its own burnout), same lightweight PointMassMotor rig as
    _two_stage_vehicle - extended by one more stage, to exercise the
    general N-stage walk beyond the degenerate 1- and 2-stage cases.
    """
    booster_rocket = Rocket(
        radius=0.1, mass=10.0, inertia=(1.0, 1.0, 0.01),
        power_off_drag=0.5, power_on_drag=0.6, center_of_mass_without_motor=0.0,
    )
    booster_rocket.add_motor(
        PointMassMotor(
            thrust_source=400, dry_mass=1.0, propellant_initial_mass=2.0,
            burn_time=1.0,
        ),
        position=0.0,
    )
    booster = Stage(name="booster", rocket=booster_rocket, separation=0.5)

    sustainer_rocket = Rocket(
        radius=0.08, mass=5.0, inertia=(0.5, 0.5, 0.005),
        power_off_drag=0.3, power_on_drag=0.4, center_of_mass_without_motor=2.0,
    )
    sustainer_rocket.add_motor(
        PointMassMotor(
            thrust_source=200, dry_mass=0.5, propellant_initial_mass=1.0,
            burn_time=1.0,
        ),
        position=2.0,
    )
    sustainer = Stage(name="sustainer", rocket=sustainer_rocket, separation=0.5)

    kick_rocket = Rocket(
        radius=0.04, mass=2.0, inertia=(0.1, 0.1, 0.001),
        power_off_drag=0.2, power_on_drag=0.3, center_of_mass_without_motor=3.0,
    )
    kick_rocket.add_motor(
        PointMassMotor(
            thrust_source=100, dry_mass=0.2, propellant_initial_mass=0.5,
            burn_time=1.0,
        ),
        position=3.0,
    )
    kick = Stage(name="kick", rocket=kick_rocket)

    return booster, sustainer, kick


def test_three_stage_mission_separates_in_order(example_plain_env):
    booster, sustainer, kick = _three_stage_vehicle()
    vehicle = MultiStageRocket(stages=[booster, sustainer, kick])

    mission = Mission(
        vehicle=vehicle,
        environment=example_plain_env,
        rail_length=1.0,
        inclination=90,
        heading=0,
        max_time=8,
    )

    assert "booster" in mission.flights
    assert "sustainer" in mission.flights
    assert "kick" in mission.flights

    event_times = dict((name, t) for t, name in mission.timeline)
    booster_separation = event_times["separation:booster"]
    sustainer_separation = event_times["separation:sustainer"]
    assert booster_separation == pytest.approx(1.5)
    # sustainer's own motor ignites at booster_separation (ignition_delay
    # defaults to 0), then burns for 1.0s and separates 0.5s after that.
    assert sustainer_separation == pytest.approx(booster_separation + 1.5)
    assert sustainer_separation > booster_separation

    # Full stack flight appears under all three; booster falls away alone
    # after the first separation; sustainer+kick continue together until
    # the second separation, then each finishes alone.
    assert mission.flights["sustainer"][0] is mission.flights["booster"][0]
    assert mission.flights["kick"][0] is mission.flights["booster"][0]
    assert mission.flights["sustainer"][1] is mission.flights["kick"][1]
    assert len(mission.flights["booster"]) == 2  # stack, then booster alone
    assert len(mission.flights["sustainer"]) == 3  # stack, stack-minus-booster, alone
    assert len(mission.flights["kick"]) == 3  # stack, stack-minus-booster, alone

    assert len(mission.all_flights) == 5
    assert len(set(id(flight) for flight in mission.all_flights)) == 5


def test_deployable_riding_sustainer_ejects_after_booster_separation(
    example_plain_env,
):
    # The case explicitly called out as unsupported before Gap 1: a
    # deployable riding the sustainer of a two-stage vehicle, still
    # aboard through the booster's own separation, only ejecting later
    # at the sustainer's own apogee.
    booster, sustainer = _two_stage_vehicle(booster_separation=0.5)
    vehicle = MultiStageRocket(stages=[booster, sustainer])
    vehicle.add_deployable(
        name="payload", mass=0.2, inertia=(0.001, 0.001, 0.0001), position=1.5,
        stage=sustainer, free_rocket=Rocket(
            radius=0.02, mass=0.2, inertia=(0.001, 0.001, 0.0001),
            power_off_drag=0.5, power_on_drag=0.5, center_of_mass_without_motor=0.0,
        ),
        ejection="apogee",
    )

    mission = Mission(
        vehicle=vehicle,
        environment=example_plain_env,
        rail_length=1.0,
        inclination=90,
        heading=0,
        max_time=20,
    )

    assert "booster" in mission.flights
    assert "sustainer" in mission.flights
    assert "payload" in mission.flights

    event_times = dict((name, t) for t, name in mission.timeline)
    booster_separation = event_times["separation:booster"]
    payload_ejection = event_times["ejection:payload"]
    assert payload_ejection > booster_separation
    assert not any(name.startswith("separation:") and name != "separation:booster"
                   for name in event_times)

    # Full stack (with payload aboard) appears under all three bodies;
    # booster falls away alone; sustainer+payload continue together
    # until apogee, then each finishes alone.
    assert mission.flights["sustainer"][0] is mission.flights["booster"][0]
    assert mission.flights["payload"][0] is mission.flights["booster"][0]
    assert mission.flights["sustainer"][1] is mission.flights["payload"][1]
    assert len(mission.flights["booster"]) == 2  # stack, then booster alone
    assert len(mission.flights["sustainer"]) == 3  # stack, stack-minus-booster, alone
    assert len(mission.flights["payload"]) == 3  # stack, stack-minus-booster, alone

    assert len(mission.all_flights) == 5
    assert len(set(id(flight) for flight in mission.all_flights)) == 5


def test_deployable_ejection_handles_handoff_already_past_apogee(example_plain_env):
    # A departing stage's separation_delta_v can leave it already
    # descending (vz <= 0) right at the handoff instant - e.g. a big
    # negative kick applied to a stage that's already near its own
    # apogee. Flight's own apogee root-finding assumes a flight starts
    # ascending; hand it an already-past-apogee initial_solution with
    # terminate_on_apogee=True and it crashes deep inside Flight
    # (IndexError from FlightPhases.add) - found via a 300-run
    # randomized sweep over varied Mission configurations (seed 60).
    booster, sustainer = _two_stage_vehicle(
        booster_separation=0.5, booster_separation_delta_v=1000.0,
    )
    vehicle = MultiStageRocket(stages=[booster, sustainer])
    vehicle.add_deployable(
        name="payload", mass=0.2, inertia=(0.001, 0.001, 0.0001), position=0.5,
        stage=booster, free_rocket=Rocket(
            radius=0.02, mass=0.2, inertia=(0.001, 0.001, 0.0001),
            power_off_drag=0.5, power_on_drag=0.5, center_of_mass_without_motor=0.0,
        ),
        ejection="apogee",
    )

    mission = Mission(
        vehicle=vehicle, environment=example_plain_env, rail_length=1.0,
        inclination=90, heading=0, max_time=10,
    )

    assert "payload" in mission.flights
    event_times = dict((name, t) for t, name in mission.timeline)
    assert "ejection:payload" in event_times


def test_two_stage_mission_handoff_matches_hand_computed_kinematics_off_vertical(
    example_plain_env,
):
    # Every other handoff test uses inclination=90, heading=0, which makes
    # the quaternion Identity and the rotation term in _handoff_state a
    # no-op - the assertions below don't assume that (they read the actual
    # quaternion/omega off the ending state and apply the general formula),
    # but nothing had exercised a genuinely non-Identity rotation. This
    # does: inclination=84, heading=30 gives real lateral (x, y) motion and
    # a non-trivial quaternion at separation.
    booster, sustainer = _two_stage_vehicle(
        booster_separation=0.5, booster_separation_delta_v=0.0
    )
    vehicle = MultiStageRocket(stages=[booster, sustainer])

    mission = Mission(
        vehicle=vehicle,
        environment=example_plain_env,
        rail_length=1.0,
        inclination=84,
        heading=30,
        max_time=5,
    )

    stack_flight = mission.flights["booster"][0]
    sustainer_flight = mission.flights["sustainer"][-1]
    ending_t, x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3 = (
        stack_flight.solution[-1]
    )

    # Sanity check this test actually exercises a non-Identity rotation -
    # otherwise it would silently degenerate into a duplicate of the
    # inclination=90 test above.
    rotation = Matrix.transformation((e0, e1, e2, e3))
    assert rotation != Matrix.identity()
    assert x != pytest.approx(0.0)

    # Independent re-derivation of the handoff formula, same as the
    # vertical-launch test above.
    stack_rocket = vehicle.flight_rocket(active_stages=(booster, sustainer))
    offset = (
        sustainer_flight.rocket.center_of_dry_mass_position
        - stack_rocket.center_of_dry_mass_position
    )
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


def _pointmass_stage(name, radius, mass, thrust, position=0.0, **stage_kwargs):
    rocket = Rocket(
        radius=radius, mass=mass, inertia=(0.5, 0.5, 0.01),
        power_off_drag=0.5, power_on_drag=0.5, center_of_mass_without_motor=0.0,
    )
    rocket.add_motor(
        PointMassMotor(
            thrust_source=thrust, dry_mass=0.3, propellant_initial_mass=0.5,
            burn_time=1.0,
        ),
        position=position,
    )
    return Stage(name=name, rocket=rocket, **stage_kwargs)


def test_two_stage_mission_with_hybrid_motor_bottom_stage(hybrid_motor, example_plain_env):
    # hybrid_motor: thrust 2000-100t, burn_time (0, 10), tank-based mass
    # (not a plain PointMassMotor) - the bottom (never-shifted) stage,
    # so this exercises flight_rocket()'s mass/inertia composition with
    # a real HybridMotor, not _shift_motor_ignition.
    booster_rocket = Rocket(
        radius=0.15, mass=6.0, inertia=(2.0, 2.0, 0.05),
        power_off_drag=0.5, power_on_drag=0.6, center_of_mass_without_motor=0.0,
    )
    booster_rocket.add_motor(hybrid_motor, position=0.0)
    booster = Stage(name="booster", rocket=booster_rocket, separation=2.0)

    sustainer = _pointmass_stage("sustainer", 0.08, 3.0, thrust=100)

    vehicle = MultiStageRocket(stages=[booster, sustainer])
    mission = Mission(
        vehicle=vehicle, environment=example_plain_env, rail_length=1.0,
        inclination=90, heading=0, max_time=30,
    )

    event_times = dict((name, t) for t, name in mission.timeline)
    assert event_times["separation:booster"] == pytest.approx(
        hybrid_motor.burn_out_time + 2.0
    )
    assert "impact:booster" in event_times
    assert "impact:sustainer" in event_times
    for flight in mission.all_flights:
        for state in flight.solution:
            assert all(math.isfinite(v) for v in state)


def test_two_stage_mission_with_liquid_motor_sustainer(liquid_motor, example_plain_env):
    # liquid_motor: burn_time (8, 20) - a burn window that does NOT
    # start at the motor's own local t=0, and tank-derived mass
    # Functions (not a plain analytic curve). As the sustainer (always
    # shifted via _shift_motor_ignition), this checks that shifting
    # correctly re-anchors both the non-zero-start burn window AND the
    # tank-based mass/inertia Functions, not just simple ones.
    booster = _pointmass_stage(
        "booster", 0.2, 10.0, thrust=3000, separation=1.5,
    )

    sustainer_rocket = Rocket(
        radius=0.15, mass=8.0, inertia=(3.0, 3.0, 0.05),
        power_off_drag=0.4, power_on_drag=0.5, center_of_mass_without_motor=0.0,
    )
    sustainer_rocket.add_motor(liquid_motor, position=0.0)
    sustainer = Stage(name="sustainer", rocket=sustainer_rocket)

    vehicle = MultiStageRocket(stages=[booster, sustainer])
    mission = Mission(
        vehicle=vehicle, environment=example_plain_env, rail_length=1.0,
        inclination=90, heading=0, max_time=60,
    )

    event_times = dict((name, t) for t, name in mission.timeline)
    separation_time = event_times["separation:booster"]
    ignition_time = event_times["ignition:sustainer"]
    assert ignition_time == pytest.approx(separation_time)

    sustainer_flight = mission.flights["sustainer"][-1]
    shifted_burn_out = ignition_time + liquid_motor.burn_out_time
    assert sustainer_flight.rocket.motor.burn_out_time == pytest.approx(shifted_burn_out)
    for state in sustainer_flight.solution:
        assert all(math.isfinite(v) for v in state)


def test_two_stage_mission_with_generic_motor_sustainer(generic_motor, example_plain_env):
    # generic_motor: burn_time (2, 7) - thrust is exactly zero for the
    # first 2s of the motor's own local clock. Mirrors the liquid-motor
    # test but for GenericMotor specifically (a different Motor
    # subclass with its own Function wiring).
    booster = _pointmass_stage("booster", 0.2, 8.0, thrust=1500, separation=1.0)

    sustainer_rocket = Rocket(
        radius=0.15, mass=6.0, inertia=(1.5, 1.5, 0.03),
        power_off_drag=0.4, power_on_drag=0.5, center_of_mass_without_motor=0.0,
    )
    sustainer_rocket.add_motor(generic_motor, position=0.0)
    sustainer = Stage(name="sustainer", rocket=sustainer_rocket)

    vehicle = MultiStageRocket(stages=[booster, sustainer])
    mission = Mission(
        vehicle=vehicle, environment=example_plain_env, rail_length=1.0,
        inclination=90, heading=0, max_time=40,
    )

    event_times = dict((name, t) for t, name in mission.timeline)
    ignition_time = event_times["ignition:sustainer"]
    sustainer_flight = mission.flights["sustainer"][-1]
    expected_burn_out = ignition_time + generic_motor.burn_out_time
    assert sustainer_flight.rocket.motor.burn_out_time == pytest.approx(expected_burn_out)
    for state in sustainer_flight.solution:
        assert all(math.isfinite(v) for v in state)


def test_mixed_coordinate_system_orientation_across_stages(example_plain_env):
    # Nothing requires every stage's own Rocket to share the same
    # coordinate_system_orientation - each stage's _csys is read from
    # its own rocket independently throughout multistage.py. Booster
    # keeps the default (tail_to_nose); sustainer is built nose_to_tail
    # instead, so its own positive-z direction points the opposite way.
    booster_rocket = Rocket(
        radius=0.1, mass=10.0, inertia=(1.0, 1.0, 0.01),
        power_off_drag=0.5, power_on_drag=0.6, center_of_mass_without_motor=0.0,
        coordinate_system_orientation="tail_to_nose",
    )
    booster_rocket.add_motor(
        PointMassMotor(thrust_source=400, dry_mass=1.0, propellant_initial_mass=2.0, burn_time=1.0),
        position=0.0,
    )
    booster = Stage(name="booster", rocket=booster_rocket, separation=0.5)

    sustainer_rocket = Rocket(
        radius=0.08, mass=5.0, inertia=(0.5, 0.5, 0.005),
        power_off_drag=0.3, power_on_drag=0.4, center_of_mass_without_motor=0.0,
        coordinate_system_orientation="nose_to_tail",
    )
    sustainer_rocket.add_motor(
        PointMassMotor(thrust_source=200, dry_mass=0.5, propellant_initial_mass=1.0, burn_time=1.0),
        position=0.0,
    )
    sustainer = Stage(name="sustainer", rocket=sustainer_rocket)

    vehicle = MultiStageRocket(stages=[booster, sustainer])
    mission = Mission(
        vehicle=vehicle, environment=example_plain_env, rail_length=1.0,
        inclination=90, heading=0, max_time=20,
    )

    assert "booster" in mission.flights
    assert "sustainer" in mission.flights
    for flight in mission.all_flights:
        for state in flight.solution:
            assert all(math.isfinite(v) for v in state)


def test_deployable_with_powered_free_rocket(example_plain_env):
    # A deployable's free_rocket is "a fully built Rocket or
    # PointMassRocket" per Deployable's own docstring - nothing
    # restricts it to being unpowered. A kick-stage payload that
    # ignites its own motor after ejecting is a real use case.
    carrier = _pointmass_stage("carrier", 0.1, 10.0, thrust=400)

    kick_rocket = Rocket(
        radius=0.03, mass=1.0, inertia=(0.01, 0.01, 0.001),
        power_off_drag=0.5, power_on_drag=0.5, center_of_mass_without_motor=0.0,
    )
    kick_rocket.add_motor(
        PointMassMotor(thrust_source=50, dry_mass=0.2, propellant_initial_mass=0.3, burn_time=1.0),
        position=0.0,
    )

    vehicle = MultiStageRocket(stages=[carrier])
    vehicle.add_deployable(
        name="kick", mass=1.5, inertia=(0.01, 0.01, 0.001), position=1.0,
        free_rocket=kick_rocket, ejection="apogee",
    )

    mission = Mission(
        vehicle=vehicle, environment=example_plain_env, rail_length=1.0,
        inclination=90, heading=0, max_time=30,
    )

    assert "kick" in mission.flights
    kick_flight = mission.flights["kick"][-1]
    # The kick stage's own motor must have fired at its own local t=0
    # relative to the handoff instant - i.e. burn_out_time measured
    # from the flight's own start should match the motor's own,
    # unshifted burn_out_time (a deployable's free_rocket motor is
    # never re-anchored by _shift_motor_ignition - it only applies to
    # active_stages, not deployables).
    assert kick_flight.rocket.motor.burn_out_time == pytest.approx(1.0)
    for state in kick_flight.solution:
        assert all(math.isfinite(v) for v in state)


def test_plot_timeline_marks_every_event_and_plots_every_flight(example_plain_env):
    # A "mission profile" chart: altitude vs time for every flight, with
    # every timeline event (ignition, burnout, separation, ejection,
    # apogee, impact, ...) marked at its own time - useful today from
    # Mission's own deterministic timeline, and designed to keep working
    # unchanged if that timeline is ever built from real Event objects
    # instead (upstream PR #968) - plot_timeline() only ever reads
    # mission.timeline's (time, name) tuples, never how they got there.
    booster, sustainer = _two_stage_vehicle(booster_separation=0.5)
    vehicle = MultiStageRocket(stages=[booster, sustainer])

    mission = Mission(
        vehicle=vehicle, environment=example_plain_env, rail_length=1.0,
        inclination=84, heading=30, max_time=5,
    )

    assert mission.plot_timeline(filename=None) is None
    ax = plt.gca()

    # One vertical marker line (2-point, x0==x1) per timeline event.
    event_lines = [
        line for line in ax.lines
        if len(line.get_xdata()) == 2 and line.get_xdata()[0] == line.get_xdata()[1]
    ]
    assert len(event_lines) == len(mission.timeline)

    # One trajectory curve (many points) per flight.
    trajectory_lines = [line for line in ax.lines if len(line.get_xdata()) > 2]
    assert len(trajectory_lines) == len(mission.all_flights)


def test_plot_timeline_runs_for_the_degenerate_single_flight_case(
    calisto, example_plain_env
):
    mission = Mission(
        vehicle=calisto, environment=example_plain_env, rail_length=5.2,
        inclination=85, heading=0,
    )

    assert mission.plot_timeline(filename=None) is None


def test_mission_timeline_omits_apogee_for_a_flight_truncated_before_reaching_it(
    example_plain_env,
):
    # Flight.apogee_time defaults to 0 (not "not found") when a flight
    # ends - via separation, ejection, or max_time - before it ever
    # reaches a genuine local-altitude-maximum. The full-stack flight
    # here is cut short by booster separation while still accelerating
    # upward (never actually apogees) - recording "apogee:<name>" at
    # t=0 for it would be flatly wrong (claiming apogee at liftoff).
    booster, sustainer = _two_stage_vehicle(booster_separation=0.5)
    vehicle = MultiStageRocket(stages=[booster, sustainer])

    mission = Mission(
        vehicle=vehicle, environment=example_plain_env, rail_length=1.0,
        inclination=84, heading=30, max_time=10,
    )

    stack_flight = mission.flights["booster"][0]
    assert stack_flight.solution[-1][6] > 0  # still ascending when it ends
    event_names = [name for _, name in mission.timeline]
    assert f"apogee:{stack_flight.name}" not in event_names

    # The two post-separation flights DO reach a genuine apogee within
    # their own duration - those must still be recorded correctly.
    booster_flight = mission.flights["booster"][-1]
    sustainer_flight = mission.flights["sustainer"][-1]
    event_times = dict((name, t) for t, name in mission.timeline)
    assert event_times[f"apogee:{booster_flight.name}"] == pytest.approx(
        booster_flight.apogee_time
    )
    assert event_times[f"apogee:{sustainer_flight.name}"] == pytest.approx(
        sustainer_flight.apogee_time
    )


def test_flight_covering_time_finds_the_right_flight(example_plain_env):
    booster, sustainer = _two_stage_vehicle(booster_separation=0.5)
    vehicle = MultiStageRocket(stages=[booster, sustainer])

    mission = Mission(
        vehicle=vehicle, environment=example_plain_env, rail_length=1.0,
        inclination=84, heading=30, max_time=10,
    )

    all_flights = mission.all_flights
    assert len(all_flights) == 3
    stack_flight, booster_flight, sustainer_flight = all_flights[0], all_flights[1], all_flights[2]

    assert mission._flight_covering_time(0.0) is stack_flight
    assert mission._flight_covering_time(0.75) is stack_flight
    # at the exact separation instant, either the ending stack flight or
    # the just-starting child is an equally correct answer (position is
    # continuous across the handoff) - only assert it resolves to ONE
    # of the two plausible flights, not a specific one.
    assert mission._flight_covering_time(1.5) in (
        stack_flight, booster_flight, sustainer_flight,
    )
    assert mission._flight_covering_time(booster_flight.apogee_time) is booster_flight
    assert mission._flight_covering_time(sustainer_flight.t_final) is sustainer_flight
    assert mission._flight_covering_time(1000.0) is None


def test_plot_trajectory_events_marks_every_resolvable_event(example_plain_env):
    # 3D trajectory with every timeline event as a colored, labeled-by-
    # legend POINT (not a line - a line has no natural per-instant
    # meaning in 3D the way a vertical line does on an altitude-vs-time
    # axis). Reads mission.timeline the same way plot_timeline() does -
    # forward compatible with a future Event-object-backed timeline for
    # the same reason.
    booster, sustainer = _two_stage_vehicle(booster_separation=0.5)
    vehicle = MultiStageRocket(stages=[booster, sustainer])

    mission = Mission(
        vehicle=vehicle, environment=example_plain_env, rail_length=1.0,
        inclination=84, heading=30, max_time=10,
    )

    assert mission.plot_trajectory_events(filename=None) is None
    ax = plt.gcf().axes[0]

    resolvable = [
        (t, name) for t, name in mission.timeline
        if mission._flight_covering_time(t) is not None
    ]
    assert len(resolvable) == len(mission.timeline)  # every event resolves

    total_points = sum(
        collection.get_offsets().shape[0] if hasattr(collection, "get_offsets")
        else len(collection._offsets3d[0])
        for collection in ax.collections
    )
    assert total_points == len(mission.timeline)


def test_plot_trajectory_events_runs_for_the_degenerate_single_flight_case(
    calisto, example_plain_env
):
    mission = Mission(
        vehicle=calisto, environment=example_plain_env, rail_length=5.2,
        inclination=85, heading=0,
    )

    assert mission.plot_trajectory_events(filename=None) is None


def test_mission_timeline_includes_apogee_for_an_apogee_terminated_flight(
    example_plain_env,
):
    # A deployable-ejection flight runs with terminate_on_apogee=True -
    # it ends EXACTLY at its own apogee, by design. apogee_time then
    # equals (or sits right at the boundary of) that flight's own end
    # time - a strict "start < apogee_time < end" check would wrongly
    # exclude this genuine apogee, the same way it correctly excludes a
    # flight that never reaches one at all.
    vehicle, _stage, _deployable = _single_stage_with_deployable_vehicle()

    mission = Mission(
        vehicle=vehicle, environment=example_plain_env, rail_length=1.0,
        inclination=90, heading=0, max_time=20,
    )

    carrier_flight = mission.flights["carrier"][0]
    assert carrier_flight.solution[-1][6] <= 0  # already at/past apogee by the end
    event_names = [name for _, name in mission.timeline]
    assert f"apogee:{carrier_flight.name}" in event_names
