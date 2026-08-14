from rocketpy import Flight
from rocketpy.rocket.multistage import MultiStageRocket, Stage
from rocketpy.simulation.mission import Mission


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
