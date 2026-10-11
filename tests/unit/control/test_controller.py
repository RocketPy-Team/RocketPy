import copy
import json
from types import SimpleNamespace

import pytest

from rocketpy import AirBrakes, Controller, GenericSurface
from rocketpy._encoders import RocketPyDecoder, RocketPyEncoder
from rocketpy.control.control_signal import initial_controls
from rocketpy.control.controller import _PositionalController
from rocketpy.simulation.events.event_context import EventContext


def _air_brakes(name="AirBrakes"):
    return AirBrakes(
        drag_coefficient_curve=lambda deployment_level, mach: deployment_level,
        reference_area=0.01,
        name=name,
    )


def _canards():
    return GenericSurface(
        0.01,
        0.1,
        {"cl": lambda mach, deflection: deflection},
        name="Canards",
        controls=("deflection",),
        control_limits={"deflection": (-0.2, 0.2)},
    )


def _flight_of(*actuators):
    """A stand-in flight holding the control inputs of ``actuators``."""
    controls, channels, limits = initial_controls(actuators)
    return SimpleNamespace(
        control_signal=controls,
        _control_channels=channels,
        _control_limits=limits,
        rocket=SimpleNamespace(actuators=list(actuators)),
    )


def _run(controller, flight, time=0.0):
    """Run ``controller`` once at ``time`` in the stand-in ``flight``."""
    context = EventContext(
        time=time,
        canonical_state=[0.0] * 13,
        raw_state=[0.0] * 13,
        phase_names=(),
        flight=flight,
        rocket=flight.rocket,
        sensors=[],
        environment=None,
    )
    controller(context)
    return controller.callback_log[-1] if controller.callback_log else None


def test_a_controller_writes_the_flight_controls_not_the_object():
    brakes = _air_brakes()
    flight = _flight_of(brakes)

    def function(context):
        context.controls["air_brakes.deployment_level"] = 0.7

    controller = Controller(function, sampling_rate=10)
    _run(controller, flight)

    assert flight.control_signal["air_brakes.deployment_level"] == 0.7
    assert brakes.deployment_level == 0.0
    assert controller.sampling_rate == 10
    assert controller.priority == 3


def test_a_write_is_kept_inside_the_limits():
    flight = _flight_of(_air_brakes())
    controller = Controller(
        lambda context: context.controls.update({"air_brakes.deployment_level": 3}),
        sampling_rate=10,
    )
    _run(controller, flight)
    assert flight.control_signal["air_brakes.deployment_level"] == 1.0


def test_a_returned_dict_is_recorded():
    flight = _flight_of(_air_brakes())

    def function(context):
        return {"estimated_apogee": 3000.0 + context.time}

    controller = Controller(function, sampling_rate=10, name="Apogee Controller")
    _run(controller, flight, time=0.0)
    _run(controller, flight, time=0.1)

    record = controller.records["estimated_apogee"]
    assert list(record.x_array) == [0.0, 0.1]
    assert record(0.05) == 3000.0


def test_a_returned_value_that_is_not_a_number_is_explained():
    controller = Controller(lambda context: {"state": [1, 2]}, sampling_rate=10)
    with pytest.raises(TypeError, match="must be a single number"):
        _run(controller, _flight_of(_air_brakes()))


def test_a_returned_tuple_is_not_recorded():
    controller = Controller(lambda context: (1.0, 2.0), sampling_rate=10)
    assert _run(controller, _flight_of(_air_brakes())) == (1.0, 2.0)
    assert not controller.records


@pytest.mark.parametrize(
    "writes",
    [
        "canards.deflection",
        "canards",
        ["canards"],
        ("canards.deflection",),
    ],
)
def test_writes_allows_the_channels_named(writes):
    flight = _flight_of(_air_brakes(), _canards())

    def function(context):
        context.controls["canards.deflection"] = 0.1

    controller = Controller(function, sampling_rate=10, writes=writes)
    _run(controller, flight)
    assert flight.control_signal["canards.deflection"] == 0.1


def test_writes_takes_the_actuator_itself():
    canards = _canards()
    flight = _flight_of(_air_brakes(), canards)
    controller = Controller(
        lambda context: context.controls.update({"canards.deflection": 0.1}),
        sampling_rate=10,
        writes=canards,
    )
    assert controller.writes == ("canards",)
    _run(controller, flight)
    assert flight.control_signal["canards.deflection"] == 0.1


def test_a_write_outside_writes_raises_and_is_undone():
    flight = _flight_of(_air_brakes(), _canards())

    def function(context):
        context.controls["canards.deflection"] = 0.1
        context.controls["air_brakes.deployment_level"] = 0.5

    controller = Controller(function, sampling_rate=10, writes="canards")
    with pytest.raises(ValueError, match="'air_brakes.deployment_level'"):
        _run(controller, flight)
    assert flight.control_signal == {
        "air_brakes.deployment_level": 0.0,
        "canards.deflection": 0.0,
    }


def test_writing_an_unchanged_value_outside_writes_is_allowed():
    flight = _flight_of(_air_brakes(), _canards())

    def function(context):
        context.controls["air_brakes.deployment_level"] = 0.0

    controller = Controller(function, sampling_rate=10, writes="canards")
    _run(controller, flight)


def test_writes_naming_no_channel_says_which_exist():
    controller = Controller(lambda context: None, sampling_rate=10, writes="fins")
    with pytest.raises(ValueError, match="air_brakes.deployment_level"):
        _run(controller, _flight_of(_air_brakes()))


@pytest.mark.parametrize("writes", [3, [""], [None]])
def test_writes_must_be_names_or_actuators(writes):
    with pytest.raises((TypeError, ValueError)):
        Controller(lambda context: None, sampling_rate=10, writes=writes)


def test_writes_is_checked_again_in_the_next_flight():
    controller = Controller(lambda context: None, sampling_rate=10, writes="canards")
    _run(controller, _flight_of(_canards()))
    controller.reset()
    with pytest.raises(ValueError, match="no control channel"):
        _run(controller, _flight_of(_air_brakes()))


@pytest.mark.parametrize("function", [lambda: None, lambda a, b: None, 3])
def test_the_function_must_take_the_context(function):
    with pytest.raises((TypeError, ValueError)):
        Controller(function, sampling_rate=10)


@pytest.mark.parametrize("sampling_rate", [0, -1, None])
def test_the_sampling_rate_must_be_positive(sampling_rate):
    with pytest.raises(ValueError, match="sampling rate"):
        Controller(lambda context: None, sampling_rate=sampling_rate)


def test_memory_starts_again_at_every_flight():
    def count(context):
        context.event.memory["calls"] += 1

    controller = Controller(count, sampling_rate=10, memory={"calls": 0})
    flight = _flight_of(_air_brakes())
    _run(controller, flight)
    _run(controller, flight)
    assert controller.memory["calls"] == 2
    controller.reset()
    assert controller.memory["calls"] == 0


def test_a_controller_write_marks_its_commands():
    """A controller is a plain event: a write that changes a control cuts the
    solver step at the sampling time, and a write that changes nothing does
    not."""
    flight = _flight_of(_air_brakes())
    controller = Controller(
        lambda context: context.controls.update({"air_brakes.deployment_level": 0.5}),
        sampling_rate=10,
    )
    _run(controller, flight)
    assert not controller.changes_dynamics
    assert controller.commands.wrote_controls is True
    assert controller.commands.control_changes == {"air_brakes.deployment_level": 0.5}
    _run(controller, flight)
    assert controller.commands.wrote_controls is False


def test_a_positional_function_moves_the_air_brakes_by_name(recwarn):
    """The deprecated positional form receives the air brakes as an object
    whose deployment level is the flight's channel."""
    brakes = _air_brakes()
    flight = _flight_of(_canards(), brakes)
    received = []

    def legacy(time, sampling_rate, state, observed_variables, air_brakes):  # pylint: disable=unused-argument
        received.append((sampling_rate, observed_variables))
        air_brakes.deployment_level = 0.3
        return air_brakes.deployment_level

    controller = Controller(
        _PositionalController(legacy, 20, actuator_name="AirBrakes"),
        sampling_rate=20,
        memory={"observed_variables": [1.0]},
    )
    assert _run(controller, flight) == 0.3
    assert received == [(20, [1.0])]
    assert flight.control_signal["air_brakes.deployment_level"] == 0.3
    assert brakes.deployment_level == 0.0
    assert not recwarn.list


def test_a_positional_function_without_a_name_needs_one_actuator():
    controller = Controller(
        _PositionalController(lambda t, r, s, o, b: None, 10), sampling_rate=10
    )
    _run(controller, _flight_of(_air_brakes()))
    with pytest.raises(ValueError, match="2 that it could be"):
        _run(controller, _flight_of(_air_brakes(), _canards()))


def _round_trip(obj, **options):
    return json.loads(
        json.dumps(obj, cls=RocketPyEncoder, **options), cls=RocketPyDecoder
    )


def _saved_function(context):
    context.controls["canards.deflection"] = 0.1
    return {"time": context.time}


def test_a_controller_is_saved_and_loaded():
    controller = Controller(
        _saved_function,
        sampling_rate=25,
        writes="canards",
        memory={"gain": 2.0},
        name="Roll Damper",
        enabled=False,
        disable_on="apogee",
        enable_on=1.5,
        priority=5,
    )
    loaded = _round_trip(controller)

    assert isinstance(loaded, Controller)
    assert loaded.sampling_rate == 25
    assert loaded.writes == ("canards",)
    assert loaded.memory == {"gain": 2.0}
    assert loaded.name == "Roll Damper"
    assert loaded.enabled is False
    assert loaded._gates == ("apogee", 1.5)
    assert loaded.priority == 5
    loaded.enabled = True
    flight = _flight_of(_canards())
    assert _run(loaded, flight) == {"time": 0.0}
    assert flight.control_signal["canards.deflection"] == 0.1


def test_a_controller_saved_without_its_function_loads_disabled():
    controller = Controller(_saved_function, sampling_rate=10)
    with pytest.warns(UserWarning, match="could not be loaded"):
        loaded = _round_trip(controller, allow_pickle=False)
    assert loaded.enabled is False
    loaded.enabled = True
    flight = _flight_of(_canards())
    assert _run(loaded, flight) is None
    assert flight.control_signal["canards.deflection"] == 0.0


def test_a_controller_saved_by_the_old_class_loads():
    """What ``_Controller.to_dict`` saved, with its extra keys, loads as a
    Controller."""
    data = {
        "controller_function": Controller(_saved_function, 10).to_dict()[
            "controller_function"
        ],
        "sampling_rate": 10,
        "name": "Old",
        "controlled_objects_name": "air_brakes",
        "controlled_objects_hash": 1234,
        "memory": {},
        "enabled": True,
        "disable_on": None,
        "enable_on": None,
        "signature": {"module": "rocketpy.control.controller", "name": "_Controller"},
    }
    loaded = json.loads(json.dumps(data), cls=RocketPyDecoder)
    assert isinstance(loaded, Controller)
    assert loaded.name == "Old"


def test_a_controller_can_be_copied():
    controller = Controller(_saved_function, sampling_rate=10, memory={"a": [1]})
    copied = copy.deepcopy(controller)
    assert copied.callback.__self__ is copied
    flight = _flight_of(_canards())
    _run(copied, flight)
    assert copied.triggered_times == [0.0]
    assert controller.triggered_times == []


def test_info_lists_what_the_controller_writes(capsys):
    controller = Controller(
        _saved_function, sampling_rate=10, writes="canards", memory={"gain": 1}
    )
    controller.info()
    out = capsys.readouterr().out
    assert "_saved_function" in out
    assert "Writes: canards" in out
    assert "Memory keys: gain" in out
