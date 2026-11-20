import copy
import json
from types import SimpleNamespace

import pytest

from rocketpy import AirBrakes, Controller, GenericSurface
from rocketpy._encoders import RocketPyDecoder, RocketPyEncoder
from rocketpy.control._helpers import initial_controls
from rocketpy.control._legacy import _PositionalController
from rocketpy.simulation.events.event_context import EventContext
from rocketpy.simulation.events.event_execution import call_event
from rocketpy.simulation.events.views import output_events


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


def _flight_of(*controllables):
    """A stand-in flight holding the control inputs of ``controllables``."""
    controls, channels, limits = initial_controls(controllables)
    return SimpleNamespace(
        t=0.0,
        _controls=controls,
        _control_names=channels,
        _control_limits=limits,
        rocket=SimpleNamespace(controllables=list(controllables)),
        # The rows a positional controller function receives as state_history
        solution=SimpleNamespace(_rows=[[0.0] * 14]),
    )


def _run(controller, flight, time=0.0, sensors=None, sharing=()):
    """Run ``controller`` once at ``time`` in the stand-in ``flight``, whose
    events with outputs are among ``sharing``."""
    flight.t = time
    context = EventContext(
        time=time,
        canonical_state=[0.0] * 13,
        raw_state=[0.0] * 13,
        phase_names=(),
        controls=flight._controls,
        rocket=flight.rocket,
        sensors=sensors or {},
        environment=None,
        solution=flight.solution,
        output_events=output_events(sharing),
    )
    call_event(flight, controller, context, None, 0, 0)


def test_a_controller_writes_the_flight_controls_not_the_object():
    brakes = _air_brakes()
    flight = _flight_of(brakes)

    def function(context):
        context.controls.air_brakes.deployment_level = 0.7

    controller = Controller(function, sampling_rate=10)
    _run(controller, flight)

    assert flight._controls["air_brakes.deployment_level"] == 0.7
    assert brakes.initial_deployment_level == 0.0
    assert controller.sampling_rate == 10
    assert controller.priority == 3


def test_a_write_is_kept_inside_the_limits():
    flight = _flight_of(_air_brakes())
    controller = Controller(
        lambda context: setattr(context.controls.air_brakes, "deployment_level", 3),
        sampling_rate=10,
    )
    _run(controller, flight)
    assert flight._controls["air_brakes.deployment_level"] == 1.0


def test_a_controller_records_with_the_context():
    flight = _flight_of(_air_brakes())

    def function(context):
        context.record(estimated_apogee=3000.0 + context.time)

    controller = Controller(function, sampling_rate=10, name="Apogee Controller")
    _run(controller, flight, time=0.0)
    _run(controller, flight, time=0.1)

    assert controller.records["estimated_apogee"] == [(0.0, 3000.0), (0.1, 3000.1)]


def test_an_outer_loop_hands_a_setpoint_to_an_inner_loop():
    """A controller's outputs are found under its name: the controller writes
    them and another controller reads them."""

    def guidance_function(context):
        context.outputs.guidance.pitch_setpoint = 0.2

    def attitude_function(context):
        context.controls.canards.deflection = context.outputs.guidance.pitch_setpoint

    guidance = Controller(
        guidance_function,
        sampling_rate=10,
        name="Guidance",
        outputs={"pitch_setpoint": 0.05},
    )
    attitude = Controller(attitude_function, sampling_rate=100, name="Attitude")
    assert guidance.outputs == ("pitch_setpoint",)
    assert guidance.output_defaults == {"pitch_setpoint": 0.05}
    assert attitude.outputs == ()

    flight = _flight_of(_canards())
    both = (guidance, attitude)
    # An output is not a control input of the flight
    assert list(flight._controls) == ["canards.deflection"]
    _run(attitude, flight, sharing=both)
    assert flight._controls["canards.deflection"] == 0.05
    _run(guidance, flight, time=0.1, sharing=both)
    # Writing an output does not change the trajectory
    assert guidance.commands.control_changes == {}
    _run(attitude, flight, time=0.1, sharing=both)
    assert flight._controls["canards.deflection"] == 0.2
    assert guidance.output_history == {"pitch_setpoint": [(0.1, 0.2)]}


def test_a_output_holds_any_value_and_is_written_by_its_event_only():
    def estimator_function(context):
        context.outputs.estimator.attitude = [1.0, 0.0, 0.0, 0.0]
        context.outputs["Estimator"].mode = "coast"

    def reader_function(context):
        context.memory["read"] = context.outputs.estimator.mode
        context.outputs.estimator.mode = "boost"

    estimator = Controller(
        estimator_function,
        sampling_rate=10,
        name="Estimator",
        outputs=("attitude", "mode"),
    )
    reader = Controller(reader_function, sampling_rate=10, name="Reader")
    flight = _flight_of(_canards())
    both = (estimator, reader)
    assert estimator.output_defaults == {"attitude": None, "mode": None}
    _run(estimator, flight, sharing=both)
    assert estimator.output_history == {
        "attitude": [(0.0, [1.0, 0.0, 0.0, 0.0])],
        "mode": [(0.0, "coast")],
    }
    with pytest.raises(AttributeError, match="written only by 'Estimator'"):
        _run(reader, flight, sharing=both)
    assert reader.memory["read"] == "coast"
    assert estimator._outputs["mode"] == "coast"


def test_a_output_goes_back_with_the_flight_and_starts_again():
    controller = Controller(
        lambda context: None,
        sampling_rate=10,
        name="Guidance",
        outputs={"setpoint": 0.05, "path": None},
    )
    controller._write_output(0.1, "setpoint", 0.2)
    controller._write_output(0.2, "setpoint", 0.3)
    controller._write_output(0.2, "path", [1, 2])
    controller._drop_after(0.1)
    assert controller._outputs == {"setpoint": 0.2, "path": None}
    assert controller.output_history == {"setpoint": [(0.1, 0.2)], "path": []}
    controller.reset()
    assert controller._outputs == {"setpoint": 0.05, "path": None}
    assert controller.output_history == {}


def test_printing_the_context_lists_what_a_controller_can_use():
    seen = {}

    def function(context):
        seen["text"] = repr(context)
        seen["names"] = dir(context)

    controller = Controller(
        function, sampling_rate=10, name="Guidance", outputs=("setpoint",)
    )
    _run(controller, _flight_of(_canards()), sharing=(controller,))
    assert "EventContext at t = 0 s, event 'Guidance'" in seen["text"]
    assert "canards: deflection" in seen["text"]
    assert "guidance: setpoint" in seen["text"]
    assert "x, y, z, vx, vy, vz" in seen["text"]
    # Only what a function reads, not the methods of the dictionary
    assert {"time", "state", "controls", "outputs", "record"} <= set(seen["names"])
    assert not {"pop", "clear", "update", "keys"} & set(seen["names"])


def test_output_names_are_checked():
    with pytest.raises(ValueError, match="letters, digits and underscores"):
        Controller(lambda context: None, sampling_rate=10, outputs=("1st",))


def test_a_rocket_checks_its_controllers_outputs(calisto):
    controller = Controller(
        lambda context: None, sampling_rate=10, name="Guidance", outputs=("x", "y")
    )
    calisto.add_controller(controller)
    # Outputs are not control inputs
    assert calisto.control_names == ()
    clash = Controller(
        lambda context: None, sampling_rate=10, name="Guidance", outputs=("z",)
    )
    with pytest.raises(ValueError, match="both have outputs"):
        calisto.add_controller(clash)
    assert calisto.controllers == (controller,)


def test_a_controller_chooses_how_the_flight_reaches_its_samples():
    assert Controller(lambda context: None, sampling_rate=10).time_overshootable
    stopped = Controller(
        lambda context: None, sampling_rate=10, time_overshootable=False
    )
    assert stopped.time_overshootable is False


def test_what_a_controller_returns_is_ignored():
    controller = Controller(lambda context: {"state": [1, 2]}, sampling_rate=10)
    _run(controller, _flight_of(_air_brakes()), time=0.5)
    assert not controller.records


@pytest.mark.parametrize("function", [lambda: None, lambda a, b: None, 3])
def test_the_function_must_take_the_context(function):
    with pytest.raises((TypeError, ValueError)):
        Controller(function, sampling_rate=10)


@pytest.mark.parametrize("sampling_rate", [0, -1, None])
def test_the_sampling_rate_must_be_positive(sampling_rate):
    """A controller always has a rate: without one it would run once per
    solver step, as often as the solver's tolerances happen to make it."""
    with pytest.raises(ValueError, match="sampling rate"):
        Controller(lambda context: None, sampling_rate=sampling_rate)


def test_memory_starts_again_at_every_flight():
    def count(context):
        context.memory["calls"] += 1

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
        lambda context: setattr(context.controls.air_brakes, "deployment_level", 0.5),
        sampling_rate=10,
    )
    _run(controller, flight)
    assert controller.commands.control_changes == {"air_brakes.deployment_level": 0.5}
    assert controller.commands.changes_trajectory is True
    _run(controller, flight)
    assert controller.commands.changes_trajectory is False


def test_a_positional_function_moves_the_air_brakes_by_name(recwarn):
    """The deprecated positional form of v1.13 receives the state history and
    the air brakes as an object whose deployment level is the flight's
    channel."""
    brakes = _air_brakes()
    flight = _flight_of(_canards(), brakes)
    received = []

    def legacy(  # pylint: disable=unused-argument
        time, sampling_rate, state, state_history, observed_variables, air_brakes
    ):
        received.append((sampling_rate, state_history, list(observed_variables)))
        air_brakes.deployment_level = 0.3
        return air_brakes.deployment_level

    controller = Controller(
        _PositionalController(
            legacy, 20, part_name="AirBrakes", initial_observed_variables=1.0
        ),
        sampling_rate=20,
    )
    _run(controller, flight)
    assert received == [(20, flight.solution._rows, [1.0])]
    assert controller.controller_function.observed_variables == [1.0, 0.3]
    # Kept on the wrapper, not in the memory copied before every run
    assert controller.memory == {}
    assert flight._controls["air_brakes.deployment_level"] == 0.3
    assert brakes.initial_deployment_level == 0.0
    assert not recwarn.list


def test_a_positional_function_gathers_what_it_returns():
    """As in v1.13, observed_variables starts with the initial value and gains
    every value returned other than None; it starts again at every flight."""
    seen = []

    def legacy(t, rate, state, history, observed, air_brakes):  # pylint: disable=unused-argument
        seen.append(list(observed))
        return None if len(seen) == 2 else len(seen)

    controller = Controller(
        _PositionalController(legacy, 10, initial_observed_variables="start"),
        sampling_rate=10,
    )
    flight = _flight_of(_air_brakes())
    for _ in range(3):
        _run(controller, flight)
    assert seen == [["start"], ["start", 1], ["start", 1]]
    assert controller.controller_function.observed_variables == ["start", 1, 3]
    # A saved controller starts with the initial value again
    saved = copy.deepcopy(controller.controller_function)
    assert saved.observed_variables == ["start"]
    controller.reset()
    assert controller.controller_function.observed_variables == ["start"]


def test_a_positional_function_receives_the_sensor_objects():
    """As in v1.13, the seventh argument is the list of the rocket's sensors,
    one per position, in the order they were added."""
    barometer, gyroscope = SimpleNamespace(name="b"), SimpleNamespace(name="g")
    received = []

    def legacy(t, rate, state, history, observed, air_brakes, sensors):  # pylint: disable=unused-argument
        received.append(sensors)

    controller = Controller(_PositionalController(legacy, 10), sampling_rate=10)
    _run(
        controller,
        _flight_of(_air_brakes()),
        sensors={"barometer_1": (barometer, None), "gyroscope": (gyroscope, None)},
    )
    assert received == [[barometer, gyroscope]]


def test_a_positional_function_without_a_name_needs_one_part():
    controller = Controller(
        _PositionalController(lambda t, r, s, h, o, b: None, 10), sampling_rate=10
    )
    _run(controller, _flight_of(_air_brakes()))
    with pytest.raises(ValueError, match="2 that it could be"):
        _run(controller, _flight_of(_air_brakes(), _canards()))


def _round_trip(obj, **options):
    return json.loads(
        json.dumps(obj, cls=RocketPyEncoder, **options), cls=RocketPyDecoder
    )


def _saved_function(context):
    context.controls.canards.deflection = 0.1
    context.record(time=context.time)


def test_a_controller_is_saved_and_loaded():
    controller = Controller(
        _saved_function,
        sampling_rate=25,
        memory={"gain": 2.0},
        name="Roll Damper",
        enabled=False,
        disable_on="apogee",
        enable_on=1.5,
        outputs={"rate_setpoint": 0.3},
        time_overshootable=False,
    )
    loaded = _round_trip(controller)

    assert isinstance(loaded, Controller)
    assert loaded.sampling_rate == 25
    assert loaded.memory == {"gain": 2.0}
    assert loaded.name == "Roll Damper"
    assert loaded.enabled is False
    assert loaded._gates == ("apogee", 1.5)
    assert loaded.output_defaults == {"rate_setpoint": 0.3}
    assert loaded.time_overshootable is False
    loaded.enabled = True
    flight = _flight_of(_canards())
    _run(loaded, flight)
    assert loaded._records == {"time": [(0.0, 0.0)]}
    assert flight._controls["canards.deflection"] == 0.1


def test_a_controller_saved_without_its_function_loads_disabled():
    controller = Controller(_saved_function, sampling_rate=10)
    with pytest.warns(UserWarning, match="could not be loaded"):
        loaded = _round_trip(controller, allow_pickle=False)
    assert loaded.enabled is False
    loaded.enabled = True
    flight = _flight_of(_canards())
    _run(loaded, flight)
    assert flight._controls["canards.deflection"] == 0.0


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
    flight = _flight_of(_canards())
    _run(copied, flight)
    assert copied.triggered_times == [0.0]
    assert controller.triggered_times == []


def test_info_lists_the_function_and_memory(capsys):
    controller = Controller(
        _saved_function, sampling_rate=10, memory={"gain": 1}, outputs=("setpoint",)
    )
    controller.info()
    out = capsys.readouterr().out
    assert "_saved_function" in out
    assert "Outputs: setpoint" in out
    assert "Memory keys: gain" in out


def _controller_with_a_gain(context, gain=2.0):  # pylint: disable=unused-argument
    return None


def test_a_function_with_extra_default_arguments_takes_the_context():
    """Saved and loaded as it is, not read as a v1.13 positional function."""
    controller = Controller(_controller_with_a_gain, 10, name="Gain")
    loaded = Controller.from_dict(controller.to_dict())
    assert not isinstance(loaded.controller_function, _PositionalController)
    assert loaded.enabled


def _gate(context):  # pylint: disable=unused-argument
    return False


def test_a_gate_saved_without_pickling_loads_the_controller_disabled():
    controller = Controller(lambda context: None, 10, disable_on=_gate)
    data = controller.to_dict(allow_pickle=False)
    with pytest.warns(UserWarning, match="could not be loaded"):
        loaded = Controller.from_dict(data)
    assert not loaded.enabled


def test_a_preset_gate_is_loaded_as_saved():
    controller = Controller(lambda context: None, 10, disable_on="apogee")
    loaded = Controller.from_dict(controller.to_dict())
    assert loaded.enabled and loaded._gates[0] == "apogee"
