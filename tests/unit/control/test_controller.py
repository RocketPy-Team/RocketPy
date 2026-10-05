from types import SimpleNamespace

import pytest

from rocketpy.control.controlled import Controlled
from rocketpy.control.controller import _Controller
from rocketpy.simulation.events.event_context import EventContext


def _run(controller):
    """Run a controller's event once and return what its function logged."""
    context = EventContext(
        time=0.0,
        canonical_state=[0.0] * 13,
        raw_state=[0.0] * 13,
        phase_names=(),
    )
    controller.event(context)
    return controller.log[-1]


def test_a_named_object_is_read_by_name_and_by_position():
    brakes = SimpleNamespace(deployment_level=0.0)

    def function(context):
        context.controlled.air_brakes.deployment_level = 0.5
        return {
            "by_position": context.controlled[0],
            "controller": context.controller,
            "count": len(context.controlled),
        }

    controller = _Controller(
        function, brakes, sampling_rate=10, controlled_objects_name="air_brakes"
    )
    log = _run(controller)

    assert brakes.deployment_level == 0.5
    assert log["by_position"] is brakes
    assert log["controller"] is controller
    assert log["count"] == 1


def test_several_objects_are_read_by_their_own_names_in_order():
    brakes, fins = SimpleNamespace(), SimpleNamespace()
    controller = _Controller(
        lambda context: {
            "brakes": context.controlled.brakes,
            "fins": context.controlled.fins,
            "all": list(context.controlled),
        },
        [brakes, fins],
        sampling_rate=10,
        controlled_objects_name=["brakes", "fins"],
    )
    log = _run(controller)

    assert log["brakes"] is brakes
    assert log["fins"] is fins
    assert log["all"] == [brakes, fins]


def test_objects_without_names_are_read_by_position():
    brakes, fins = SimpleNamespace(), SimpleNamespace()
    controller = _Controller(
        lambda context: {"second": context.controlled[1]},
        [brakes, fins],
        sampling_rate=10,
    )

    assert _run(controller)["second"] is fins


def test_a_name_no_object_was_given_says_which_names_exist():
    controlled = Controlled([SimpleNamespace()], ["air_brakes"])

    with pytest.raises(AttributeError, match="fins.*available.*air_brakes"):
        _ = controlled.fins


def test_the_controlled_objects_leave_the_context_with_the_controller():
    """The next event of the step must not see another controller's objects."""
    controller = _Controller(lambda context: None, SimpleNamespace(), sampling_rate=10)
    context = EventContext(
        time=0.0, canonical_state=[0.0] * 13, raw_state=[0.0] * 13, phase_names=()
    )
    controller.event(context)
    assert context.controlled is not None

    context.bind(SimpleNamespace(sampling_rate=None))

    assert context.controlled is None
    assert context.controller is None


def test_a_name_may_be_one_the_context_itself_uses():
    """Objects live under ``context.controlled``, so no name is reserved."""
    controller = _Controller(
        lambda context: {"time": context.time, "mine": context.controlled.time},
        "the object",
        sampling_rate=10,
        controlled_objects_name="time",
    )

    assert _run(controller) == {"time": 0.0, "mine": "the object"}


@pytest.mark.parametrize("name", ["air brakes", "1st", "_hidden", "class", ""])
def test_a_name_that_cannot_follow_a_dot_is_rejected(name):
    with pytest.raises(ValueError, match="context.controlled"):
        _Controller(
            lambda context: None,
            SimpleNamespace(),
            sampling_rate=10,
            controlled_objects_name=name,
        )


def test_names_must_match_the_objects():
    one, two = SimpleNamespace(), SimpleNamespace()
    with pytest.raises(ValueError, match="unique"):
        _Controller(
            lambda context: None, [one, two], 10, controlled_objects_name=["a", "a"]
        )
    with pytest.raises(ValueError, match="Length"):
        _Controller(lambda context: None, [one, two], 10, controlled_objects_name=["a"])
    with pytest.raises(ValueError, match="not a list"):
        _Controller(lambda context: None, one, 10, controlled_objects_name=["a"])
    with pytest.raises(TypeError):
        _Controller(lambda context: None, one, 10, controlled_objects_name=3)


def test_rebinding_points_the_context_at_the_new_objects():
    """A loaded rocket reconnects its controller to its own rebuilt objects."""
    saved, loaded = SimpleNamespace(), SimpleNamespace()
    controller = _Controller(
        lambda context: {
            "by_name": context.controlled.air_brakes,
            "by_position": context.controlled[0],
        },
        saved,
        sampling_rate=10,
        controlled_objects_name="air_brakes",
    )

    controller.rebind_controlled_objects(loaded)
    log = _run(controller)

    assert controller.controlled_objects is loaded
    assert log["by_name"] is loaded
    assert log["by_position"] is loaded
