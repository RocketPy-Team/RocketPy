import pytest

from rocketpy.simulation.events import Event


def test_verify_trigger_accepts_only_kwargs():
    def trigger(**kwargs) -> bool:
        return True

    def action(**kwargs):
        return None

    event = Event(trigger=trigger, action=action, name="test")
    assert event.trigger is trigger


def test_verify_trigger_rejects_missing_kwargs():
    def trigger(a, b) -> bool:
        return True

    def action(**kwargs):
        return None

    with pytest.raises(
        ValueError,
        match=r"The Trigger function of the test event must accept only keyword arguments. def trigger\(\*\*kwargs\) -> bool:",
    ):
        Event(trigger=trigger, action=action, name="test")


def test_verify_trigger_rejects_args_with_kwargs():
    def trigger(a, b, **kwargs) -> bool:
        return True

    def action(**kwargs):
        return None

    with pytest.raises(
        ValueError,
        match=r"The Trigger function of the test event must accept only keyword arguments. def trigger\(\*\*kwargs\) -> bool:",
    ):
        Event(trigger=trigger, action=action, name="test")


def test_verify_trigger_rejects_triggers_without_bool_return_annotation():
    def trigger(**kwargs):
        return True

    def action(**kwargs):
        return None

    with pytest.raises(
        ValueError,
        match="The Trigger function of the test event must return a boolean value and must be annotated with '-> bool' for type checking.\n"
        + r"def trigger\(\*\*kwargs\) -> bool\:",
    ):
        Event(trigger=trigger, action=action, name="test")
