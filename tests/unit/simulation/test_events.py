import pytest

from rocketpy.simulation.events import Event


def test_verify_trigger_accepts_required_args_with_kwargs():
    def trigger(a: int, b: float, **kwargs) -> bool:
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

    with pytest.raises(ValueError, match=r"must accept \*\*kwargs"):
        Event(trigger=trigger, action=action, name="test")
