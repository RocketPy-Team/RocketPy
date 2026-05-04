import pytest

from rocketpy.simulation.events import Event


def test_verify_trigger_accepts_only_kwargs():
    def trigger(**kwargs) -> bool:
        return True

    def action(**kwargs) -> None:
        return None

    event = Event(trigger=trigger, action=action, name="test")
    assert event.trigger is trigger


def test_verify_trigger_evaluation_of_number_of_parameters():
    def trigger(**kwargs) -> bool:
        a = kwargs["a"]
        b = kwargs["b"]
        c = kwargs["c"]
        return a + b + c == 6

    def action(**kwargs) -> None:
        return None

    kwargs_test = {"a": 1, "b": 2, "c": 3}
    assert trigger(**kwargs_test)

    event = Event(trigger=trigger, action=action, name="test")
    assert event.trigger is trigger


def test_verify_trigger_rejects_missing_kwargs():
    def trigger(a, b) -> bool:
        return True

    def action(**kwargs) -> None:
        return None

    with pytest.raises(
        ValueError,
        match=r"The Trigger function of the test event must accept only keyword arguments. def trigger\(\*\*kwargs\) -> bool:",
    ):
        Event(trigger=trigger, action=action, name="test")


def test_verify_trigger_rejects_args_with_kwargs():
    def trigger(a, b, **kwargs) -> bool:
        return True

    def action(**kwargs) -> None:
        return None

    with pytest.raises(
        ValueError,
        match=r"The Trigger function of the test event must accept only keyword arguments. def trigger\(\*\*kwargs\) -> bool:",
    ):
        Event(trigger=trigger, action=action, name="test")


def test_verify_trigger_rejects_triggers_with_no_parameters():
    def trigger() -> bool:
        return True

    def action(**kwargs) -> None:
        return None

    with pytest.raises(
        ValueError,
        match=r"The Trigger function of the test event must accept only keyword arguments. def trigger\(\*\*kwargs\) -> bool:",
    ):
        Event(trigger=trigger, action=action, name="test")


def test_verify_trigger_rejects_triggers_without_bool_return_annotation():
    def trigger(**kwargs):
        return True

    def action(**kwargs) -> None:
        return None

    with pytest.raises(
        ValueError,
        match="The Trigger function of the test event must return a boolean value and must be annotated with '-> bool' for type checking.\n"
        + r"def trigger\(\*\*kwargs\) -> bool\:",
    ):
        Event(trigger=trigger, action=action, name="test")


# The following tests verify if action functions were correctly implemented


def test_verify_action_accepts_only_kwargs():
    def trigger(**kwargs) -> bool:
        return True

    def action(**kwargs) -> None:
        return None

    event = Event(trigger=trigger, action=action, name="test")
    assert event.action is action


def test_verify_action_rejects_missing_kwargs():
    def trigger(**kwargs) -> bool:
        return True

    def action(a, b) -> None:
        return None

    with pytest.raises(
        ValueError,
        match=r"The Action function of the test event must accept only keyword arguments. def action\(\*\*kwargs\) -> None \| dict:",
    ):
        Event(trigger=trigger, action=action, name="test")


def test_verify_action_rejects_args_with_kwargs():
    def trigger(**kwargs) -> bool:
        return True

    def action(a, b, **kwargs) -> None:
        return None

    with pytest.raises(
        ValueError,
        match=r"The Action function of the test event must accept only keyword arguments. def action\(\*\*kwargs\) -> None \| dict:",
    ):
        Event(trigger=trigger, action=action, name="test")


def test_verify_action_accepts_dict_return_type():
    def trigger(**kwargs) -> bool:
        return True

    def action(**kwargs) -> dict:
        return {"key": "value"}

    event = Event(trigger=trigger, action=action, name="test")
    assert event.action is action


def test_verify_action_accepts_none_return_type():
    def trigger(**kwargs) -> bool:
        return True

    def action(**kwargs) -> None:
        return None

    event = Event(trigger=trigger, action=action, name="test")
    assert event.action is action


# this was also allowed because some actions functions already return bool, they need to be updated
# then this test can be removed and the check for bool return type can be removed from the events.py file
def test_verify_action_accepts_bool_return_type():
    def trigger(**kwargs) -> bool:
        return True

    def action(**kwargs) -> bool:
        return True

    event = Event(trigger=trigger, action=action, name="test")
    assert event.action is action
