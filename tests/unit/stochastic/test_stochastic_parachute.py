import pytest

from rocketpy.stochastic import StochasticParachute
from rocketpy.rocket.parachute import Parachute


def test_stochastic_parachute_create_object(stochastic_main_parachute):
    """Test create object method of StochasticParachute class.

    This test checks if the create_object method of the StochasticParachute
    class creates a StochasticParachute object from the randomly generated
    input arguments.

    Parameters
    ----------
    stochastic_main_parachute : StochasticParachute
        StochasticParachute object to be tested.

    Returns
    -------
    None
    """
    obj = stochastic_main_parachute.create_object()
    assert isinstance(obj, Parachute)


def _at_apogee(pressure, height, state):  # pylint: disable=unused-argument
    """A trigger of the kind `Parachute` and `Flight` already accept.

    Keeps the full signature rather than underscoring the unused two, since the
    signature is the contract being tested."""
    return state[5] < 0


@pytest.mark.parametrize(
    "trigger",
    [[_at_apogee], ["apogee"], [800], [_at_apogee, "apogee", 800]],
    ids=["callable", "apogee", "height", "mixed"],
)
def test_every_documented_trigger_form_is_accepted(calisto_main_chute, trigger):
    """The docstring promises callables, "apogee" and numbers. The check read
    `isinstance(member, (str, int, float) or callable(member))`, and a non-empty
    type tuple is truthy, so the `or` short-circuited and callables were
    refused. The two non-callable forms passed throughout, which is why it went
    unnoticed."""
    StochasticParachute(calisto_main_chute, trigger=trigger)


@pytest.mark.parametrize("trigger", [_at_apogee, "apogee", 800, [None], [{}]], ids=str)
def test_a_trigger_that_is_not_a_list_of_those_is_still_refused(
    calisto_main_chute, trigger
):
    """The control. Moving the `or` must not turn the check into one that
    accepts anything: a bare callable is not a list, and None is none of the
    three."""
    with pytest.raises(AssertionError, match="must be a list"):
        StochasticParachute(calisto_main_chute, trigger=trigger)
