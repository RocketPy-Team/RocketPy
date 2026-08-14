import inspect

import numpy as np
import pytest

from rocketpy.rocket.parachute import Parachute
from rocketpy.stochastic import StochasticParachute


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


@pytest.mark.parametrize(
    "trigger",
    [
        _at_apogee,
        "apogee",
        800,
        (800,),
        [],
        [None],
        [{}],
        ["banana"],
        [True],
        [_at_apogee, None],
    ],
    ids=str,
)
def test_a_trigger_that_is_not_a_list_of_those_is_refused(calisto_main_chute, trigger):
    """The control, and four that the check used to wave through.

    `Parachute` refuses "banana" with a ValueError, so accepting it here only
    moved the failure to create time. `True` is worse: it is an `int`, so it
    was taken as a height of one metre. An empty list passed because `all([])`
    is True. And the docstring's tuple form was never implemented.
    """
    with pytest.raises(AssertionError, match="must be a non-empty list"):
        StochasticParachute(calisto_main_chute, trigger=trigger)


@pytest.mark.parametrize(
    "member",
    [_at_apogee, "apogee", "APOGEE", 800, 800.0, np.float64(800)],
    ids=str,
)
def test_what_this_accepts_is_what_a_parachute_accepts(calisto_main_chute, member):
    """The property, rather than a list of types. Anything this lets through
    has to survive `Parachute`, or the check has only moved the failure."""
    StochasticParachute(calisto_main_chute, trigger=[member])

    Parachute("probe", 10.0, member, 105, 1.5)


@pytest.mark.parametrize("member", [np.int64(800), np.int32(800)], ids=str)
def test_a_numpy_integer_is_refused_here_because_parachute_refuses_it(
    calisto_main_chute, member
):
    """`Parachute` checks `isinstance(trigger, (int, float))`. `numpy.float64`
    subclasses `float` and passes; `numpy.int64` subclasses neither and raises.

    So this check matches that one rather than `numbers.Real`, which would be
    the wider and more natural spelling but would let these through to fail at
    create time. The asymmetry is `Parachute`'s and is worth fixing there.
    """
    with pytest.raises(ValueError, match="Unable to set the trigger"):
        Parachute("probe", 10.0, member, 105, 1.5)

    with pytest.raises(AssertionError, match="must be a non-empty list"):
        StochasticParachute(calisto_main_chute, trigger=[member])


def test_the_check_is_not_stripped_by_python_dash_o():
    """`python -O` removes an `assert` outright, and this check is the only
    thing between a bad trigger and a `Parachute` that either refuses it much
    later or reads `True` as a height."""
    source = inspect.getsource(StochasticParachute._validate_trigger)

    assert "raise AssertionError" in source
    assert not any(line.strip().startswith("assert ") for line in source.splitlines())


def test_a_callable_trigger_reaches_the_parachute_and_gets_called(
    calisto_main_chute,
):
    """Constructing the wrapper is not the property that matters. The callable
    has to survive `create_object` and be what `Flight` ends up calling."""
    stochastic = StochasticParachute(calisto_main_chute, trigger=[_at_apogee])
    stochastic._set_stochastic(42)

    built = stochastic.create_object()

    assert built.trigger is _at_apogee
    # 13, matching the state Flight passes: x y z vx vy vz e0 e1 e2 e3 wx wy wz
    descending = [0.0] * 5 + [-5.0] + [0.0] * 7
    ascending = [0.0] * 5 + [5.0] + [0.0] * 7
    assert built.triggerfunc(0.0, 100.0, descending, [], [])
    assert not built.triggerfunc(0.0, 100.0, ascending, [], [])
