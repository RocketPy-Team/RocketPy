import inspect

import numpy as np
import pytest

from rocketpy.stochastic import StochasticParachute
from rocketpy.rocket.parachute import Parachute
from rocketpy.stochastic.stochastic_parachute import _is_a_trigger


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
    [
        _at_apogee,
        "apogee",
        "APOGEE",
        800,
        800.0,
        np.float64(800),
        np.float32(800),
        np.int64(800),
        np.int32(800),
    ],
    ids=str,
)
def test_what_this_accepts_is_what_a_parachute_accepts(calisto_main_chute, member):
    """The property, rather than a list of types. Anything this lets through
    has to survive `Parachute`, or the check has only moved the failure.

    The NumPy integers used to belong to the test below, refused by both
    because `Parachute` spelled its height check `(int, float)`: `numpy.float64`
    subclasses `float` and passed, `numpy.int64` subclasses neither and raised.
    `Parachute` now reads a height as `numbers.Real`, so they are heights like
    any other and belong here."""
    StochasticParachute(calisto_main_chute, trigger=[member])

    Parachute("probe", 10.0, member, 105, 1.5)


@pytest.mark.parametrize(
    "member",
    [np.bool_(True), complex(800), np.complex64(800), "banana", None, {}],
    ids=str,
)
def test_what_this_refuses_is_what_a_parachute_refuses(calisto_main_chute, member):
    """The other half of the same property, and the half that keeps the widened
    height check honest.

    `numbers.Real` was the wider spelling, but not an unbounded one: neither
    `numpy.bool_` nor the complex types are `Real`, so they still reach the
    error rather than being read as a height. `numpy.bool_` needs no exclusion
    of its own for the same reason -- unlike `bool`, which is an `int` and is
    ruled out by hand."""
    with pytest.raises(ValueError, match="Unable to set the trigger"):
        Parachute("probe", 10.0, member, 105, 1.5)

    with pytest.raises(AssertionError, match="must be a non-empty list"):
        StochasticParachute(calisto_main_chute, trigger=[member])


def test_neither_check_can_drift_from_the_other_again():
    """The two checks were written out separately and disagreed: a
    `numpy.int64` height was refused in `stochastic/` and accepted by the
    `Parachute` that would have been built from it. Nothing failed, because
    each side had a test asserting its own half.

    They now share one predicate, so this asserts the agreement itself over the
    whole boundary rather than a list of types on either side."""
    boundary = [
        800,
        800.0,
        np.float64(800),
        np.float32(800),
        np.int64(800),
        np.int32(800),
        True,
        np.bool_(True),
        complex(800),
        np.complex64(800),
        "apogee",
        "banana",
        None,
    ]

    for member in boundary:
        try:
            Parachute("probe", 10.0, member, 105, 1.5)
        except ValueError:
            parachute_accepts = False
        else:
            parachute_accepts = True

        assert _is_a_trigger(member) is parachute_accepts, (
            f"{member!r}: stochastic/ says {_is_a_trigger(member)}, "
            f"Parachute says {parachute_accepts}"
        )


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
