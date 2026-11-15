import warnings

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


def test_stochastic_parachute_noise_warns_once_and_is_not_passed_on(
    calisto_main_chute,
):
    """A noise given to StochasticParachute is ignored. The warning comes once,
    when the stochastic object is created, and not again for every parachute
    created from it."""
    with pytest.warns(FutureWarning, match="StochasticParachute"):
        stochastic_parachute = StochasticParachute(
            parachute=calisto_main_chute,
            noise=[(0, 8.3, 0.5)],
        )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        parachute = stochastic_parachute.create_object()

    assert tuple(parachute.noise) == (0, 0, 0)
    assert "noise" not in stochastic_parachute.last_rnd_dict
