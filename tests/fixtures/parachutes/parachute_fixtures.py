import pytest

from rocketpy import Parachute


@pytest.fixture
def calisto_drogue_parachute_trigger():
    """The trigger for the drogue parachute of the Calisto rocket.

    Returns
    -------
    function
        The trigger for the drogue parachute of the Calisto rocket.
    """

    def drogue_trigger(p, h, y):  # pylint: disable=unused-argument
        # activate drogue when vertical velocity is negative
        return y[5] < 0

    return drogue_trigger


@pytest.fixture
def calisto_main_parachute_trigger():
    """The trigger for the main parachute of the Calisto rocket.

    Returns
    -------
    function
        The trigger for the main parachute of the Calisto rocket.
    """

    def main_trigger(p, h, y):  # pylint: disable=unused-argument
        # activate main when vertical velocity is <0 and altitude is below 800m
        return y[5] < 0 and h < 800

    return main_trigger


@pytest.fixture
def calisto_main_chute(calisto_main_parachute_trigger):
    """The main parachute of the Calisto rocket.

    Parameters
    ----------
    calisto_main_parachute_trigger : function
        The trigger for the main parachute of the Calisto rocket. This is a
        pytest fixture too.

    Returns
    -------
    rocketpy.Parachute
        The main parachute of the Calisto rocket.
    """
    return Parachute(
        name="calisto_main_chute",
        cd_s=10.0,
        trigger=calisto_main_parachute_trigger,
        sampling_rate=105,
        lag=1.5,
        noise=(0, 8.3, 0.5),
    )


@pytest.fixture
def calisto_drogue_chute(calisto_drogue_parachute_trigger):
    """The drogue parachute of the Calisto rocket.

    Parameters
    ----------
    calisto_drogue_parachute_trigger : function
        The trigger for the drogue parachute of the Calisto rocket. This is a
        pytest fixture too.

    Returns
    -------
    rocketpy.Parachute
        The drogue parachute of the Calisto rocket.
    """
    return Parachute(
        name="calisto_drogue_chute",
        cd_s=1.0,
        trigger=calisto_drogue_parachute_trigger,
        sampling_rate=105,
        lag=1.5,
        noise=(0, 8.3, 0.5),
    )
