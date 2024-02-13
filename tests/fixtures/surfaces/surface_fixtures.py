import pytest

from rocketpy import NoseCone, RailButtons, Tail, TrapezoidalFins


@pytest.fixture
def calisto_nose_cone():
    """The nose cone of the Calisto rocket.

    Returns
    -------
    rocketpy.NoseCone
        The nose cone of the Calisto rocket.
    """
    return NoseCone(
        length=0.55829,
        kind="vonkarman",
        base_radius=0.0635,
        rocket_radius=0.0635,
        name="calisto_nose_cone",
    )


@pytest.fixture
def calisto_tail():
    """The boat tail of the Calisto rocket.

    Returns
    -------
    rocketpy.Tail
        The boat tail of the Calisto rocket.
    """
    return Tail(
        top_radius=0.0635,
        bottom_radius=0.0435,
        length=0.060,
        rocket_radius=0.0635,
        name="calisto_tail",
    )


@pytest.fixture
def calisto_trapezoidal_fins():
    """The trapezoidal fins of the Calisto rocket.

    Returns
    -------
    rocketpy.TrapezoidalFins
        The trapezoidal fins of the Calisto rocket.
    """
    return TrapezoidalFins(
        n=4,
        span=0.100,
        root_chord=0.120,
        tip_chord=0.040,
        rocket_radius=0.0635,
        name="calisto_trapezoidal_fins",
        cant_angle=0,
        sweep_length=None,
        sweep_angle=None,
        airfoil=None,
    )


@pytest.fixture
def calisto_rail_buttons():
    """The rail buttons of the Calisto rocket.

    Returns
    -------
    rocketpy.RailButtons
        The rail buttons of the Calisto rocket.
    """
    return RailButtons(
        buttons_distance=0.7,
        angular_position=45,
        name="Rail Buttons",
    )

