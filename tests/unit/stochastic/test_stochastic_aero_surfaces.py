from rocketpy import AirBrakes
from rocketpy.rocket.aero_surface import NoseCone, RailButtons, Tail, TrapezoidalFins
from rocketpy.stochastic import StochasticAirBrakes

## NOSE CONE


def test_stochastic_nose_cone_create_object(stochastic_nose_cone):
    """Test create object method of StochasticNoseCone class.

    This test checks if the create_object method of the StochasticNoseCone
    class creates a StochasticNoseCone object from the randomly generated
    input arguments.

    Parameters
    ----------
    stochastic_nose_cone : StochasticNoseCone
        StochasticNoseCone object to be tested.

    Returns
    -------
    None
    """
    obj = stochastic_nose_cone.create_object()
    assert isinstance(obj, NoseCone)


## TRAPEZOIDAL FINS


def test_stochastic_trapezoidal_fins_create_object(stochastic_trapezoidal_fins):
    """Test create object method of StochasticTrapezoidalFins class.

    This test checks if the create_object method of the StochasticTrapezoidalFins
    class creates a StochasticTrapezoidalFins object from the randomly generated
    input arguments.

    Parameters
    ----------
    stochastic_trapezoidal_fins : StochasticTrapezoidalFins
        StochasticTrapezoidalFins object to be tested.

    Returns
    -------
    None
    """
    obj = stochastic_trapezoidal_fins.create_object()
    assert isinstance(obj, TrapezoidalFins)


## TAIL


def test_stochastic_tail_create_object(stochastic_tail):
    """Test create object method of StochasticTail class.

    This test checks if the create_object method of the StochasticTail
    class creates a StochasticTail object from the randomly generated
    input arguments.

    Parameters
    ----------
    stochastic_tail : StochasticTail
        StochasticTail object to be tested.

    Returns
    -------
    None
    """
    obj = stochastic_tail.create_object()
    assert isinstance(obj, Tail)


## RAIL BUTTONS


def test_stochastic_rail_buttons_create_object(stochastic_rail_buttons):
    """Test create object method of StochasticRailButtons class.

    This test checks if the create_object method of the StochasticRailButtons
    class creates a StochasticRailButtons object from the randomly generated
    input arguments.

    Parameters
    ----------
    stochastic_rail_buttons : StochasticRailButtons
        StochasticRailButtons object to be tested.

    Returns
    -------
    None
    """
    obj = stochastic_rail_buttons.create_object()
    assert isinstance(obj, RailButtons)


## AIR BRAKES


def test_stochastic_air_brakes_scale_the_drag_and_keep_the_settings():
    """The drag curve factor reaches the drag flown, and the motor phase the
    air brakes act in and their switch are carried over."""
    brakes = AirBrakes(
        "data/rockets/calisto/air_brakes_cd.csv",
        0.01,
        active_during="power_off",
        active=False,
    )
    obj = StochasticAirBrakes(
        brakes, drag_coefficient_curve_factor=(2.0, 0)
    ).create_object()
    assert obj.cA(0, 0, 0.5, 0, 0, 0, 0, 1.0) == 2 * brakes.cA(
        0, 0, 0.5, 0, 0, 0, 0, 1.0
    )
    assert obj.active_during == "power_off" and obj.active is False
