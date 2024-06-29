from rocketpy.rocket.aero_surface import NoseCone, RailButtons, Tail, TrapezoidalFins

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
