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
