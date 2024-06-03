from rocketpy.environment.environment import Environment


def test_str(stochastic_environment):
    """Test __str__ method of StochasticEnvironment class.

    This test checks if the __str__ method of the StochasticEnvironment class
    returns a string without raising any exceptions.

    Parameters
    ----------
    stochastic_environment : StochasticEnvironment
        StochasticEnvironment object to be tested.

    Returns
    -------
    None
    """
    assert isinstance(str(stochastic_environment), str)


# def test_validate_ensemble(stochastic_environment):
#     print("Implement this later")


def test_create_object(stochastic_environment):
    """Test create object method of StochasticEnvironment class.

    This test checks if the create_object method of the StochasticEnvironment
    class creates a StochasticEnvironment object from the randomly generated
    input arguments.

    Parameters
    ----------
    stochastic_environment : StochasticEnvironment
        StochasticEnvironment object to be tested.

    Returns
    -------
    None
    """
    obj = stochastic_environment.create_object()
    assert isinstance(obj, Environment)
