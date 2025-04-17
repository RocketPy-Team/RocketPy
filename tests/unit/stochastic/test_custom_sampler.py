from rocketpy.environment.environment import Environment


def test_create_object(stochastic_environment_custom_sampler):
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
    obj = stochastic_environment_custom_sampler.create_object()
    assert isinstance(obj, Environment)
