import numpy as np
import pytest

from rocketpy.environment.environment import Environment
from rocketpy.stochastic import StochasticEnvironment


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


def _two_member_ensemble(first=10.0, second=30.0):
    """An Environment with two ensemble members whose winds differ.

    Built here rather than read from a NetCDF file so the two winds are known
    exactly and a factor applied to the wrong one is visible in the result.
    """
    levels = np.array([100000.0, 90000.0, 80000.0])
    height = np.array([0.0, 1000.0, 2000.0])
    temperature = np.array([288.0, 282.0, 275.0])
    winds = (first, second)

    environment = Environment()
    environment.set_atmospheric_model(type="custom_atmosphere", wind_u=0, wind_v=0)
    environment.atmospheric_model_type = "Ensemble"
    environment.num_ensemble_members = 2
    environment.level_ensemble = levels
    environment.height_ensemble = np.tile(height, (2, 1))
    environment.temperature_ensemble = np.tile(temperature, (2, 1))
    environment.wind_u_ensemble = np.array([np.full(3, wind) for wind in winds])
    environment.wind_v_ensemble = np.zeros((2, 3))
    environment.wind_speed_ensemble = np.array([np.full(3, wind) for wind in winds])
    environment.wind_heading_ensemble = np.full((2, 3), 90.0)
    environment.wind_direction_ensemble = np.full((2, 3), 270.0)
    environment.ensemble_member = 0
    environment.select_ensemble_member(0)
    return environment


def test_create_object_scales_the_wind_of_the_member_it_selected():
    """A wind factor must multiply the selected member's own wind.

    ``select_ensemble_member`` rebuilds the wind from that member's profile, so
    a factor applied before it used to be discarded and the run flew the raw
    member wind while the input record still reported the factor.
    """
    environment = _two_member_ensemble(first=10.0, second=30.0)
    stochastic = StochasticEnvironment(
        environment=environment,
        ensemble_member=[1],
        wind_velocity_x_factor=(2.0, 0),
    )
    stochastic._set_stochastic(7)

    wind = float(stochastic.create_object().wind_velocity_x(500))

    assert wind == pytest.approx(60.0, rel=1e-9)
    assert wind != pytest.approx(30.0, rel=1e-9)  # factor dropped
    assert wind != pytest.approx(20.0, rel=1e-9)  # member 0's cached wind


def test_create_object_does_not_compound_the_factor_across_calls():
    """Each call scales the member's profile once, not the previous result."""
    environment = _two_member_ensemble(first=10.0, second=30.0)
    stochastic = StochasticEnvironment(
        environment=environment,
        ensemble_member=[1],
        wind_velocity_x_factor=(2.0, 0),
    )
    stochastic._set_stochastic(7)

    winds = [float(stochastic.create_object().wind_velocity_x(500)) for _ in range(3)]

    assert winds == pytest.approx([60.0, 60.0, 60.0], rel=1e-9)


def test_create_object_without_a_member_still_scales_the_construction_value():
    """Without ensemble members the factor keeps multiplying the original wind."""
    environment = Environment()
    environment.set_atmospheric_model(type="custom_atmosphere", wind_u=10, wind_v=0)
    stochastic = StochasticEnvironment(
        environment=environment, wind_velocity_x_factor=(2.0, 0)
    )
    stochastic._set_stochastic(7)

    winds = [float(stochastic.create_object().wind_velocity_x(500)) for _ in range(3)]

    assert winds == pytest.approx([20.0, 20.0, 20.0], rel=1e-9)
