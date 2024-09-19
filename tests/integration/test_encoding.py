import json
import os

import numpy as np
import pytest

from rocketpy._encoders import RocketPyDecoder, RocketPyEncoder


@pytest.mark.slow
@pytest.mark.parametrize("flight_name", ["flight_calisto", "flight_calisto_robust"])
def test_flight_save_load(flight_name, request):
    """Test encoding a ``rocketpy.Flight``.

    Parameters
    ----------
    flight_name : str
        Name flight fixture to encode.
    request : pytest.FixtureRequest
        Pytest request object.
    """
    flight_to_save = request.getfixturevalue(flight_name)

    with open("flight.json", "w") as f:
        json.dump(flight_to_save, f, cls=RocketPyEncoder, indent=2)

    with open("flight.json", "r") as f:
        flight_loaded = json.load(f, cls=RocketPyDecoder)

    assert np.isclose(flight_to_save.t_initial, flight_loaded.t_initial)
    assert np.isclose(flight_to_save.out_of_rail_time, flight_loaded.out_of_rail_time)
    assert np.isclose(flight_to_save.apogee_time, flight_loaded.apogee_time)
    assert np.isclose(flight_to_save.t_final, flight_loaded.t_final, rtol=1e-2)

    os.remove("flight.json")


@pytest.mark.parametrize(
    "function_name", ["lambda_quad_func", "spline_interpolated_func"]
)
def test_function_encoder(function_name, request):
    """Test encoding a ``rocketpy.Function``.

    Parameters
    ----------
    function_name : str
        Name of the function to encode.
    request : pytest.FixtureRequest
        Pytest request object.
    """
    function_to_encode = request.getfixturevalue(function_name)

    json_encoded = json.dumps(function_to_encode, cls=RocketPyEncoder)

    function_loaded = json.loads(json_encoded, cls=RocketPyDecoder)

    assert isinstance(function_loaded, type(function_to_encode))
    assert np.isclose(function_to_encode(0), function_loaded(0))


@pytest.mark.parametrize(
    "environment_name", ["example_plain_env", "environment_spaceport_america_2023"]
)
def test_environment_encoder(environment_name, request):
    """Test encoding a ``rocketpy.Environment``.

    Parameters
    ----------
    environment_name : str
        Name of the environment fixture to encode.
    request : pytest.FixtureRequest
        Pytest request object.
    """
    env_to_encode = request.getfixturevalue(environment_name)

    json_encoded = json.dumps(env_to_encode, cls=RocketPyEncoder)

    env_loaded = json.loads(json_encoded, cls=RocketPyDecoder)

    test_heights = np.linspace(0, 10000, 4)

    assert np.isclose(env_to_encode.elevation, env_loaded.elevation)
    assert np.isclose(env_to_encode.latitude, env_loaded.latitude)
    assert np.isclose(env_to_encode.longitude, env_loaded.longitude)
    assert env_to_encode.datum == env_loaded.datum
    assert np.allclose(
        env_to_encode.wind_velocity_x(test_heights),
        env_loaded.wind_velocity_x(test_heights),
    )
    assert np.allclose(
        env_to_encode.wind_velocity_y(test_heights),
        env_loaded.wind_velocity_y(test_heights),
    )
    assert np.allclose(
        env_to_encode.temperature(test_heights), env_loaded.temperature(test_heights)
    )
    assert np.allclose(
        env_to_encode.pressure(test_heights), env_loaded.pressure(test_heights)
    )
    assert np.allclose(
        env_to_encode.density(test_heights), env_loaded.density(test_heights)
    )
