import json

import pytest

from rocketpy._encoders import RocketPyEncoder

# TODO: this tests should be improved with better validation and decoding


@pytest.mark.parametrize("flight_name", ["flight_calisto", "flight_calisto_robust"])
def test_encode_flight(flight_name, request):
    """Test encoding a ``rocketpy.Flight``.

    Parameters
    ----------
    flight_name : str
        Name flight fixture to encode.
    request : pytest.FixtureRequest
        Pytest request object.
    """
    flight = request.getfixturevalue(flight_name)

    json_encoded = json.dumps(flight, cls=RocketPyEncoder)

    flight_dict = json.loads(json_encoded)

    assert json_encoded is not None
    assert flight_dict is not None


@pytest.mark.parametrize(
    "function_name", ["lambda_quad_func", "spline_interpolated_func"]
)
def test_encode_function(function_name, request):
    """Test encoding a ``rocketpy.Function``.

    Parameters
    ----------
    function_name : str
        Name of the function to encode.
    request : pytest.FixtureRequest
        Pytest request object.
    """
    function = request.getfixturevalue(function_name)

    json_encoded = json.dumps(function, cls=RocketPyEncoder)

    function_dict = json.loads(json_encoded)

    assert json_encoded is not None
    assert function_dict is not None
