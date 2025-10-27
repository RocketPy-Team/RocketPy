import json
import os
from unittest.mock import patch

import numpy as np
import pytest

from rocketpy._encoders import RocketPyDecoder, RocketPyEncoder
from rocketpy.tools import from_hex_decode


# pylint: disable=unused-argument
@patch("matplotlib.pyplot.show")
@pytest.mark.parametrize(
    ["flight_name", "include_outputs"],
    [
        ("flight_calisto", False),
        ("flight_calisto", True),
        ("flight_calisto_robust", True),
        ("flight_calisto_liquid_modded", False),
        ("flight_calisto_hybrid_modded", False),
        ("flight_calisto_air_brakes", False),
        ("flight_calisto_with_sensors", False),
    ],
)
def test_flight_save_load_no_resimulate(
    mock_show, flight_name, include_outputs, request
):
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
        json.dump(
            flight_to_save,
            f,
            cls=RocketPyEncoder,
            indent=2,
            include_outputs=include_outputs,
        )

    with open("flight.json", "r") as f:
        flight_loaded = json.load(f, cls=RocketPyDecoder, resimulate=False)

    assert np.isclose(flight_to_save.t_initial, flight_loaded.t_initial)
    assert np.isclose(flight_to_save.out_of_rail_time, flight_loaded.out_of_rail_time)
    assert np.isclose(flight_to_save.apogee_time, flight_loaded.apogee_time)

    # Higher tolerance due to random parachute trigger
    assert np.isclose(flight_to_save.t_final, flight_loaded.t_final, rtol=1e-3)

    flight_loaded.all_info()

    os.remove("flight.json")


@pytest.mark.slow
@pytest.mark.parametrize(
    ["flight_name", "include_outputs"],
    [
        ("flight_calisto", False),
        ("flight_calisto", True),
        ("flight_calisto_robust", True),
        ("flight_calisto_liquid_modded", False),
        ("flight_calisto_hybrid_modded", False),
        ("flight_calisto_air_brakes", False),
        ("flight_calisto_with_sensors", False),
    ],
)
def test_flight_save_load_resimulate(flight_name, include_outputs, request):
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
        json.dump(
            flight_to_save,
            f,
            cls=RocketPyEncoder,
            indent=2,
            include_outputs=include_outputs,
        )

    with open("flight.json", "r") as f:
        flight_loaded = json.load(f, cls=RocketPyDecoder, resimulate=True)

    assert np.isclose(flight_to_save.t_initial, flight_loaded.t_initial)
    assert np.isclose(flight_to_save.out_of_rail_time, flight_loaded.out_of_rail_time)
    assert np.isclose(flight_to_save.apogee_time, flight_loaded.apogee_time)

    # Higher tolerance due to random parachute trigger
    assert np.isclose(flight_to_save.t_final, flight_loaded.t_final, rtol=5e-3)

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

    test_heights = np.linspace(0, 10000, 100)

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


@pytest.mark.parametrize(
    "motor_name", ["cesaroni_m1670", "liquid_motor", "hybrid_motor", "generic_motor"]
)
def test_motor_encoder(motor_name, request):
    """Test encoding a ``rocketpy.Motor``.

    Parameters
    ----------
    motor_name : str
        Name of the motor fixture to encode.
    request : pytest.FixtureRequest
        Pytest request object.
    """
    motor_to_encode = request.getfixturevalue(motor_name)

    json_encoded = json.dumps(motor_to_encode, cls=RocketPyEncoder)

    motor_loaded = json.loads(json_encoded, cls=RocketPyDecoder)

    sample_times = np.linspace(*motor_to_encode.burn_time, 100)

    assert np.allclose(
        motor_to_encode.thrust(sample_times), motor_loaded.thrust(sample_times)
    )
    assert np.allclose(
        motor_to_encode.total_mass(sample_times), motor_loaded.total_mass(sample_times)
    )
    assert np.allclose(
        motor_to_encode.center_of_mass(sample_times),
        motor_loaded.center_of_mass(sample_times),
    )
    assert np.allclose(
        motor_to_encode.I_11(sample_times), motor_loaded.I_11(sample_times)
    )


@pytest.mark.parametrize(
    "rocket_name", ["calisto_robust", "calisto_liquid_modded", "calisto_hybrid_modded"]
)
def test_rocket_encoder(rocket_name, request):
    """Test encoding a ``rocketpy.Rocket``.

    Parameters
    ----------
    rocket_name : str
        Name of the rocket fixture to encode.
    request : pytest.FixtureRequest
        Pytest request object.
    """
    rocket_to_encode = request.getfixturevalue(rocket_name)

    json_encoded = json.dumps(rocket_to_encode, cls=RocketPyEncoder)

    rocket_loaded = json.loads(json_encoded, cls=RocketPyDecoder)

    sample_times = np.linspace(*rocket_to_encode.motor.burn_time, 100)

    assert np.allclose(
        rocket_to_encode.evaluate_total_mass()(sample_times),
        rocket_loaded.evaluate_total_mass()(sample_times),
    )
    assert np.allclose(
        rocket_to_encode.static_margin(sample_times),
        rocket_loaded.static_margin(sample_times),
    )


@pytest.mark.parametrize(
    "fin_name",
    [
        "calisto_trapezoidal_fins",
        "calisto_trapezoidal_fins_custom_sweep_length",
        "calisto_trapezoidal_fins_custom_sweep_angle",
    ],
)
def test_trapezoidal_fins_encoder(fin_name, request):
    """Test encoding a ``rocketpy.TrapezoidalFins``.

    Parameters
    ----------
    fin_name : str
        Name of the fin fixture to encode.
    request : pytest.FixtureRequest
        Pytest request object.
    """
    fin_to_encode = request.getfixturevalue(fin_name)

    json_encoded = json.dumps(fin_to_encode, cls=RocketPyEncoder)

    fin_loaded = json.loads(json_encoded, cls=RocketPyDecoder)

    assert isinstance(fin_loaded, type(fin_to_encode))
    assert fin_to_encode.n == fin_loaded.n
    assert np.isclose(fin_to_encode.span, fin_loaded.span)
    assert np.isclose(fin_to_encode.root_chord, fin_loaded.root_chord)
    assert np.isclose(fin_to_encode.tip_chord, fin_loaded.tip_chord)
    assert np.isclose(fin_to_encode.rocket_radius, fin_loaded.rocket_radius)
    assert np.isclose(fin_to_encode.sweep_length, fin_loaded.sweep_length)
    if fin_to_encode._sweep_angle is not None and fin_loaded._sweep_angle is not None:
        assert np.isclose(fin_to_encode.sweep_angle, fin_loaded.sweep_angle)


@pytest.mark.parametrize("rocket_name", ["calisto_robust", "calisto_hybrid_modded"])
def test_encoder_discretize(rocket_name, request):
    """Test encoding the total mass of ``rocketpy.Rocket`` with
    discretized encoding.

    Parameters
    ----------
    rocket_name : str
        Name of the rocket fixture to encode.
    request : pytest.FixtureRequest
        Pytest request object.
    """
    rocket_to_encode = request.getfixturevalue(rocket_name)

    json_encoded = json.dumps(
        rocket_to_encode, cls=RocketPyEncoder, discretize=True, include_outputs=True
    )

    mass_loaded = json.loads(
        json.dumps(json.loads(json_encoded)["total_mass"]), cls=RocketPyDecoder
    )

    sample_times = np.linspace(*rocket_to_encode.motor.burn_time, 100)

    np.testing.assert_allclose(
        mass_loaded(sample_times),
        rocket_to_encode.total_mass(sample_times),
        rtol=1e-3,
        atol=1e-1,
    )
    assert isinstance(mass_loaded.source, np.ndarray)


@pytest.mark.parametrize("parachute_name", ["calisto_main_chute"])
def test_encoder_no_pickle(parachute_name, request):
    """Test encoding of a ``rocketpy.Parachute`` disallowing
    pickle usage.
    """
    parachute_to_encode = request.getfixturevalue(parachute_name)

    json_encoded = json.dumps(
        parachute_to_encode,
        cls=RocketPyEncoder,
        allow_pickle=False,
    )

    trigger_loaded = json.loads(json_encoded)["trigger"]

    with pytest.raises(ValueError, match=r"non-hexadecimal number found"):
        from_hex_decode(trigger_loaded)
