import base64
import pathlib

import numpy as np
import pytest
import requests
import scipy.integrate

from rocketpy import Function, Motor
from rocketpy.motors.motor import GenericMotor

BURN_TIME = (2, 7)


def thrust_source(t):
    return 2000 - 100 * (t - 2)


CHAMBER_HEIGHT = 0.5
CHAMBER_RADIUS = 0.075
CHAMBER_POSITION = -0.25
PROPELLANT_INITIAL_MASS = 5.0
NOZZLE_POSITION = -0.5
NOZZLE_RADIUS = 0.075
DRY_MASS = 8.0
DRY_INERTIA = (0.2, 0.2, 0.08)


def test_generic_motor_basic_parameters(generic_motor):
    """Tests the GenericMotor class construction parameters.

    Parameters
    ----------
    generic_motor : rocketpy.GenericMotor
        The GenericMotor object to be used in the tests.
    """
    assert generic_motor.burn_time == BURN_TIME
    assert generic_motor.dry_mass == DRY_MASS
    assert (
        generic_motor.dry_I_11,
        generic_motor.dry_I_22,
        generic_motor.dry_I_33,
    ) == DRY_INERTIA
    assert generic_motor.nozzle_position == NOZZLE_POSITION
    assert generic_motor.nozzle_radius == NOZZLE_RADIUS
    assert generic_motor.chamber_position == CHAMBER_POSITION
    assert generic_motor.chamber_radius == CHAMBER_RADIUS
    assert generic_motor.chamber_height == CHAMBER_HEIGHT
    assert generic_motor.propellant_initial_mass == PROPELLANT_INITIAL_MASS


def test_generic_motor_thrust_parameters(generic_motor):
    """Tests the GenericMotor thrust parameters.

    Parameters
    ----------
    generic_motor : rocketpy.GenericMotor
        The GenericMotor object to be used in the tests.
    """
    expected_thrust = np.array(
        [(t, thrust_source(t)) for t in np.linspace(*BURN_TIME, 50)]
    )
    expected_total_impulse = scipy.integrate.trapezoid(
        expected_thrust[:, 1], expected_thrust[:, 0]
    )
    expected_exhaust_velocity = expected_total_impulse / PROPELLANT_INITIAL_MASS
    expected_mass_flow_rate = -1 * expected_thrust[:, 1] / expected_exhaust_velocity

    # Discretize mass flow rate for testing purposes
    mass_flow_rate = generic_motor.total_mass_flow_rate.set_discrete(*BURN_TIME, 50)

    assert generic_motor.thrust.y_array == pytest.approx(expected_thrust[:, 1])
    assert generic_motor.total_impulse == pytest.approx(expected_total_impulse)
    assert generic_motor.exhaust_velocity.average(*BURN_TIME) == pytest.approx(
        expected_exhaust_velocity
    )
    assert mass_flow_rate.y_array == pytest.approx(expected_mass_flow_rate)


def test_generic_motor_center_of_mass(generic_motor):
    """Tests the GenericMotor center of mass.

    Parameters
    ----------
    generic_motor : rocketpy.GenericMotor
        The GenericMotor object to be used in the tests.
    """
    center_of_propellant_mass = -0.25
    center_of_dry_mass = -0.25
    center_of_mass = -0.25

    # Discretize center of mass for testing purposes
    generic_motor.center_of_propellant_mass.set_discrete(*BURN_TIME, 50)
    generic_motor.center_of_mass.set_discrete(*BURN_TIME, 50)

    assert generic_motor.center_of_propellant_mass.y_array == pytest.approx(
        center_of_propellant_mass
    )
    assert generic_motor.center_of_dry_mass_position == pytest.approx(
        center_of_dry_mass
    )
    assert generic_motor.center_of_mass.y_array == pytest.approx(center_of_mass)


def test_generic_motor_inertia(generic_motor):
    """Tests the GenericMotor inertia.

    Parameters
    ----------
    generic_motor : rocketpy.GenericMotor
        The GenericMotor object to be used in the tests.
    """
    # Tests the inertia formulation from the propellant mass
    propellant_mass = generic_motor.propellant_mass.set_discrete(*BURN_TIME, 50).y_array

    propellant_I_11 = propellant_mass * (CHAMBER_RADIUS**2 / 4 + CHAMBER_HEIGHT**2 / 12)
    propellant_I_22 = propellant_I_11
    propellant_I_33 = propellant_mass * (CHAMBER_RADIUS**2 / 2)

    # Centers of mass coincide, so no translation is needed
    I_11 = propellant_I_11 + DRY_INERTIA[0]
    I_22 = propellant_I_22 + DRY_INERTIA[1]
    I_33 = propellant_I_33 + DRY_INERTIA[2]

    # Discretize inertia for testing purposes
    generic_motor.propellant_I_11.set_discrete(*BURN_TIME, 50)
    generic_motor.propellant_I_22.set_discrete(*BURN_TIME, 50)
    generic_motor.propellant_I_33.set_discrete(*BURN_TIME, 50)
    generic_motor.I_11.set_discrete(*BURN_TIME, 50)
    generic_motor.I_22.set_discrete(*BURN_TIME, 50)
    generic_motor.I_33.set_discrete(*BURN_TIME, 50)

    assert generic_motor.propellant_I_11.y_array == pytest.approx(propellant_I_11)
    assert generic_motor.propellant_I_22.y_array == pytest.approx(propellant_I_22)
    assert generic_motor.propellant_I_33.y_array == pytest.approx(propellant_I_33)
    assert generic_motor.I_11.y_array == pytest.approx(I_11)
    assert generic_motor.I_22.y_array == pytest.approx(I_22)
    assert generic_motor.I_33.y_array == pytest.approx(I_33)


def test_load_from_eng_file(generic_motor):
    """Tests the GenericMotor.load_from_eng_file method.

    Parameters
    ----------
    generic_motor : rocketpy.GenericMotor
        The GenericMotor object to be used in the tests.
    """
    # using cesaroni data as example
    burn_time = (0, 3.9)
    dry_mass = 5.231 - 3.101  # 2.130 kg
    propellant_initial_mass = 3.101
    chamber_radius = 75 / 1000
    chamber_height = 757 / 1000
    nozzle_radius = chamber_radius * 0.85  # 85% of chamber radius

    # Parameters from manual testing using the SolidMotor class as a reference
    average_thrust = 1545.218
    total_impulse = 6026.350
    max_thrust = 2200.0
    exhaust_velocity = 1943.357

    # creating motor from .eng file
    generic_motor = generic_motor.load_from_eng_file(
        "data/motors/cesaroni/Cesaroni_M1670.eng"
    )

    # testing relevant parameters
    assert generic_motor.burn_time == burn_time
    assert generic_motor.dry_mass == dry_mass
    assert generic_motor.propellant_initial_mass == propellant_initial_mass
    assert generic_motor.chamber_radius == chamber_radius
    assert generic_motor.chamber_height == chamber_height
    assert generic_motor.chamber_position == 0
    assert generic_motor.average_thrust == pytest.approx(average_thrust)
    assert generic_motor.total_impulse == pytest.approx(total_impulse)
    assert generic_motor.exhaust_velocity.average(*burn_time) == pytest.approx(
        exhaust_velocity
    )
    assert generic_motor.max_thrust == pytest.approx(max_thrust)
    assert generic_motor.nozzle_radius == pytest.approx(nozzle_radius)

    # testing thrust curve
    _, _, points = Motor.import_eng("data/motors/cesaroni/Cesaroni_M1670.eng")
    assert generic_motor.thrust.y_array == pytest.approx(
        Function(points, "Time (s)", "Thrust (N)", "linear", "zero").y_array
    )


def test_load_from_rse_file(generic_motor):
    """Tests the GenericMotor.load_from_rse_file method.

    Parameters
    ----------
    generic_motor : rocketpy.GenericMotor
        The GenericMotor object to be used in the tests.
    """

    # Test the load_from_rse_file method
    generic_motor = generic_motor.load_from_rse_file(
        "data/motors/rse_example/rse_motor_example_file.rse"
    )

    # Check if the engine has been loaded correctly
    assert generic_motor.thrust is not None
    assert generic_motor.dry_mass == 0.0363  # Total mass - propellant mass
    assert generic_motor.propellant_initial_mass == 0.0207
    assert generic_motor.burn_time == (0.0, 2.2)
    assert generic_motor.nozzle_radius == 0.00
    assert generic_motor.chamber_radius == 0.024
    assert generic_motor.chamber_height == 0.07

    # Check the thrust curve values
    thrust_curve = generic_motor.thrust.source
    assert thrust_curve[0][0] == 0.0  # First time point
    assert thrust_curve[0][1] == 0.0  # First thrust point
    assert thrust_curve[-1][0] == 2.2  # Last point of time
    assert thrust_curve[-1][1] == 0.0  # Last thrust point


class MockResponse:
    """Mocked response for requests."""

    def __init__(self, json_data):
        self._json_data = json_data

    def json(self):
        return self._json_data

    def raise_for_status(self):
        return None


def _mock_get(search_results=None, download_results=None):
    """Return a mock_get function with predefined search/download results."""

    def _get(url, **_kwargs):
        if "search.json" in url:
            return MockResponse(search_results or {"results": []})
        if "download.json" in url:
            return MockResponse(download_results or {"results": []})
        raise RuntimeError(f"Unexpected URL: {url}")

    return _get


# Module-level constant for expected motor specs
EXPECTED_MOTOR_SPECS = {
    "burn_time": (0, 3.9),
    "dry_mass": 2.130,
    "propellant_initial_mass": 3.101,
    "chamber_radius": 75 / 1000,
    "chamber_height": 757 / 1000,
    "nozzle_radius": (75 / 1000) * 0.85,
    "average_thrust": 1545.218,
    "total_impulse": 6026.350,
    "max_thrust": 2200.0,
    "exhaust_velocity": 1943.357,
    "chamber_position": 0,
}


def assert_motor_specs(motor):
    specs = EXPECTED_MOTOR_SPECS
    assert motor.burn_time == specs["burn_time"]
    assert motor.dry_mass == specs["dry_mass"]
    assert motor.propellant_initial_mass == specs["propellant_initial_mass"]
    assert motor.chamber_radius == specs["chamber_radius"]
    assert motor.chamber_height == specs["chamber_height"]
    assert motor.chamber_position == specs["chamber_position"]
    assert motor.average_thrust == pytest.approx(specs["average_thrust"])
    assert motor.total_impulse == pytest.approx(specs["total_impulse"])
    assert motor.exhaust_velocity.average(*specs["burn_time"]) == pytest.approx(
        specs["exhaust_velocity"]
    )
    assert motor.max_thrust == pytest.approx(specs["max_thrust"])
    assert motor.nozzle_radius == pytest.approx(specs["nozzle_radius"])


def test_load_from_thrustcurve_api(monkeypatch, generic_motor):
    """Tests GenericMotor.load_from_thrustcurve_api with mocked API."""

    eng_path = "data/motors/cesaroni/Cesaroni_M1670.eng"
    with open(eng_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    search_json = {
        "results": [
            {
                "motorId": "12345",
                "designation": "Cesaroni_M1670",
                "manufacturer": "Cesaroni",
            }
        ]
    }
    download_json = {"results": [{"data": encoded}]}
    monkeypatch.setattr(requests, "get", _mock_get(search_json, download_json))
    monkeypatch.setattr(requests.Session, "get", _mock_get(search_json, download_json))

    motor = type(generic_motor).load_from_thrustcurve_api("M1670")

    assert_motor_specs(motor)

    _, _, points = Motor.import_eng(eng_path)
    assert motor.thrust.y_array == pytest.approx(
        Function(points, "Time (s)", "Thrust (N)", "linear", "zero").y_array
    )

    error_cases = [
        ("No motor found", {"results": []}, None),
        (
            "No .eng file found",
            {
                "results": [
                    {"motorId": "123", "designation": "Fake", "manufacturer": "Test"}
                ]
            },
            {"results": []},
        ),
        (
            "Downloaded .eng data",
            {
                "results": [
                    {"motorId": "123", "designation": "Fake", "manufacturer": "Test"}
                ]
            },
            {"results": [{"data": ""}]},
        ),
    ]

    for msg, search_res, download_res in error_cases:
        monkeypatch.setattr(requests, "get", _mock_get(search_res, download_res))
        monkeypatch.setattr(
            requests.Session, "get", _mock_get(search_res, download_res)
        )
        with pytest.raises(ValueError, match=msg):
            type(generic_motor).load_from_thrustcurve_api("FakeMotor")


def test_thrustcurve_api_cache(monkeypatch, tmp_path):
    """Tests that ThrustCurve API caching works correctly."""

    eng_path = "data/motors/cesaroni/Cesaroni_M1670.eng"
    with open(eng_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    search_json = {"results": [{"motorId": "12345"}]}
    download_json = {"results": [{"data": encoded}]}

    # Patch requests.get to return mocked API responses
    monkeypatch.setattr(requests, "get", _mock_get(search_json, download_json))

    # Patch the module-level CACHE_DIR to use the tmp_path
    monkeypatch.setattr("rocketpy.motors.motor.CACHE_DIR", tmp_path)

    # First call writes to cache
    motor1 = GenericMotor.load_from_thrustcurve_api("M1670")
    cache_file = tmp_path / "M1670.eng.b64"
    assert cache_file.exists()

    # Second call reads from cache; API should not be called
    monkeypatch.setattr(
        requests,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("API should not be called")
        ),
    )
    motor2 = GenericMotor.load_from_thrustcurve_api("M1670")
    assert motor2.thrust.y_array == pytest.approx(motor1.thrust.y_array)

    # Bypass cache with no_cache=True
    monkeypatch.setattr(requests, "get", _mock_get(search_json, download_json))
    motor3 = GenericMotor.load_from_thrustcurve_api("M1670", no_cache=True)
    assert motor3.thrust.y_array == pytest.approx(motor1.thrust.y_array)


def test_thrustcurve_api_cache_robustness(monkeypatch, tmp_path):  # pylint: disable=too-many-statements
    """
    Tests exception handling for cache operations to ensure 100% coverage.
    Simulates OS errors for mkdir, write, and read operations.
    """

    # 1. Setup Mock API to return success
    eng_path = "data/motors/cesaroni/Cesaroni_M1670.eng"
    with open(eng_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    search_json = {"results": [{"motorId": "12345"}]}
    download_json = {"results": [{"data": encoded}]}
    monkeypatch.setattr(requests, "get", _mock_get(search_json, download_json))

    # Point cache to tmp_path so we don't mess with real home
    monkeypatch.setattr("rocketpy.motors.motor.CACHE_DIR", tmp_path)

    # CASE 1: mkdir fails -> should warn and continue (disable caching)
    original_mkdir = pathlib.Path.mkdir

    def mock_mkdir_fail(self, *args, **kwargs):
        if self == tmp_path:
            raise OSError("Simulated mkdir error")
        return original_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "mkdir", mock_mkdir_fail)

    with pytest.warns(UserWarning, match="Could not create cache directory"):
        GenericMotor.load_from_thrustcurve_api("M1670")

    # Reset mkdir logic for next test
    monkeypatch.setattr(pathlib.Path, "mkdir", original_mkdir)

    # CASE 2: write_text fails -> should warn and continue
    original_write = pathlib.Path.write_text

    def mock_write_fail(self, *args, **kwargs):
        if "M1670.eng.b64" in str(self):
            raise OSError("Simulated write error")
        return original_write(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "write_text", mock_write_fail)

    with pytest.warns(RuntimeWarning, match="Could not write to cache file"):
        GenericMotor.load_from_thrustcurve_api("M1670")

    # Reset write logic
    monkeypatch.setattr(pathlib.Path, "write_text", original_write)

    # CASE 3: read_text fails (corrupt file) -> should warn and fetch fresh
    cache_file = tmp_path / "M1670.eng.b64"
    cache_file.write_text("corrupted_data")

    original_read = pathlib.Path.read_text

    def mock_read_fail(self, *args, **kwargs):
        if self == cache_file:
            raise UnicodeDecodeError("utf-8", b"", 0, 1, "bad")
        return original_read(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "read_text", mock_read_fail)

    with pytest.warns(UserWarning, match="Failed to read cached motor file"):
        GenericMotor.load_from_thrustcurve_api("M1670")
