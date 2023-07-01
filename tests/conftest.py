import datetime

import numericalunits
import pytest

from rocketpy import (
    Environment,
    EnvironmentAnalysis,
    Function,
    Rocket,
    SolidMotor,
    Flight,
)


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


@pytest.fixture
def solid_motor():
    example_motor = SolidMotor(
        thrust_source="data/motors/Cesaroni_M1670.eng",
        burn_time=3.9,
        dry_mass=1.815,
        dry_inertia=(0.125, 0.125, 0.002),
        center_of_dry_mass=0.317,
        nozzle_position=0,
        grain_number=5,
        grain_density=1815,
        nozzle_radius=33 / 1000,
        throat_radius=11 / 1000,
        grain_separation=5 / 1000,
        grain_outer_radius=33 / 1000,
        grain_initial_height=120 / 1000,
        grains_center_of_mass_position=0.397,
        grain_initial_inner_radius=15 / 1000,
        interpolation_method="linear",
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )
    return example_motor


@pytest.fixture
def rocket(solid_motor):
    example_rocket = Rocket(
        radius=127 / 2000,
        mass=19.197 - 2.956 - 1.815,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag="data/calisto/powerOffDragCurve.csv",
        power_on_drag="data/calisto/powerOnDragCurve.csv",
        center_of_mass_without_motor=0,
        coordinate_system_orientation="tail_to_nose",
    )
    example_rocket.add_motor(solid_motor, position=-1.255 - 0.1182359460624346)
    return example_rocket


@pytest.fixture
def flight(rocket, example_env):
    rocket.set_rail_buttons(
        upper_button_position=0.2 - 0.1182359460624346,
        lower_button_position=-0.5 - 0.1182359460624346,
        angular_position=45,
    )

    nose_cone = rocket.add_nose(
        length=0.55829,
        kind="von_karman",
        position=1.278 - 0.1182359460624346,
        name="nose_cone",
    )
    fin_set = rocket.add_trapezoidal_fins(
        4,
        span=0.100,
        root_chord=0.120,
        tip_chord=0.040,
        position=-1.04956 - 0.1182359460624346,
    )
    tail = rocket.add_tail(
        top_radius=0.0635,
        bottom_radius=0.0435,
        length=0.060,
        position=-1.194656 - 0.1182359460624346,
    )

    flight = Flight(
        environment=example_env,
        rocket=rocket,
        inclination=85,
        heading=90,
        terminate_on_apogee=True,
    )
    return flight


@pytest.fixture
def m():
    return numericalunits.m


@pytest.fixture
def kg():
    return numericalunits.kg


@pytest.fixture
def dimensionless_solid_motor(kg, m):
    example_motor = SolidMotor(
        thrust_source="data/motors/Cesaroni_M1670.eng",
        burn_time=3.9,
        dry_mass=1.815 * kg,
        dry_inertia=(
            0.125 * (kg * m**2),
            0.125 * (kg * m**2),
            0.002 * (kg * m**2),
        ),
        center_of_dry_mass=0.317 * m,
        grain_number=5,
        grain_separation=5 / 1000 * m,
        grain_density=1815 * (kg / m**3),
        grain_outer_radius=33 / 1000 * m,
        grain_initial_inner_radius=15 / 1000 * m,
        grain_initial_height=120 / 1000 * m,
        nozzle_radius=33 / 1000 * m,
        throat_radius=11 / 1000 * m,
        interpolation_method="linear",
        grains_center_of_mass_position=0.397 * m,
        nozzle_position=0 * m,
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )
    return example_motor


@pytest.fixture
def dimensionless_rocket(kg, m, dimensionless_solid_motor):
    example_rocket = Rocket(
        radius=127 / 2000 * m,
        mass=(19.197 - 2.956 - 1.815) * kg,
        inertia=(6.321 * (kg * m**2), 6.321 * (kg * m**2), 0.034 * (kg * m**2)),
        power_off_drag="data/calisto/powerOffDragCurve.csv",
        power_on_drag="data/calisto/powerOnDragCurve.csv",
        center_of_mass_without_motor=0 * m,
        coordinate_system_orientation="tail_to_nose",
    )
    example_rocket.add_motor(
        dimensionless_solid_motor, position=(-1.255 - 0.1182359460624346) * m
    )
    return example_rocket


@pytest.fixture
def example_env():
    env = Environment(datum="WGS84")
    return env


@pytest.fixture
def example_env_robust():
    env = Environment(
        latitude=32.990254,
        longitude=-106.974998,
        elevation=1400,
        datum="WGS84",
    )
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    env.set_date(
        (tomorrow.year, tomorrow.month, tomorrow.day, 12)
    )  # Hour given in UTC time
    return env


# Create a simple object of the Environment Analysis class
@pytest.fixture
def env_analysis():
    """Create a simple object of the Environment Analysis class to be used in
    the tests. This allows to avoid repeating the same code in all tests.

    Returns
    -------
    EnvironmentAnalysis
        A simple object of the Environment Analysis class
    """
    env_analysis = EnvironmentAnalysis(
        start_date=datetime.datetime(2019, 10, 23),
        end_date=datetime.datetime(2021, 10, 23),
        latitude=39.3897,
        longitude=-8.28896388889,
        start_hour=6,
        end_hour=18,
        surface_data_file="./data/weather/EuroC_single_level_reanalysis_2002_2021.nc",
        pressure_level_data_file="./data/weather/EuroC_pressure_levels_reanalysis_2001-2021.nc",
        timezone=None,
        unit_system="metric",
        forecast_date=None,
        forecast_args=None,
        max_expected_altitude=None,
    )

    return env_analysis


@pytest.fixture
def linear_func():
    """Create a linear function based on a list of points. The function
    represents y = x and may be used on different tests.

    Returns
    -------
    Function
        A linear function representing y = x.
    """
    return Function(
        [[0, 0], [1, 1], [2, 2], [3, 3]],
    )


@pytest.fixture
def linearly_interpolated_func():
    """Create a linearly interpolated function based on a list of points.

    Returns
    -------
    Function
        Piece-wise linearly interpolated, with constant extrapolation
    """
    return Function(
        [[0, 0], [1, 7], [2, -3], [3, -1], [4, 3]],
        interpolation="spline",
        extrapolation="constant",
    )


@pytest.fixture
def spline_interpolated_func():
    """Create a spline interpolated function based on a list of points.

    Returns
    -------
    Function
        Spline interpolated, with natural extrapolation
    """
    return Function(
        [[0, 0], [1, 7], [2, -3], [3, -1], [4, 3]],
        interpolation="spline",
        extrapolation="natural",
    )


@pytest.fixture
def func_from_csv():
    func = Function(
        source="tests/fixtures/airfoils/e473-10e6-degrees.csv",
        inputs=["Scalar"],
        outputs=["Scalar"],
        interpolation="linear",
        extrapolation="natural",
    )
    return func


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
