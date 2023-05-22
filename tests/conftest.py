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
        thrustSource="data/motors/Cesaroni_M1670.eng",
        dry_mass=10,
        burnOut=3.9,
        grainsCenterOfMassPosition=0.39796,
        grainNumber=5,
        grainSeparation=5 / 1000,
        grainDensity=1815,
        grainOuterRadius=33 / 1000,
        grainInitialInnerRadius=15 / 1000,
        grainInitialHeight=120 / 1000,
        nozzleRadius=33 / 1000,
        nozzlePosition=0,
        throatRadius=11 / 1000,
        reshapeThrustCurve=False,
        interpolationMethod="linear",
        coordinateSystemOrientation="nozzleToCombustionChamber",
    )
    return example_motor


@pytest.fixture
def rocket(solid_motor):
    example_rocket = Rocket(
        radius=127 / 2000,
        mass=19.197 - 2.956,
        inertia=(6.60, 6.60, 0.0351),
        powerOffDrag="data/calisto/powerOffDragCurve.csv",
        powerOnDrag="data/calisto/powerOnDragCurve.csv",
        centerOfDryMassPosition=0,
        coordinateSystemOrientation="tailToNose",
    )
    example_rocket.addMotor(solid_motor, position=-1.255)
    return example_rocket


@pytest.fixture
def flight(rocket, example_env):
    rocket.setRailButtons([0.2, -0.5])

    NoseCone = rocket.addNose(
        length=0.55829, kind="vonKarman", position=1.278, name="NoseCone"
    )
    FinSet = rocket.addTrapezoidalFins(
        4, span=0.100, rootChord=0.120, tipChord=0.040, position=-1.04956
    )
    Tail = rocket.addTail(
        topRadius=0.0635, bottomRadius=0.0435, length=0.060, position=-1.194656
    )

    flight = Flight(
        environment=example_env,
        rocket=rocket,
        inclination=85,
        heading=90,
        terminateOnApogee=True,
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
        thrustSource="data/motors/Cesaroni_M1670.eng",
        burnOut=3.9,
        dry_mass=10,
        grainNumber=5,
        grainSeparation=5 / 1000 * m,
        grainDensity=1815 * (kg / m**3),
        grainOuterRadius=33 / 1000 * m,
        grainInitialInnerRadius=15 / 1000 * m,
        grainInitialHeight=120 / 1000 * m,
        nozzleRadius=33 / 1000 * m,
        throatRadius=11 / 1000 * m,
        interpolationMethod="linear",
        grainsCenterOfMassPosition=0.39796 * m,
        nozzlePosition=0 * m,
        coordinateSystemOrientation="nozzleToCombustionChamber",
    )
    return example_motor


@pytest.fixture
def dimensionless_rocket(kg, m, dimensionless_solid_motor):
    example_rocket = Rocket(
        radius=127 / 2000 * m,
        mass=(19.197 - 2.956) * kg,
        inertia=(6.60 * (kg * m**2), 6.60 * (kg * m**2), 0.0351 * (kg * m**2)),
        powerOffDrag="data/calisto/powerOffDragCurve.csv",
        powerOnDrag="data/calisto/powerOnDragCurve.csv",
        centerOfDryMassPosition=0 * m,
        coordinateSystemOrientation="tailToNose",
    )
    example_rocket.addMotor(dimensionless_solid_motor, position=-1.255 * m)
    return example_rocket


@pytest.fixture
def example_env():
    Env = Environment(railLength=5, datum="WGS84")
    return Env


@pytest.fixture
def example_env_robust():
    Env = Environment(
        railLength=5,
        latitude=32.990254,
        longitude=-106.974998,
        elevation=1400,
        datum="WGS84",
    )
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    Env.setDate(
        (tomorrow.year, tomorrow.month, tomorrow.day, 12)
    )  # Hour given in UTC time
    return Env


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
        surfaceDataFile="./data/weather/EuroC_single_level_reanalysis_2002_2021.nc",
        pressureLevelDataFile="./data/weather/EuroC_pressure_levels_reanalysis_2001-2021.nc",
        timezone=None,
        unit_system="metric",
        forecast_date=None,
        forecast_args=None,
        maxExpectedAltitude=None,
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
        extrapolation="linear",
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
