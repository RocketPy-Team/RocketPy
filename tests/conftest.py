import datetime

import numericalunits
import pytest

from rocketpy import Environment, Rocket, SolidMotor


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
        inertiaI=6.60,
        inertiaZ=0.0351,
        powerOffDrag="data/calisto/powerOffDragCurve.csv",
        powerOnDrag="data/calisto/powerOnDragCurve.csv",
        centerOfDryMassPosition=0,
        coordinateSystemOrientation="tailToNose",
    )
    example_rocket.addMotor(solid_motor, position=-1.255)
    return example_rocket


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
        inertiaI=6.60 * (kg * m**2),
        inertiaZ=0.0351 * (kg * m**2),
        powerOffDrag="data/calisto/powerOffDragCurve.csv",
        powerOnDrag="data/calisto/powerOnDragCurve.csv",
        centerOfDryMassPosition=0 * m,
        coordinateSystemOrientation="tailToNose",
    )
    example_rocket.addMotor(dimensionless_solid_motor, position=-1.255 * m)
    return example_rocket


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


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
    )  # Hour given in UT/C time
    return Env
