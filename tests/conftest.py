import datetime

import numericalunits
import pytest

from rocketpy import (
    Dispersion,
    Environment,
    EnvironmentAnalysis,
    Flight,
    Function,
    Rocket,
    SolidMotor,
)
from rocketpy.monte_carlo import (
    McEnvironment,
    McFlight,
    McNoseCone,
    McParachute,
    McRailButtons,
    McRocket,
    McSolidMotor,
    McTail,
    McTrapezoidalFins,
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
        burnOutTime=3.9,
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
        burnOutTime=3.9,
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


@pytest.fixture
def mc_env(example_env):
    """Create an object to be used as the McEnvironment fixture in the tests.

    Parameters
    ----------
    example_env : Environment
        An object of the Environment class

    Returns
    -------
    McEnvironment
        An object of the McEnvironment class
    """

    # modify the environment to have a non-zero wind
    example_env.windX = 3
    example_env.windY = 4

    return McEnvironment(
        environment=example_env,
        railLength=0.0005,
        windXFactor=(1.0, 0.33, "normal"),
        windYFactor=(1.0, 0.33, "normal"),
    )


@pytest.fixture
def mc_solid_motor(solid_motor):
    """Create an object to be used as the McSolidMotor fixture in the tests.

    Parameters
    ----------
    solid_motor : SolidMotor
        An object of the SolidMotor class

    Returns
    -------
    McSolidMotor
        An object of the McSolidMotor class
    """
    return McSolidMotor(
        solidMotor=solid_motor,
        burnOutTime=(3.9, 0.5),
        grainsCenterOfMassPosition=0.001 / 10,
        grainDensity=50,
        grainSeparation=1 / 1000,
        grainInitialHeight=1 / 1000,
        grainInitialInnerRadius=0.1 / 1000,
        grainOuterRadius=0,
        totalImpulse=(3000, 30, "normal"),
        nozzlePosition=0.001,
    )


@pytest.fixture
def mc_rocket(rocket, mc_solid_motor):
    """Create an object to be used as the McRocket fixture in the tests.

    Parameters
    ----------
    example_rocket : Rocket
        An object of the Rocket class

    Returns
    -------
    McRocket
        An object of the McRocket class
    """

    mc_rocket = McRocket(
        rocket=rocket,
        radius=0.001,
        mass=0.001,
        inertiaI=0.0300,
        inertiaZ=0.0001,
        powerOffDragFactor=(1, 0.033),
        powerOnDragFactor=(1, 0.033),
    )

    nose_cone = rocket.addNose(
        length=0.55829, kind="vonKarman", position=0.71971 + 0.558291
    )

    mc_nose_cone = McNoseCone(
        nosecone=nose_cone,
        length=0.001,
    )

    fin_set = rocket.addTrapezoidalFins(
        4, span=0.100, rootChord=0.120, tipChord=0.040, position=-1.04956
    )

    mc_fin_set = McTrapezoidalFins(
        trapezoidalFins=fin_set,
        rootChord=0.0005,
        tipChord=0.0005,
        span=0.0005,
    )

    tail = rocket.addTail(
        topRadius=0.0635, bottomRadius=0.0435, length=0.060, position=-1.194656
    )

    mc_tail = McTail(tail=tail, bottomRadius=0.0005, length=0.001)

    def mainTrigger(p, y):
        # p = pressure
        # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
        # activate main when vz < 0 m/s and z < 800 m.
        return True if y[5] < 0 and y[2] < 800 else False

    main_chute = rocket.addParachute(
        "Main",
        CdS=10.0,
        trigger=mainTrigger,
        samplingRate=105,
        lag=1.5,
        noise=(0, 8.3, 0.5),
    )

    mc_main = McParachute(
        parachute=main_chute,
        CdS=0.07,
        lag=0.3,
    )

    def drogueTrigger(p, y):
        # p = pressure
        # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
        # activate drogue when vz < 0 m/s.
        return True if y[5] < 0 else False

    drogue_chute = rocket.addParachute(
        "Drogue",
        CdS=1.0,
        trigger=drogueTrigger,
        samplingRate=105,
        lag=1.5,
        noise=(0, 8.3, 0.5),
    )

    mc_drogue = McParachute(
        parachute=drogue_chute,
        CdS=0.07,
    )
    mc_rocket.addMotor(mc_solid_motor, position=0.001)
    mc_rocket.addNose(mc_nose_cone, position=(1.134, 0.001))
    mc_rocket.addTrapezoidalFins(mc_fin_set, position=(0.001, "normal"))
    mc_rocket.addTail(mc_tail, position=(-1.194656, 0.001, "normal"))
    # TODO: add rail buttons to be tested
    mc_rocket.addParachute(mc_main)
    mc_rocket.addParachute(mc_drogue)

    return mc_rocket


@pytest.fixture
def mc_flight(flight):
    """Create an object to be used as the McFlight fixture in the tests.

    Parameters
    ----------
    example_flight : Flight
        An object of the Flight class

    Returns
    -------
    McFlight
        An object of the McFlight class
    """
    return McFlight(
        flight=flight,
        inclination=(84.7, 1),
        heading=(53, 2),
    )


@pytest.fixture
def dispersion(mc_env, mc_rocket, mc_flight):
    return Dispersion(
        filename="test_dispersion_class",
        environment=mc_env,
        rocket=mc_rocket,
        flight=mc_flight,
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
