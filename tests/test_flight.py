import os
import datetime
from unittest.mock import patch

import matplotlib as plt
import numpy as np
import pytest
from scipy import optimize

from rocketpy import Environment, Flight, Function, Rocket, SolidMotor

plt.rcParams.update({"figure.max_open_warning": 0})

# Helper functions


def setup_rocket_with_given_static_margin(rocket, static_margin):
    """Takes any rocket, removes its aerodynamic surfaces and adds a set of
    nose, fins and tail specially designed to have a given static margin.
    The rocket is modified in place.

    Parameters
    ----------
    rocket : Rocket
        Rocket to be modified
    static_margin : float
        Static margin that the given rocket shall have

    Returns
    -------
    rocket : Rocket
        Rocket with the given static margin.
    """

    def compute_static_margin_error_given_distance(distanceToCM, static_margin, rocket):
        rocket.aerodynamicSurfaces = []
        rocket.addNose(length=0.5, kind="vonKarman", distanceToCM=1.0)
        rocket.addTrapezoidalFins(
            4,
            span=0.100,
            rootChord=0.100,
            tipChord=0.100,
            distanceToCM=distanceToCM,
        )
        rocket.addTail(
            topRadius=0.0635,
            bottomRadius=0.0435,
            length=0.060,
            distanceToCM=-1.194656,
        )
        return rocket.staticMargin(0) - static_margin

    sol = optimize.root_scalar(
        compute_static_margin_error_given_distance,
        bracket=[-2.0, 2.0],
        method="brentq",
        args=(static_margin, rocket),
    )

    return rocket


@patch("matplotlib.pyplot.show")
def test_flight(mock_show):
    test_env = Environment(
        railLength=5,
        latitude=32.990254,
        longitude=-106.974998,
        elevation=1400,
        datum="WGS84",
    )
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    test_env.setDate(
        (tomorrow.year, tomorrow.month, tomorrow.day, 12)
    )  # Hour given in UTC time

    test_motor = SolidMotor(
        thrustSource="data/motors/Cesaroni_M1670.eng",
        burnOut=3.9,
        grainNumber=5,
        grainSeparation=5 / 1000,
        grainDensity=1815,
        grainOuterRadius=33 / 1000,
        grainInitialInnerRadius=15 / 1000,
        grainInitialHeight=120 / 1000,
        nozzleRadius=33 / 1000,
        throatRadius=11 / 1000,
        interpolationMethod="linear",
    )

    test_rocket = Rocket(
        motor=test_motor,
        radius=127 / 2000,
        mass=19.197 - 2.956,
        inertiaI=6.60,
        inertiaZ=0.0351,
        distanceRocketNozzle=-1.255,
        distanceRocketPropellant=-0.85704,
        powerOffDrag="data/calisto/powerOffDragCurve.csv",
        powerOnDrag="data/calisto/powerOnDragCurve.csv",
    )

    test_rocket.setRailButtons([0.2, -0.5])

    NoseCone = test_rocket.addNose(
        length=0.55829, kind="vonKarman", distanceToCM=0.71971
    )
    FinSet = test_rocket.addTrapezoidalFins(
        4, span=0.100, rootChord=0.120, tipChord=0.040, distanceToCM=-1.04956
    )
    Tail = test_rocket.addTail(
        topRadius=0.0635, bottomRadius=0.0435, length=0.060, distanceToCM=-1.194656
    )

    def drogueTrigger(p, y):
        # p = pressure
        # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
        # activate drogue when vz < 0 m/s.
        return True if y[5] < 0 else False

    def mainTrigger(p, y):
        # p = pressure
        # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
        # activate main when vz < 0 m/s and z < 800 m.
        return True if y[5] < 0 and y[2] < 800 else False

    Main = test_rocket.addParachute(
        "Main",
        CdS=10.0,
        trigger=mainTrigger,
        samplingRate=105,
        lag=1.5,
        noise=(0, 8.3, 0.5),
    )

    Drogue = test_rocket.addParachute(
        "Drogue",
        CdS=1.0,
        trigger=drogueTrigger,
        samplingRate=105,
        lag=1.5,
        noise=(0, 8.3, 0.5),
    )

    test_flight = Flight(
        rocket=test_rocket, environment=test_env, inclination=85, heading=0
    )

    assert test_flight.allInfo() == None


@patch("matplotlib.pyplot.show")
def test_initial_solution(mock_show):
    test_env = Environment(
        railLength=5,
        latitude=32.990254,
        longitude=-106.974998,
        elevation=1400,
        datum="WGS84",
    )
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    test_env.setDate(
        (tomorrow.year, tomorrow.month, tomorrow.day, 12)
    )  # Hour given in UTC time

    test_motor = SolidMotor(
        thrustSource="data/motors/Cesaroni_M1670.eng",
        burnOut=3.9,
        grainNumber=5,
        grainSeparation=5 / 1000,
        grainDensity=1815,
        grainOuterRadius=33 / 1000,
        grainInitialInnerRadius=15 / 1000,
        grainInitialHeight=120 / 1000,
        nozzleRadius=33 / 1000,
        throatRadius=11 / 1000,
        interpolationMethod="linear",
    )

    test_rocket = Rocket(
        motor=test_motor,
        radius=127 / 2000,
        mass=19.197 - 2.956,
        inertiaI=6.60,
        inertiaZ=0.0351,
        distanceRocketNozzle=-1.255,
        distanceRocketPropellant=-0.85704,
        powerOffDrag="data/calisto/powerOffDragCurve.csv",
        powerOnDrag="data/calisto/powerOnDragCurve.csv",
    )

    test_rocket.setRailButtons([0.2, -0.5])

    NoseCone = test_rocket.addNose(
        length=0.55829, kind="vonKarman", distanceToCM=0.71971
    )
    FinSet = test_rocket.addTrapezoidalFins(
        4, span=0.100, rootChord=0.120, tipChord=0.040, distanceToCM=-1.04956
    )
    Tail = test_rocket.addTail(
        topRadius=0.0635, bottomRadius=0.0435, length=0.060, distanceToCM=-1.194656
    )

    def drogueTrigger(p, y):
        # p = pressure
        # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
        # activate drogue when vz < 0 m/s.
        return True if y[5] < 0 else False

    def mainTrigger(p, y):
        # p = pressure
        # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
        # activate main when vz < 0 m/s and z < 800 m.
        return True if y[5] < 0 and y[2] < 800 else False

    Main = test_rocket.addParachute(
        "Main",
        CdS=10.0,
        trigger=mainTrigger,
        samplingRate=105,
        lag=1.5,
        noise=(0, 8.3, 0.5),
    )

    Drogue = test_rocket.addParachute(
        "Drogue",
        CdS=1.0,
        trigger=drogueTrigger,
        samplingRate=105,
        lag=1.5,
        noise=(0, 8.3, 0.5),
    )

    test_flight = Flight(
        rocket=test_rocket,
        environment=test_env,
        inclination=85,
        heading=0,
        # maxTime=300*60,
        # minTimeStep=0.1,
        # maxTimeStep=10,
        rtol=1e-8,
        atol=1e-6,
        verbose=True,
        initialSolution=[
            0.0,
            0.0,
            0.0,
            1.5e3,
            10,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    )

    assert test_flight.allInfo() == None


@pytest.mark.parametrize("wind_u, wind_v", [(0, 10), (0, -10), (10, 0), (-10, 0)])
@pytest.mark.parametrize(
    "static_margin, max_time",
    [(-0.1, 2), (-0.01, 5), (0, 5), (0.01, 20), (0.1, 20), (1.0, 20)],
)
def test_stability_static_margins(wind_u, wind_v, static_margin, max_time):
    """Test stability margins for a constant velocity flight, 100 m/s, wind a
    lateral wind speed of 10 m/s. Rocket has infinite mass to prevent side motion.
    Check if a restoring moment exists depending on static margins."""

    # Create an environment with ZERO gravity to keep the rocket's speed constant
    Env = Environment(gravity=0, railLength=0, latitude=0, longitude=0, elevation=0)
    Env.setAtmosphericModel(
        type="CustomAtmosphere",
        wind_u=wind_u,
        wind_v=wind_v,
        pressure=101325,
        temperature=300,
    )
    # Make sure that the freestreamMach will always be 0, so that the rocket
    # behaves as the STATIC (freestreamMach=0) margin predicts
    Env.speedOfSound = Function(1e16)

    # Create a motor with ZERO thrust and ZERO mass to keep the rocket's speed constant
    DummyMotor = SolidMotor(
        thrustSource=1e-300,
        burnOut=1e-10,
        grainNumber=5,
        grainSeparation=5 / 1000,
        grainDensity=1e-300,
        grainOuterRadius=33 / 1000,
        grainInitialInnerRadius=15 / 1000,
        grainInitialHeight=120 / 1000,
        nozzleRadius=33 / 1000,
        throatRadius=11 / 1000,
    )

    # Create a rocket with ZERO drag and HUGE mass to keep the rocket's speed constant
    DummyRocket = Rocket(
        motor=DummyMotor,
        radius=127 / 2000,
        mass=1e16,
        inertiaI=1,
        inertiaZ=0.0351,
        distanceRocketNozzle=-1.255,
        distanceRocketPropellant=-0.85704,
        powerOffDrag=0,
        powerOnDrag=0,
    )
    DummyRocket.setRailButtons([0.2, -0.5])
    setup_rocket_with_given_static_margin(DummyRocket, static_margin)

    # Simulate
    init_pos = [0, 0, 100]  # Start at 100 m of altitude
    init_vel = [0, 0, 100]  # Start at 100 m/s
    init_att = [1, 0, 0, 0]  # Inclination of 90 deg and heading of 0 deg
    init_angvel = [0, 0, 0]
    initial_solution = [0] + init_pos + init_vel + init_att + init_angvel
    TestFlight = Flight(
        rocket=DummyRocket,
        environment=Env,
        initialSolution=initial_solution,
        maxTime=max_time,
        maxTimeStep=1e-2,
        verbose=False,
    )
    TestFlight.postProcess(interpolation="linear")

    # Check stability according to static margin
    if wind_u == 0:
        moments = TestFlight.M1.source[:, 1]
        wind_sign = np.sign(wind_v)
    else:  # wind_v == 0
        moments = TestFlight.M2.source[:, 1]
        wind_sign = -np.sign(wind_u)

    assert (
        (static_margin > 0 and np.max(moments) * np.min(moments) < 0)
        or (static_margin < 0 and np.all(moments / wind_sign <= 0))
        or (static_margin == 0 and np.all(np.abs(moments) <= 1e-10))
    )


@patch("matplotlib.pyplot.show")
def test_rolling_flight(mock_show):
    test_env = Environment(
        railLength=5,
        latitude=32.990254,
        longitude=-106.974998,
        elevation=1400,
        datum="WGS84",
    )
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    test_env.setDate(
        (tomorrow.year, tomorrow.month, tomorrow.day, 12)
    )  # Hour given in UTC time
    test_env.setAtmosphericModel(type="StandardAtmosphere")

    test_motor = SolidMotor(
        thrustSource="data/motors/Cesaroni_M1670.eng",
        burnOut=3.9,
        grainNumber=5,
        grainSeparation=5 / 1000,
        grainDensity=1815,
        grainOuterRadius=33 / 1000,
        grainInitialInnerRadius=15 / 1000,
        grainInitialHeight=120 / 1000,
        nozzleRadius=33 / 1000,
        throatRadius=11 / 1000,
        interpolationMethod="linear",
    )

    test_rocket = Rocket(
        motor=test_motor,
        radius=127 / 2000,
        mass=19.197 - 2.956,
        inertiaI=6.60,
        inertiaZ=0.0351,
        distanceRocketNozzle=-1.255,
        distanceRocketPropellant=-0.85704,
        powerOffDrag="data/calisto/powerOffDragCurve.csv",
        powerOnDrag="data/calisto/powerOnDragCurve.csv",
    )

    test_rocket.setRailButtons([0.2, -0.5])

    NoseCone = test_rocket.addNose(
        length=0.55829, kind="vonKarman", distanceToCM=0.71971
    )
    FinSet = test_rocket.addTrapezoidalFins(
        4,
        span=0.100,
        rootChord=0.120,
        tipChord=0.040,
        distanceToCM=-1.04956,
        cantAngle=0.5,
    )
    Tail = test_rocket.addTail(
        topRadius=0.0635, bottomRadius=0.0435, length=0.060, distanceToCM=-1.194656
    )

    def drogueTrigger(p, y):
        # p = pressure
        # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
        # activate drogue when vz < 0 m/s.
        return True if y[5] < 0 else False

    def mainTrigger(p, y):
        # p = pressure
        # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
        # activate main when vz < 0 m/s and z < 800 m.
        return True if y[5] < 0 and y[2] < 800 else False

    Main = test_rocket.addParachute(
        "Main",
        CdS=10.0,
        trigger=mainTrigger,
        samplingRate=105,
        lag=1.5,
        noise=(0, 8.3, 0.5),
    )

    Drogue = test_rocket.addParachute(
        "Drogue",
        CdS=1.0,
        trigger=drogueTrigger,
        samplingRate=105,
        lag=1.5,
        noise=(0, 8.3, 0.5),
    )

    test_flight = Flight(
        rocket=test_rocket, environment=test_env, inclination=85, heading=0
    )

    assert test_flight.allInfo() == None


def test_export_data():
    "Tests weather the method Flight.exportData is working as intended"

    test_env = Environment(
        railLength=5,
        latitude=32.990254,
        longitude=-106.974998,
        elevation=1400,
        datum="WGS84",
    )

    test_motor = SolidMotor(
        thrustSource=1000,
        burnOut=3.9,
        grainNumber=5,
        grainSeparation=5 / 1000,
        grainDensity=1815,
        grainOuterRadius=33 / 1000,
        grainInitialInnerRadius=15 / 1000,
        grainInitialHeight=120 / 1000,
        nozzleRadius=33 / 1000,
        throatRadius=11 / 1000,
        interpolationMethod="linear",
    )

    test_rocket = Rocket(
        motor=test_motor,
        radius=127 / 2000,
        mass=19.197 - 2.956,
        inertiaI=6.60,
        inertiaZ=0.0351,
        distanceRocketNozzle=-1.255,
        distanceRocketPropellant=-0.85704,
        powerOffDrag=0.5,
        powerOnDrag=0.5,
    )

    test_rocket.setRailButtons([0.2, -0.5])

    NoseCone = test_rocket.addNose(
        length=0.55829, kind="vonKarman", distanceToCM=0.71971
    )
    FinSet = test_rocket.addTrapezoidalFins(
        4, span=0.100, rootChord=0.120, tipChord=0.040, distanceToCM=-1.04956
    )

    test_flight = Flight(
        rocket=test_rocket, environment=test_env, inclination=85, heading=0
    )

    # Basic export
    test_flight.exportData("test_export_data_1.csv")

    # Custom export
    test_flight.exportData(
        "test_export_data_2.csv", "z", "vz", "e1", "w3", "angleOfAttack", timeStep=0.1
    )

    # Load exported files and fixtures and compare them

    test_1 = np.loadtxt("test_export_data_1.csv", delimiter=",")
    test_2 = np.loadtxt("test_export_data_2.csv", delimiter=",")

    # Delete files
    os.remove("test_export_data_1.csv")
    os.remove("test_export_data_2.csv")

    # Check if basic exported content matches data
    assert np.allclose(test_flight.x[:, 0], test_1[:, 0], atol=1e-5) == True
    assert np.allclose(test_flight.x[:, 1], test_1[:, 1], atol=1e-5) == True
    assert np.allclose(test_flight.y[:, 1], test_1[:, 2], atol=1e-5) == True
    assert np.allclose(test_flight.z[:, 1], test_1[:, 3], atol=1e-5) == True
    assert np.allclose(test_flight.vx[:, 1], test_1[:, 4], atol=1e-5) == True
    assert np.allclose(test_flight.vy[:, 1], test_1[:, 5], atol=1e-5) == True
    assert np.allclose(test_flight.vz[:, 1], test_1[:, 6], atol=1e-5) == True
    assert np.allclose(test_flight.e0[:, 1], test_1[:, 7], atol=1e-5) == True
    assert np.allclose(test_flight.e1[:, 1], test_1[:, 8], atol=1e-5) == True
    assert np.allclose(test_flight.e2[:, 1], test_1[:, 9], atol=1e-5) == True
    assert np.allclose(test_flight.e3[:, 1], test_1[:, 10], atol=1e-5) == True
    assert np.allclose(test_flight.w1[:, 1], test_1[:, 11], atol=1e-5) == True
    assert np.allclose(test_flight.w2[:, 1], test_1[:, 12], atol=1e-5) == True
    assert np.allclose(test_flight.w3[:, 1], test_1[:, 13], atol=1e-5) == True

    # Check if custom exported content matches data
    timePoints = np.arange(test_flight.tInitial, test_flight.tFinal, 0.1)
    assert np.allclose(timePoints, test_2[:, 0], atol=1e-5) == True
    assert np.allclose(test_flight.z(timePoints), test_2[:, 1], atol=1e-5) == True
    assert np.allclose(test_flight.vz(timePoints), test_2[:, 2], atol=1e-5) == True
    assert np.allclose(test_flight.e1(timePoints), test_2[:, 3], atol=1e-5) == True
    assert np.allclose(test_flight.w3(timePoints), test_2[:, 4], atol=1e-5) == True
    assert (
        np.allclose(test_flight.angleOfAttack(timePoints), test_2[:, 5], atol=1e-5)
        == True
    )


def test_export_KML():
    "Tests weather the method Flight.exportKML is working as intended"

    test_env = Environment(
        railLength=5,
        latitude=32.990254,
        longitude=-106.974998,
        elevation=1400,
        datum="WGS84",
    )

    test_motor = SolidMotor(
        thrustSource=1000,
        burnOut=1,
        grainNumber=5,
        grainSeparation=5 / 1000,
        grainDensity=1815,
        grainOuterRadius=33 / 1000,
        grainInitialInnerRadius=15 / 1000,
        grainInitialHeight=120 / 1000,
        nozzleRadius=33 / 1000,
        throatRadius=11 / 1000,
        interpolationMethod="linear",
    )

    test_rocket = Rocket(
        motor=test_motor,
        radius=127 / 2000,
        mass=19.197 - 2.956,
        inertiaI=6.60,
        inertiaZ=0.0351,
        distanceRocketNozzle=-1.255,
        distanceRocketPropellant=-0.85704,
        powerOffDrag=0.5,
        powerOnDrag=0.5,
    )

    test_rocket.setRailButtons([0.2, -0.5])

    NoseCone = test_rocket.addNose(
        length=0.55829, kind="vonKarman", distanceToCM=0.71971
    )
    FinSet = test_rocket.addTrapezoidalFins(
        4, span=0.100, rootChord=0.120, tipChord=0.040, distanceToCM=-1.04956
    )

    test_flight = Flight(
        rocket=test_rocket, environment=test_env, inclination=85, heading=0
    )

    # Basic export
    test_flight.exportKML(
        "test_export_data_1.kml", timeStep=None, extrude=True, altitudeMode="absolute"
    )

    # Load exported files and fixtures and compare them
    test_1 = open("test_export_data_1.kml", "r")
    for row in test_1:
        if row[:29] == "                <coordinates>":
            r = row[29:-15]
            r = r.split(",")
            for i, j in enumerate(r):
                r[i] = j.split(" ")
    lon, lat, z, coords = [], [], [], []
    for i in r:
        for j in i:
            coords.append(j)
    for i in range(0, len(coords), 3):
        lon.append(float(coords[i]))
        lat.append(float(coords[i + 1]))
        z.append(float(coords[i + 2]))

    # Delete temporary test file
    test_1.close()
    os.remove("test_export_data_1.kml")

    assert np.allclose(test_flight.latitude[:, 1], lat, atol=1e-3) == True
    assert np.allclose(test_flight.longitude[:, 1], lon, atol=1e-3) == True
    assert np.allclose(test_flight.z[:, 1], z, atol=1e-3) == True


@patch("matplotlib.pyplot.show")
def test_latlon_conversions(mock_show):
    test_env = Environment(
        railLength=5,
        latitude=32.990254,
        longitude=-106.974998,
        elevation=1400,
        datum="WGS84",
    )

    test_motor = SolidMotor(
        thrustSource=1545.218,
        burnOut=3.9,
        grainNumber=5,
        grainSeparation=5 / 1000,
        grainDensity=1815,
        grainOuterRadius=33 / 1000,
        grainInitialInnerRadius=15 / 1000,
        grainInitialHeight=120 / 1000,
    )

    test_rocket = Rocket(
        motor=test_motor,
        radius=127 / 2000,
        mass=19.197 - 2.956,
        inertiaI=6.60,
        inertiaZ=0.0351,
        distanceRocketNozzle=-1.255,
        distanceRocketPropellant=-0.85704,
        powerOffDrag=0.5,
        powerOnDrag=0.5,
    )

    test_rocket.setRailButtons([0.2, -0.5])

    NoseCone = test_rocket.addNose(
        length=0.55829, kind="vonKarman", distanceToCM=0.71971
    )
    FinSet = test_rocket.addTrapezoidalFins(
        4, span=0.100, rootChord=0.120, tipChord=0.040, distanceToCM=-1.04956
    )
    Tail = test_rocket.addTail(
        topRadius=0.0635, bottomRadius=0.0435, length=0.060, distanceToCM=-1.194656
    )

    def drogueTrigger(p, y):
        # p = pressure
        # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
        # activate drogue when vz < 0 m/s.
        return True if y[5] < 0 else False

    def mainTrigger(p, y):
        # p = pressure
        # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
        # activate main when vz < 0 m/s and z < 800 m.
        return True if y[5] < 0 and y[2] < 800 else False

    Main = test_rocket.addParachute(
        "Main",
        CdS=10.0,
        trigger=mainTrigger,
        samplingRate=105,
        lag=1.5,
        noise=(0, 8.3, 0.5),
    )

    Drogue = test_rocket.addParachute(
        "Drogue",
        CdS=1.0,
        trigger=drogueTrigger,
        samplingRate=105,
        lag=1.5,
        noise=(0, 8.3, 0.5),
    )

    test_flight = Flight(
        rocket=test_rocket, environment=test_env, inclination=85, heading=45
    )

    # Check for initial and final lat/lon coordinates based on launch pad coordinates
    test_flight.postProcess()
    assert abs(test_flight.latitude(0)) - abs(test_flight.env.lat) < 1e-6
    assert abs(test_flight.longitude(0)) - abs(test_flight.env.lon) < 1e-6
    assert test_flight.latitude(test_flight.tFinal) > test_flight.env.lat
    assert test_flight.longitude(test_flight.tFinal) > test_flight.env.lon


@patch("matplotlib.pyplot.show")
def test_latlon_conversions2(mock_show):
    "additional tests to capture incorrect behaviors during lat/lon conversions"
    test_motor = SolidMotor(
        thrustSource=1000,
        burnOut=3,
        grainNumber=5,
        grainSeparation=5 / 1000,
        grainDensity=1815,
        grainOuterRadius=33 / 1000,
        grainInitialInnerRadius=15 / 1000,
        grainInitialHeight=120 / 1000,
    )

    test_rocket = Rocket(
        motor=test_motor,
        radius=127 / 2000,
        mass=19.197 - 2.956,
        inertiaI=6.60,
        inertiaZ=0.0351,
        distanceRocketNozzle=-1.255,
        distanceRocketPropellant=-0.85704,
        powerOffDrag=0.5,
        powerOnDrag=0.5,
    )

    test_rocket.setRailButtons([0.2, -0.5])

    NoseCone = test_rocket.addNose(
        length=0.55829, kind="vonKarman", distanceToCM=0.71971
    )
    FinSet = test_rocket.addTrapezoidalFins(
        4, span=0.100, rootChord=0.120, tipChord=0.040, distanceToCM=-1.04956
    )
    Tail = test_rocket.addTail(
        topRadius=0.0635, bottomRadius=0.0435, length=0.060, distanceToCM=-1.194656
    )

    test_env = Environment(
        railLength=5,
        latitude=0,
        longitude=0,
        elevation=1400,
    )

    test_flight = Flight(
        rocket=test_rocket, environment=test_env, inclination=85, heading=0
    )

    test_flight.postProcess()

    assert abs(test_flight.longitude(test_flight.tFinal) - 0) < 1e-12
    assert test_flight.latitude(test_flight.tFinal) > 0
