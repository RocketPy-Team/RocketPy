import datetime
from unittest.mock import patch

import pytest

from rocketpy import Environment, SolidMotor, Rocket, Flight

import numpy as np


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
    FinSet = test_rocket.addFins(
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
    FinSet = test_rocket.addFins(
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


def test_stability_static_margins():
    # Function to get rocket with any desired static margin easily
    def setup_rocket_with_given_static_margin(rocket, static_margin):
        def compute_static_margin_error_given_distance(
            distanceToCM, static_margin, rocket
        ):
            rocket.aerodynamicSurfaces = []
            rocket.addNose(length=0.5, kind="vonKarman", distanceToCM=1.0)
            rocket.addFins(
                4,
                span=0.100,
                rootChord=0.100,
                tipChord=0.100,
                distanceToCM=distanceToCM,
            )
            return rocket.staticMargin(0) - static_margin

        from scipy import optimize

        sol = optimize.root_scalar(
            compute_static_margin_error_given_distance,
            bracket=[-2.0, 2.0],
            method="brentq",
            args=(static_margin, rocket),
        )

        return rocket

    # Create an environment with ZERO gravity and CONTROLLED wind
    Env = Environment(
        gravity=0, railLength=0, latitude=0, longitude=0, elevation=0  # zero gravity
    )
    Env.setAtmosphericModel(
        type="CustomAtmosphere",
        wind_u=10,  # 10 m/s constant wind velocity in the east direction
        wind_v=0,
        pressure=101325,
        temperature=300,
    )

    # Create a motor with ZERO thrust and ZERO mass
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

    # Create a rocket with ZERO drag and HUGE mass
    DummyRocket = Rocket(
        motor=DummyMotor,
        radius=127 / 2000,
        mass=100e3,
        inertiaI=1,
        inertiaZ=0.0351,
        distanceRocketNozzle=-1.255,
        distanceRocketPropellant=-0.85704,
        powerOffDrag=0,
        powerOnDrag=0,
    )

    DummyRocket.setRailButtons([0.2, -0.5])
    NoseCone = DummyRocket.addNose(
        length=0.55829, kind="vonKarman", distanceToCM=0.71971
    )
    FinSet = DummyRocket.addFins(
        4, span=0.100, rootChord=0.120, tipChord=0.040, distanceToCM=-1.04956
    )
    Tail = DummyRocket.addTail(
        topRadius=0.0635, bottomRadius=0.0435, length=0.060, distanceToCM=-1.194656
    )

    for wind_u, wind_v in [(0, 10), (0, -10), (10, 0), (-10, 0)]:
        Env.setAtmosphericModel(
            type="CustomAtmosphere",
            wind_u=wind_u,
            wind_v=wind_v,
            pressure=101325,
            temperature=300,
        )

        for static_margin, max_time in [
            (-0.1, 2),
            (-0.01, 5),
            (0, 5),
            (0.01, 20),
            (0.1, 20),
            (1.0, 20),
        ]:
            DummyRocket = setup_rocket_with_given_static_margin(
                DummyRocket, static_margin
            )

            # Simulate
            TestFlight = Flight(
                rocket=DummyRocket,
                environment=Env,
                inclination=90,
                heading=0,
                initialSolution=[
                    0,
                    0,
                    0,
                    100,
                    0,
                    0,
                    100,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0,
                    0,
                    0,
                ],  # Start at 100 m of altitude and 100 m/s of vertical velocity
                maxTime=max_time,
                maxTimeStep=1e-2,
                verbose=False,
            )
            TestFlight.postProcess()

            if wind_u == 0 and wind_v > 0:
                assert (
                    (
                        static_margin > 0
                        and np.max(TestFlight.M1.source[:, 1])
                        * np.min(TestFlight.M1.source[:, 1])
                        < 0
                    )
                    or (static_margin < 0 and np.all(TestFlight.M1.source[:, 1] <= 0))
                    or (static_margin == 0)
                )
            elif wind_u == 0 and wind_v < 0:
                assert (
                    (
                        static_margin > 0
                        and np.max(TestFlight.M1.source[:, 1])
                        * np.min(TestFlight.M1.source[:, 1])
                        < 0
                    )
                    or (static_margin < 0 and np.all(TestFlight.M1.source[:, 1] >= 0))
                    or (static_margin == 0)
                )
            elif wind_u > 0 and wind_v == 0:
                assert (
                    (
                        static_margin > 0
                        and np.max(TestFlight.M2.source[:, 1])
                        * np.min(TestFlight.M2.source[:, 1])
                        < 0
                    )
                    or (static_margin < 0 and np.all(TestFlight.M2.source[:, 1] >= 0))
                    or (static_margin == 0)
                )
            elif wind_u < 0 and wind_v == 0:
                assert (
                    (
                        static_margin > 0
                        and np.max(TestFlight.M2.source[:, 1])
                        * np.min(TestFlight.M2.source[:, 1])
                        < 0
                    )
                    or (static_margin < 0 and np.all(TestFlight.M2.source[:, 1] <= 0))
                    or (static_margin == 0)
                )
