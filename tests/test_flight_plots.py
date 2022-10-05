from unittest.mock import patch

import matplotlib as plt
import numpy as np
import pytest
from rocketpy import Environment, Flight, Rocket, SolidMotor
from rocketpy.plots import flight_plots

plt.rcParams.update({"figure.max_open_warning": 0})


@patch("matplotlib.pyplot.show")
def test_flight_plots(mock_show):

    test_env = Environment(railLength=5)
    test_env.setAtmosphericModel(type="CustomAtmosphere", wind_u=0.3, wind_v=-1)

    test_motor = SolidMotor(
        thrustSource=2000,
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
        powerOffDrag=0.6,
        powerOnDrag=0.6,
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

    test_flight1 = Flight(
        rocket=test_rocket, environment=test_env, inclination=90, heading=30
    )
    test_flight2 = Flight(
        rocket=test_rocket, environment=test_env, inclination=85, heading=0
    )
    test_flight3 = Flight(
        rocket=test_rocket, environment=test_env, inclination=80, heading=60
    )

    flight_plotter = flight_plots.flight_plots(
        [test_flight1, test_flight2, test_flight3]
    )

    assert flight_plotter.allInfo(mode="basic") == None
    assert flight_plotter.allInfo(mode="compare") == None
