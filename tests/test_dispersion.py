import datetime
import os
from unittest.mock import patch

from rocketpy import Dispersion, Flight
from tests.conftest import *


@patch("matplotlib.pyplot.show")
def test_dispersion_object_defined(mock_show, solid_motor, example_env_robust):
    "Test dispersion when it is defined by objects and only std are given in the dict"

    # start dispersion
    test_disp = Dispersion(filename="test_obj_defined")

    # setup rocket
    disp_rocket = Rocket(
        radius=127 / 2000,
        mass=19.197 - 2.956,
        inertiaI=6.60,
        inertiaZ=0.0351,
        powerOffDrag="data/calisto/powerOffDragCurve.csv",
        powerOnDrag="data/calisto/powerOnDragCurve.csv",
        centerOfDryMassPosition=0,
        coordinateSystemOrientation="tailToNose",
    )
    disp_rocket.setRailButtons([0.2, -0.5])
    disp_rocket.addMotor(solid_motor, position=-1.255)
    NoseCone = disp_rocket.addNose(
        length=0.55829, kind="vonKarman", position=0.71971 + 0.55829
    )
    FinSet = disp_rocket.addTrapezoidalFins(
        n=4,
        rootChord=0.120,
        tipChord=0.040,
        span=0.100,
        position=-1.04956,
        cantAngle=0,
        radius=None,
        airfoil=None,
    )
    Tail = disp_rocket.addTail(
        topRadius=0.0635, bottomRadius=0.0435, length=0.060, position=-1.194656
    )

    def drogueTrigger(p, y):
        # p = pressure
        # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
        # activate drogue when vz < 0 m/s.
        return True if y[5] < 0 else False

    def mainTrigger(p, y):
        # p = pressure
        # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
        # activate main when vz < 0 m/s and z < 500 + 100 m (+100 due to surface elevation).
        return True if y[5] < 0 and y[2] < 500 + 100 else False

    Main = disp_rocket.addParachute(
        "Main",
        CdS=10.0,
        trigger=mainTrigger,
        samplingRate=105,
        lag=1.5,
        noise=(0, 8.3, 0.5),
    )

    Drogue = disp_rocket.addParachute(
        "Drogue",
        CdS=1.0,
        trigger=drogueTrigger,
        samplingRate=105,
        lag=1.5,
        noise=(0, 8.3, 0.5),
    )

    disp_flight = Flight(disp_rocket, example_env_robust, 85, 90)

    disp_dict = {
        "railLength": 0.001,
        "gravity": 0.1,
        # "date":,
        "latitude": 0.001,
        "longitude": 0.001,
        "elevation": 50,
        "ensembleMember": [0, 1, 2],
        # "thrust": [],
        "burnOutTime": 0.2,
        # "grainNumber":,
        "grainDensity": 0.001,
        "grainOuterRadius": 0.001,
        "grainInitialInnerRadius": 0.001,
        "grainInitialHeight": 0.001,
        "grainSeparation": 0.001,
        "nozzleRadius": 0.001,
        "throatRadius": 0.001,
        "totalImpulse": 0.033 * solid_motor.totalImpulse,
        "grainsCenterOfMassPosition": 0.001,
        "nozzlePosition": 0.001,
        "mass": 0.001,
        "inertiaI": 0.001,
        "inertiaZ": 0.001,
        "radius": 0.001,
        # "powerOffDrag":,
        # "powerOnDrag":,
        "powerOffDragFactor": (1, 0.033),
        "powerOnDragFactor": (1, 0.033),
        "centerOfDryMassPosition": 0.001,
        "motorPosition": 0.001,
        "nose_NoseCone_length": 0.001,
        "nose_NoseCone_kind": ["ogive", "vonKharman"],
        "nose_NoseCone_position": 0.001,
        # "finSet_Fins_n":,
        "finSet_Fins_rootChord": 0.001,
        "finSet_Fins_tipChord": 0.001,
        "finSet_Fins_span": 0.001,
        "finSet_Fins_position": 0.001,
        # "finSet_Fins_airfoil":,
        "tail_Tail_topRadius": 0.001,
        "tail_Tail_bottomRadius": 0.001,
        "tail_Tail_length": 0.001,
        "tail_Tail_position": 0.001,
        "parachute_Main_CdS": 0.001,
        # "parachute_Main_trigger":,
        "parachute_Main_samplingRate": 0,
        "parachute_Main_lag": 0,
        # "parachute_Main_noise":,
        "parachute_Drogue_CdS": 0.001,
        "parachute_Drogue_samplingRate": 0,
        "parachute_Drogue_lag": 0,
        "positionFirstRailButton": 0.001,
        "positionSecondRailButton": 0.001,
        "railButtonAngularPosition": 0,
        "inclination": 1,
        "heading": 2,
        # "terminateOnApogee":,
        # "maxTime":,
        # "maxTimeStep":,
        # "minTimeStep":,
        # "rtol":,
        # "atol":,
        # "timeOvershoot":,
        # "verbose":,
    }

    test_disp.run_dispersion(
        number_of_simulations=10,
        dispersion_dictionary=disp_dict,
        flight=disp_flight,
    )

    test_disp.import_results()

    assert test_disp.print_results() == None
    assert test_disp.allInfo() == None

    # Delete the test file
    os.remove("test_obj_defined.disp_errors.txt")
    os.remove("test_obj_defined.disp_inputs.txt")
    os.remove("test_obj_defined.disp_outputs.txt")
    os.remove("test_obj_defined.pdf")
    os.remove("test_obj_defined.svg")


@patch("matplotlib.pyplot.show")
def test_dispersion_dict_defined(
    mock_show,
):
    "Test dispersion when it is completely defined by the dispersion dict"

    # start dispersion
    test_disp = Dispersion(filename="test_obj_defined")

    def drogueTrigger(p, y):
        # p = pressure
        # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
        # activate drogue when vz < 0 m/s.
        return True if y[5] < 0 else False

    def mainTrigger(p, y):
        # p = pressure
        # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
        # activate main when vz < 0 m/s and z < 500 + 100 m (+100 due to surface elevation).
        return True if y[5] < 0 and y[2] < 500 + 100 else False

    disp_dict = {
        "railLength": (5.2, 0.001),
        "gravity": (9.80665, 0),
        "date": [datetime.datetime(2023, 1, 28, 12, 0)],
        "latitude": (39.3897, 0),
        "longitude": (-8.288964, 0),
        "elevation": (141.80076211248732, 10),
        "ensembleMember": [0, 1, 2],
        "thrust": ["data/motors/Cesaroni_M1670.eng"],
        "burnOutTime": (3.9, 0.2),
        "grainNumber": [5],
        "grainDensity": (1815, 181.5),
        "grainOuterRadius": (0.033, 0.001),
        "grainInitialInnerRadius": (0.015, 0.001),
        "grainInitialHeight": (0.12, 0.001),
        "grainSeparation": (0.005, 0.001),
        "nozzleRadius": (0.033, 0.001),
        "throatRadius": (0.011, 0.001),
        "totalImpulse": (6026.35, 198.86955000000003),
        "grainsCenterOfMassPosition": (-0.85704, 0.001),
        "nozzlePosition": (-1.255, 0.001),
        "mass": (16.241, 0.1),
        "inertiaI": (6.6, 0.66),
        "inertiaZ": (0.0351, 0.00351),
        "radius": (0.0635, 0.001),
        "powerOffDrag": ["data/calisto/powerOffDragCurve.csv"],
        "powerOnDrag": ["data/calisto/powerOnDragCurve.csv"],
        "powerOffDragFactor": (1, 0.033),
        "powerOnDragFactor": (1, 0.033),
        "centerOfDryMassPosition": (0, 0.001),
        "motor_position": (-1.255, 0.001),
        "nose_name_kind": ["vonKarman"],
        "nose_name_length": (0.55829, 0.001),
        "nose_name_position": (1.278, 0.001),
        "finSet_name_n": [4],
        "finSet_name_rootChord": (0.12, 0.001),
        "finSet_name_tipChord": (0.04, 0.001),
        "finSet_name_span": (0.1, 0.001),
        "finSet_name_position": (-1.04956, 0.001),
        "finSet_name_airfoil": [None],
        "tail_name_topRadius": (0.0635, 0.001),
        "tail_name_bottomRadius": (0.0435, 0.001),
        "tail_name_length": (0.06, 0.001),
        "tail_name_position": (-1.194656, 0.001),
        "parachute_Main_CdS": (10, 2),
        "parachute_Main_trigger": [mainTrigger],
        "parachute_Main_samplingRate": (105, 0),
        "parachute_Main_lag": (1.5, 0),
        "parachute_Main_noise": [(0, 8.3, 0.5)],
        "parachute_Drogue_CdS": (1, 0.3),
        "parachute_Drogue_trigger": [drogueTrigger],
        "parachute_Drogue_samplingRate": (105, 0),
        "parachute_Drogue_lag": (1.5, 0),
        "parachute_Drogue_noise": [(0, 8.3, 0.5)],
        "positionFirstRailButton": (0.2, 0.001),
        "positionSecondRailButton": (-0.5, 0.001),
        "railButtonAngularPosition": (45, 0),
        "inclination": (85, 1),
        "heading": (90, 2)
        # "terminateOnApogee":,
        # "maxTime":,
        # "maxTimeStep":,
        # "minTimeStep":,
        # "rtol":,
        # "atol":,
        # "timeOvershoot":,
        # "verbose":,
    }

    test_disp.run_dispersion(
        number_of_simulations=10,
        dispersion_dictionary=disp_dict,
    )

    test_disp.import_results()

    assert test_disp.print_results() == None
    assert test_disp.allInfo() == None

    os.remove("test_obj_defined.disp_inputs.txt")
    os.remove("test_obj_defined.disp_outputs.txt")
    os.remove("test_obj_defined.disp_errors.txt")
    os.remove("test_obj_defined.svg")
