from rocketpy import Environment, SolidMotor, Rocket, Flight, Function
from scipy.signal import savgol_filter
import numpy as np
import pandas as pd


def test_ndrt_2020_rocket_data_asserts_acceptance():
    # Notre Dame Rocket Team 2020 Flight
    # Launched at 19045-18879 Avery Rd, Three Oaks, MI 49128
    # Permission to use flight data given by Brooke Mumma, 2020
    #
    # IMPORTANT RESULTS  (23rd feb)
    # Measured Stability Margin 2.875 cal
    # Official Target Altitude 4,444 ft
    # Measured Altitude 4,320 ft or 1316.736 m
    # Drift: 2275 ft

    # Importing libraries
    from rocketpy import Environment, SolidMotor, Rocket, Flight, Function
    from scipy.signal import savgol_filter
    import numpy as np
    import pandas as pd

    # Defining all parameters
    parameters = {
        # Mass Details
        "rocketMass": (23.321 - 2.475, 0.010),
        # Propulsion Details
        "impulse": (4895.050, 0.033 * 4895.050),
        "burnOut": (3.51, 0.1),
        "nozzleRadius": (49.5 / 2000, 0.001),
        "throatRadius": (21.5 / 2000, 0.001),
        "grainSeparation": (3 / 1000, 0.001),
        "grainDensity": (1519.708, 30),
        "grainOuterRadius": (33 / 1000, 0.001),
        "grainInitialInnerRadius": (15 / 1000, 0.002),
        "grainInitialHeight": (120 / 1000, 0.001),
        # Aerodynamic Details
        "dragCoefficient": (0.44, 0.1),
        "inertiaI": (83.351, 0.3 * 83.351),
        "inertiaZ": (0.15982, 0.3 * 0.15982),
        "radius": (203 / 2000, 0.001),
        "distanceRocketNozzle": (-1.255, 0.100),
        "distanceRocketPropellant": (-0.85704, 0.100),
        "powerOffDrag": (1, 0.033),
        "powerOnDrag": (1, 0.033),
        "noseLength": (0.610, 0.001),
        "noseDistanceToCM": (0.71971, 0.100),
        "finSpan": (0.165, 0.001),
        "finRootChord": (0.152, 0.001),
        "finTipChord": (0.0762, 0.001),
        "finDistanceToCM": (-1.04956, 0.100),
        "transitionTopRadius": (203 / 2000, 0.010),
        "transitionBottomRadius": (155 / 2000, 0.010),
        "transitionLength": (0.127, 0.010),
        "transitiondistanceToCM": (-1.194656, 0.010),
        # Launch and Environment Details
        "windDirection": (0, 3),
        "windSpeed": (1, 0.30),
        "inclination": (90, 1),
        "heading": (181, 3),
        "railLength": (3.353, 0.001),
        # Parachute Details
        "CdSDrogue": (1.5 * np.pi * (24 * 25.4 / 1000) * (24 * 25.4 / 1000) / 4, 0.1),
        "CdSMain": (2.2 * np.pi * (120 * 25.4 / 1000) * (120 * 25.4 / 1000) / 4, 0.1),
        "lag_rec": (1, 0.5),
    }

    # Environment conditions
    Env23 = Environment(
        railLength=parameters.get("railLength")[0],
        gravity=9.81,
        latitude=41.775447,
        longitude=-86.572467,
        date=(2020, 2, 23, 16),
        elevation=206,
    )
    Env23.setAtmosphericModel(
        type="Reanalysis",
        file="tests/fixtures/acceptance/NDRT_2020/ndrt_2020_weather_data_ERA5.nc",
        dictionary="ECMWF",
    )
    Env23.maxExpectedHeight = 2000

    # Motor Information
    L1395 = SolidMotor(
        thrustSource="tests/fixtures/acceptance/NDRT_2020/ndrt_2020_motor_Cesaroni_4895L1395-P.eng",
        burnOut=parameters.get("burnOut")[0],
        grainNumber=5,
        grainSeparation=parameters.get("grainSeparation")[0],
        grainDensity=parameters.get("grainDensity")[0],
        grainOuterRadius=parameters.get("grainOuterRadius")[0],
        grainInitialInnerRadius=parameters.get("grainInitialInnerRadius")[0],
        grainInitialHeight=parameters.get("grainInitialHeight")[0],
        nozzleRadius=parameters.get("nozzleRadius")[0],
        throatRadius=parameters.get("throatRadius")[0],
        interpolationMethod="linear",
    )

    # Rocket information
    NDRT2020 = Rocket(
        motor=L1395,
        radius=parameters.get("radius")[0],
        mass=parameters.get("rocketMass")[0],
        inertiaI=parameters.get("inertiaI")[0],
        inertiaZ=parameters.get("inertiaZ")[0],
        distanceRocketNozzle=parameters.get("distanceRocketNozzle")[0],
        distanceRocketPropellant=parameters.get("distanceRocketPropellant")[0],
        powerOffDrag=parameters.get("dragCoefficient")[0],
        powerOnDrag=parameters.get("dragCoefficient")[0],
    )
    NDRT2020.setRailButtons([0.2, -0.5], 45)
    NoseCone = NDRT2020.addNose(
        length=parameters.get("noseLength")[0],
        kind="tangent",
        distanceToCM=parameters.get("noseDistanceToCM")[0],
    )
    FinSet = NDRT2020.addFins(
        3,
        span=parameters.get("finSpan")[0],
        rootChord=parameters.get("finRootChord")[0],
        tipChord=parameters.get("finTipChord")[0],
        distanceToCM=parameters.get("finDistanceToCM")[0],
    )
    Transition = NDRT2020.addTail(
        topRadius=parameters.get("transitionTopRadius")[0],
        bottomRadius=parameters.get("transitionBottomRadius")[0],
        length=parameters.get("transitionLength")[0],
        distanceToCM=parameters.get("transitiondistanceToCM")[0],
    )

    # Parachute set-up
    def drogueTrigger(p, y):
        # p = pressure
        # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
        # activate drogue when vz < 0 m/s.
        return True if y[5] < 0 else False

    def mainTrigger(p, y):
        # p = pressure
        # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
        # activate main when vz < 0 m/s and z < 167.64 m (AGL) or 550 ft (AGL)
        return True if y[5] < 0 and y[2] < (167.64 + Env23.elevation) else False

    Drogue = NDRT2020.addParachute(
        "Drogue",
        CdS=parameters.get("CdSDrogue")[0],
        trigger=drogueTrigger,
        samplingRate=105,
        lag=parameters.get("lag_rec")[0],
        noise=(0, 8.3, 0.5),
    )
    Main = NDRT2020.addParachute(
        "Main",
        CdS=parameters.get("CdSMain")[0],
        trigger=mainTrigger,
        samplingRate=105,
        lag=parameters.get("lag_rec")[0],
        noise=(0, 8.3, 0.5),
    )

    # Flight
    Flight23 = Flight(
        rocket=NDRT2020,
        environment=Env23,
        inclination=parameters.get("inclination")[0],
        heading=parameters.get("heading")[0],
    )
    Flight23.postProcess()
    df_ndrt_rocketpy = pd.DataFrame(Flight23.z[:, :], columns=["Time", "Altitude"])
    df_ndrt_rocketpy["Vertical Velocity"] = Flight23.vz[:, 1]
    # df_ndrt_rocketpy["Vertical Acceleration"] = Flight23.az[:, 1]
    df_ndrt_rocketpy["Altitude"] -= Env23.elevation

    # Reading data from the flightData (sensors: Raven)
    df_ndrt_raven = pd.read_csv(
        "tests/fixtures/acceptance/NDRT_2020/ndrt_2020_flight_data.csv"
    )
    # convert feet to meters
    df_ndrt_raven[" Altitude (m-AGL)"] = df_ndrt_raven[" Altitude (Ft-AGL)"] / 3.28084
    # Calculate the vertical velocity as a derivative of the altitude
    velocity_raven = [0]
    for i in range(1, len(df_ndrt_raven[" Altitude (m-AGL)"]), 1):
        v = (
            df_ndrt_raven[" Altitude (m-AGL)"][i]
            - df_ndrt_raven[" Altitude (m-AGL)"][i - 1]
        ) / (df_ndrt_raven[" Time (s)"][i] - df_ndrt_raven[" Time (s)"][i - 1])
        if (
            v != 92.85844059786486
            and v != -376.85000927682034
            and v != -57.00530169566588
            and v != -52.752200796647145
            and v != 63.41561104540437
        ):
            # This way we remove the outliers
            velocity_raven.append(v)
        else:
            velocity_raven.append(velocity_raven[-1])
    velocity_raven_filt = savgol_filter(velocity_raven, 51, 3)

    apogee_time_measured = df_ndrt_raven.loc[
        df_ndrt_raven[" Altitude (Ft-AGL)"].idxmax(), " Time (s)"
    ]
    apogee_time_simulated = Flight23.apogeeTime

    assert (
        abs(max(df_ndrt_raven[" Altitude (m-AGL)"]) - max(df_ndrt_rocketpy["Altitude"]))
        / max(df_ndrt_raven[" Altitude (m-AGL)"])
        < 0.015
    )
    assert (max(velocity_raven_filt) - Flight23.maxSpeed) / max(
        velocity_raven_filt
    ) < 0.06
    assert (
        abs(apogee_time_measured - apogee_time_simulated) / apogee_time_simulated < 0.02
    )
