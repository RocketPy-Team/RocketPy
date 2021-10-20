# Valetudo Flight from Projeto Jupiter
# Launched at LASC 2019
# Permission to use flight data given by Projeto Jupiter, 2020

# Importing libraries
from rocketpy import Environment, SolidMotor, Rocket, Flight, Function
from scipy.signal import savgol_filter
import numpy as np
import os

# Import parachute trigger algorithm named SisRec
if os.name == "nt":
    from SisRecWindows import SisRec
else:
    from SisRecLinux import SisRec

# Defining all parameters
analysis_parameters = {
    # Mass Details
    "rocketMass": (8.257, 0.001),
    # Propulsion Details
    "impulse": (1415.15, 35.3),
    "burnOut": (5.274, 1),
    "nozzleRadius": (21.642 / 1000, 0.5 / 1000),
    "throatRadius": (8 / 1000, 0.5 / 1000),
    "grainSeparation": (6 / 1000, 1 / 1000),
    "grainDensity": (1707, 50),
    "grainOuterRadius": (21.4 / 1000, 0.375 / 1000),
    "grainInitialInnerRadius": (9.65 / 1000, 0.375 / 1000),
    "grainInitialHeight": (120 / 1000, 1 / 1000),
    # Aerodynamic Details
    "inertiaI": (3.675, 0.03675),
    "inertiaZ": (0.007, 0.00007),
    "radius": (40.45 / 1000, 0.001),
    "distanceRocketNozzle": (-1.024, 0.001),
    "distanceRocketPropellant": (-0.571, 0.001),
    "powerOffDrag": (0.9081 / 1.05, 0.033),
    "powerOnDrag": (0.9081 / 1.05, 0.033),
    "noseLength": (0.274, 0.001),
    "noseDistanceToCM": (1.134, 0.001),
    "finSpan": (0.077, 0.0005),
    "finRootChord": (0.058, 0.0005),
    "finTipChord": (0.018, 0.0005),
    "finDistanceToCM": (-0.906, 0.001),
    # Launch and Environment Details
    "windDirection": (0, 2),
    "windSpeed": (1, 0.033),
    "inclination": (84.7, 1),
    "heading": (53, 2),
    "railLength": (5.7, 0.0005),
    # "ensembleMember": list(range(10)),
    # Parachute Details
    "CdSDrogue": (0.349 * 1.3, 0.07),
    "lag_rec": (1, 0.5),
    # Electronic Systems Details
    "lag_se": (0.73, 0.16),
}

# Environment conditions
Env = Environment(
    railLength=5.7,
    gravity=9.8,
    date=(2019, 8, 10, 21),
    latitude=-23.363611,
    longitude=-48.011389,
)
Env.setElevation(668)
Env.maxExpectedHeight = 1500
Env.setAtmosphericModel(
    type="Reanalysis",
    file="tests/fixtures/acceptance/PJ_Valetudo/valetudo_weather_data_ERA5.nc",
    dictionary="ECMWF",
)
Env.railLength = analysis_parameters.get("railLength")[0]

# Create motor
Keron = SolidMotor(
    thrustSource="tests/fixtures/acceptance/PJ_Valetudo/valetudo_motor_Keron.csv",
    burnOut=5.274,
    reshapeThrustCurve=(
        analysis_parameters.get("burnOut")[0],
        analysis_parameters.get("impulse")[0],
    ),
    nozzleRadius=analysis_parameters.get("nozzleRadius")[0],
    throatRadius=analysis_parameters.get("throatRadius")[0],
    grainNumber=6,
    grainSeparation=analysis_parameters.get("grainSeparation")[0],
    grainDensity=analysis_parameters.get("grainDensity")[0],
    grainOuterRadius=analysis_parameters.get("grainOuterRadius")[0],
    grainInitialInnerRadius=analysis_parameters.get("grainInitialInnerRadius")[0],
    grainInitialHeight=analysis_parameters.get("grainInitialHeight")[0],
    interpolationMethod="linear",
)

# Create rocket
Valetudo = Rocket(
    motor=Keron,
    radius=analysis_parameters.get("radius")[0],
    mass=analysis_parameters.get("rocketMass")[0],
    inertiaI=analysis_parameters.get("inertiaI")[0],
    inertiaZ=analysis_parameters.get("inertiaZ")[0],
    distanceRocketNozzle=analysis_parameters.get("distanceRocketNozzle")[0],
    distanceRocketPropellant=analysis_parameters.get("distanceRocketPropellant")[0],
    powerOffDrag="tests/fixtures/acceptance/PJ_Valetudo/valetudo_drag_power_off.csv",
    powerOnDrag="tests/fixtures/acceptance/PJ_Valetudo/valetudo_drag_power_on.csv",
)
Valetudo.powerOffDrag *= analysis_parameters.get("powerOffDrag")[0]
Valetudo.powerOnDrag *= analysis_parameters.get("powerOnDrag")[0]
NoseCone = Valetudo.addNose(
    length=analysis_parameters.get("noseLength")[0],
    kind="vonKarman",
    distanceToCM=analysis_parameters.get("noseDistanceToCM")[0],
)
FinSet = Valetudo.addFins(
    n=3,
    rootChord=analysis_parameters.get("finRootChord")[0],
    tipChord=analysis_parameters.get("finTipChord")[0],
    span=analysis_parameters.get("finSpan")[0],
    distanceToCM=analysis_parameters.get("finDistanceToCM")[0],
)
Valetudo.setRailButtons([0.224, -0.93], 30)

# Set up parachutes
sisRecDrogue = SisRec.SisRecSt(0.8998194205245451, 0.2)


def drogueTrigger(p, y):
    return True if sisRecDrogue.update(p / 100000) == 2 else False


Drogue = Valetudo.addParachute(
    "Drogue",
    CdS=analysis_parameters["CdSDrogue"][0],
    trigger=drogueTrigger,
    samplingRate=105,
    lag=analysis_parameters["lag_rec"][0] + analysis_parameters["lag_se"][0],
    noise=(0, 8.3, 0.5),
)
# Prepare parachutes
sisRecDrogue.reset()
sisRecDrogue.enable()

TestFlight = Flight(
    rocket=Valetudo,
    environment=Env,
    inclination=analysis_parameters.get("inclination")[0],
    heading=analysis_parameters.get("heading")[0],
    maxTime=600,
)
TestFlight.postProcess()

# Print summary
TestFlight.info()
