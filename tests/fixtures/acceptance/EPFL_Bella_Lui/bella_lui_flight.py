# EPFL ROCKET TEAM - PROJETO JUPITER collaboration, started on 2020 September.
# Thanks to **Antoine Scardigli** for all helps during these analysis!
# *Brazil - Switzerland*

from rocketpy import Environment, SolidMotor, Rocket, Flight, Function
import numpy as np
import matplotlib.pyplot as plt

# Defining all parameters
parameters = {
    # Mass Details
    "rocketMass": (18.227, 0.010),  # 1.373 = propellant mass
    # Propulsion Details
    "impulse": (2157, 0.03 * 2157),
    "burnOut": (2.43, 0.1),
    "nozzleRadius": (44.45 / 1000, 0.001),
    "throatRadius": (21.4376 / 1000, 0.001),
    "grainSeparation": (3 / 1000, 1 / 1000),
    "grainDensity": (782.4, 30),
    "grainOuterRadius": (85.598 / 2000, 0.001),
    "grainInitialInnerRadius": (33.147 / 1000, 0.002),
    "grainInitialHeight": (152.4 / 1000, 0.001),
    # Aerodynamic Details
    "inertiaI": (0.78267, 0.03 * 0.78267),
    "inertiaZ": (0.064244, 0.03 * 0.064244),
    "radius": (156 / 2000, 0.001),
    "distanceRocketNozzle": (-1.1356, 0.100),
    "distanceRocketPropellant": (-1, 0.100),
    "powerOffDrag": (1, 0.05),
    "powerOnDrag": (1, 0.05),
    "noseLength": (0.242, 0.001),
    "noseDistanceToCM": (1.3, 0.100),
    "finSpan": (0.200, 0.001),
    "finRootChord": (0.280, 0.001),
    "finTipChord": (0.125, 0.001),
    "finDistanceToCM": (-0.75, 0.100),
    "tailTopRadius": (156 / 2000, 0.001),
    "tailBottomRadius": (135 / 2000, 0.001),
    "tailLength": (0.050, 0.001),
    "tailDistanceToCM": (-1.0856, 0.001),
    # Launch and Environment Details
    "windDirection": (0, 5),
    "windSpeed": (1, 0.05),
    "inclination": (89, 1),
    "heading": (45, 5),
    "railLength": (4.2, 0.001),
    # Parachute Details
    "CdSDrogue": (np.pi / 4, 0.20 * np.pi / 4),
    "lag_rec": (1, 0.020),
}

# Environment conditions
Env = Environment(
    railLength=parameters.get("railLength")[0],
    gravity=9.81,
    latitude=47.213476,
    longitude=9.003336,
    date=(2020, 2, 22, 13),
    elevation=407,
)
Env.setAtmosphericModel(type="Reanalysis", file="tests/fixtures/motor/acceptance/EPFL_Bella_Lui/environment.nc", dictionary="ECMWF")
Env.maxExpectedHeight = 2000

# Motor Information
K828FJ = SolidMotor(
    thrustSource="tests/fixtures/motor/acceptance/EPFL_Bella_Lui/AeroTech_K828FJ.eng",
    burnOut=parameters.get("burnOut")[0],
    grainNumber=3,
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
BellaLui = Rocket(
    motor=K828FJ,
    radius=parameters.get("radius")[0],
    mass=parameters.get("rocketMass")[0],
    inertiaI=parameters.get("inertiaI")[0],
    inertiaZ=parameters.get("inertiaZ")[0],
    distanceRocketNozzle=parameters.get("distanceRocketNozzle")[0],
    distanceRocketPropellant=parameters.get("distanceRocketPropellant")[0],
    powerOffDrag=0.43,
    powerOnDrag=0.43,
)
BellaLui.setRailButtons([0.1, -0.5])
NoseCone = BellaLui.addNose(
    length=parameters.get("noseLength")[0],
    kind="tangent",
    distanceToCM=parameters.get("noseDistanceToCM")[0],
)
FinSet = BellaLui.addFins(
    3,
    span=parameters.get("finSpan")[0],
    rootChord=parameters.get("finRootChord")[0],
    tipChord=parameters.get("finTipChord")[0],
    distanceToCM=parameters.get("finDistanceToCM")[0],
)
Tail = BellaLui.addTail(
    topRadius=parameters.get("tailTopRadius")[0],
    bottomRadius=parameters.get("tailBottomRadius")[0],
    length=parameters.get("tailLength")[0],
    distanceToCM=parameters.get("tailDistanceToCM")[0],
)
def drogueTrigger(p, y):
    # p = pressure
    # y = [x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
    # activate drogue when vz < 0 m/s.
    return True if y[5] < 0 else False
Drogue = BellaLui.addParachute(
    "Drogue",
    CdS=parameters.get("CdSDrogue")[0],
    trigger=drogueTrigger,
    samplingRate=105,
    lag=parameters.get("lag_rec")[0],
    noise=(0, 8.3, 0.5),
)

# Define aerodynamic drag coefficients
BellaLui.powerOffDrag = Function(
    [
        (0.01, 0.51),
        (0.02, 0.46),
        (0.04, 0.43),
        (0.28, 0.43),
        (0.29, 0.44),
        (0.45, 0.44),
        (0.49, 0.46),
    ],
    "Mach Number",
    "Drag Coefficient with Power Off",
    "linear",
    "constant",
)
BellaLui.powerOnDrag = Function(
    [
        (0.01, 0.51),
        (0.02, 0.46),
        (0.04, 0.43),
        (0.28, 0.43),
        (0.29, 0.44),
        (0.45, 0.44),
        (0.49, 0.46),
    ],
    "Mach Number",
    "Drag Coefficient with Power On",
    "linear",
    "constant",
)
BellaLui.powerOffDrag *= parameters.get("powerOffDrag")[0]
BellaLui.powerOnDrag *= parameters.get("powerOnDrag")[0]

# Flight
TestFlight = Flight(
    rocket=BellaLui,
    environment=Env,
    inclination=parameters.get("inclination")[0],
    heading=parameters.get("heading")[0],
)
TestFlight.postProcess()

# Comparision with Real Data
flightData = np.loadtxt(
    "tests/fixtures/motor/acceptance/EPFL_Bella_Lui/ERT_KaltBrunn_avionics_clean.csv", skiprows=1, delimiter=",", usecols=(2, 3, 4)
)
time_Kalt = flightData[:573, 0]
altitude_Kalt = flightData[:573, 1]
vertVel_Kalt = flightData[:573, 2]

# Make sure that all vectors have the same length
time_rcp = []
altitude_rcp = []
velocity_rcp = []
acceleration_rcp = []
i = 0
while i <= int(TestFlight.tFinal):
    time_rcp.append(i)
    altitude_rcp.append(TestFlight.z(i) - TestFlight.env.elevation)
    velocity_rcp.append(TestFlight.vz(i))
    acceleration_rcp.append(TestFlight.az(i))
    i += 0.005

time_rcp.append(TestFlight.tFinal)
altitude_rcp.append(0)
velocity_rcp.append(TestFlight.vz(TestFlight.tFinal))
acceleration_rcp.append(TestFlight.az(TestFlight.tFinal))

# Acceleration comparison (will not be used in our publication)
from scipy.signal import savgol_filter
# Calculate the acceleration as a velocity derivative
acceleration_Kalt = [0]
for i in range(1, len(vertVel_Kalt), 1):
    acc = (vertVel_Kalt[i] - vertVel_Kalt[i - 1]) / (time_Kalt[i] - time_Kalt[i - 1])
    acceleration_Kalt.append(acc)

acceleration_Kalt_filt = savgol_filter(acceleration_Kalt, 51, 3)  # Filter our data

# Summary
print("Apogee (AGL)")
print("RocketPy: {:.2f} meters".format(TestFlight.apogee - TestFlight.env.elevation))
print("Real data: {:.2f} meters".format(max(altitude_Kalt)))
print(
    "RocketPy - Real data: {:.2f} meters".format(
        abs(max(altitude_Kalt) - TestFlight.apogee + TestFlight.env.elevation)
    )
)
print()
print("Max Velocity")
print("RocketPy:  {:.2f} m/s".format(max(velocity_rcp)))
print("Real data: {:.2f} m/s".format(max(vertVel_Kalt)))
print("RocketPy - Real data: {:.2f} m/s".format(max(velocity_rcp) - max(vertVel_Kalt)))
print()
print("Max Acceleration")
print("RocketPy:  {:.2f} m/s²".format(max(acceleration_rcp)))
print("Real data (derivative): {:.2f} m/s²".format(max(acceleration_Kalt_filt)))
print(
    "RocketPy - Real data: {:.2f} m/s^2".format(
        max(acceleration_rcp) - max(acceleration_Kalt_filt)
    )
)