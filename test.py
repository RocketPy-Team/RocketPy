from rocketpy import Rocket, Motor, Flight, SolidMotor

example_motor = SolidMotor(
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

example_rocket = Rocket(
    motor=example_motor,
    radius=0.0525,
    mass=19.197 - 2.956,
    inertiaI=6.60,
    inertiaZ=0.0351,
    distanceRocketNozzle=-1.255,
    distanceRocketPropellant=-0.85704,
    powerOffDrag=0.5,
    powerOnDrag=0.5,
)

FinSet = example_rocket.addTrapezoidalFins(
    n=4,
    span=0.095,
    rootChord=0.3,
    tipChord=0.05,
    sweepLength=0.23,
    distanceToCM=-0.81,
)

print(FinSet["cl"](1, 0))

example_rocket.allInfo()
