from unittest.mock import patch

import numpy as np
import pytest

from rocketpy import SolidMotor, Function


@patch("matplotlib.pyplot.show")
def test_motor(mock_show):
    example_motor = SolidMotor(
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

    assert example_motor.allInfo() == None


def test_initilize_motor_correctly(solid_motor):
    grain_vol = 0.12 * (np.pi * (0.033**2 - 0.015**2))
    grain_mass = grain_vol * 1815

    assert solid_motor.maxThrust == 2200.0
    assert solid_motor.maxThrustTime == 0.15
    assert solid_motor.burnOutTime == 3.9
    assert solid_motor.totalImpulse == solid_motor.thrust.integral(0, 3.9)
    assert solid_motor.averageThrust == solid_motor.thrust.integral(0, 3.9) / 3.9
    assert solid_motor.grainInitialVolume == grain_vol
    assert solid_motor.grainInitialMass == grain_mass
    assert solid_motor.propellantInitialMass == 5 * grain_mass
    assert solid_motor.exhaustVelocity == solid_motor.thrust.integral(0, 3.9) / (5 * grain_mass)


def test_grain_geometry_progession(solid_motor):
    assert np.allclose(solid_motor.grainInnerRadius.getSource()[-1][-1], solid_motor.grainOuterRadius)
    assert solid_motor.grainInnerRadius.getSource()[0][-1] < solid_motor.grainInnerRadius.getSource()[-1][-1]
    assert solid_motor.grainHeight.getSource()[0][-1] > solid_motor.grainHeight.getSource()[-1][-1]
