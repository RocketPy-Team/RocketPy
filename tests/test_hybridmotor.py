from unittest.mock import patch
import os

import numpy as np
import pytest

from rocketpy import HybridMotor


def test_Initial_Center_Of_Mass_Position_correct():
    example_motor = HybridMotor(
        thrustSource="data/motors/Hypertek_835CC125J-K240.eng",
        burnOut=5.4,
        distanceNozzleMotorReference=1,
        grainNumber=6,
        grainDensity=1707,
        grainOuterRadius=21.40 / 1000,
        grainInitialInnerRadius=9.65 / 1000,
        grainInitialHeight=120 / 1000,
        oxidizerTankRadius=62.5 / 1000,
        oxidizerTankHeight=600 / 1000,
        oxidizerInitialPressure=51.03,
        oxidizerDensity=1.98,
        oxidizerMolarMass=44.01,
        oxidizerInitialVolume=62.5 / 1000 * 62.5 / 1000 * np.pi * 600 / 1000,
        distanceGrainToTank=200 / 1000,
        injectorArea=3e-05,
    )

    assert abs(example_motor.zCM(0)) - abs(0.005121644685784456) < 1e-6

def hybrid_rse_input():
    nozzle_reference = 609.6/1000
    rse_motor = HybridMotor(
        thrustSource="tests/fixtures/motor/Contrail_K234-BG.rse",
        burnOut=7.05,
        distanceNozzleMotorReference=nozzle_reference,
        grainNumber=1,
        grainDensity=1707,
        grainOuterRadius=21.40 / 1000,
        grainInitialInnerRadius=9.65 / 1000,
        grainInitialHeight=120 / 1000,
        oxidizerTankRadius=62.5 / 1000,
        oxidizerTankHeight=600 / 1000,
        oxidizerInitialPressure=51.03,
        oxidizerDensity=1.98,
        oxidizerMolarMass=44.01,
        oxidizerInitialVolume= np.pi * (62.5 / 1000) ** 2  * 600 / 1000,
        distanceGrainToTank=200 / 1000,
        injectorArea=3e-05,
    )

    assert rse_motor.zCM(rse_motor.burnOutTime) == nozzle_reference
    