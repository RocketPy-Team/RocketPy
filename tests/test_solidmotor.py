from unittest.mock import patch
import os

import numpy as np
import pytest

from rocketpy import SolidMotor

burnOut = 3.9
grainNumber = 5
grainSeparation = 5 / 1000
grainDensity = 1815
grainOuterRadius = 33 / 1000
grainInitialInnerRadius = 15 / 1000
grainInitialHeight = 120 / 1000
nozzleRadius = 33 / 1000
throatRadius = 11 / 1000


@patch("matplotlib.pyplot.show")
def test_motor(mock_show):
    example_motor = SolidMotor(
        thrustSource="tests/fixtures/motor/Cesaroni_M1670.eng",
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


def test_initialize_motor_asserts_dynamic_values(solid_motor):
    grain_vol = grainInitialHeight * (
        np.pi * (grainOuterRadius**2 - grainInitialInnerRadius**2)
    )
    grain_mass = grain_vol * grainDensity

    assert solid_motor.maxThrust == 2200.0
    assert solid_motor.maxThrustTime == 0.15
    assert solid_motor.burnOutTime == burnOut
    assert solid_motor.totalImpulse == solid_motor.thrust.integral(0, burnOut)
    assert (
        solid_motor.averageThrust == solid_motor.thrust.integral(0, burnOut) / burnOut
    )
    assert solid_motor.grainInitialVolume == grain_vol
    assert solid_motor.grainInitialMass == grain_mass
    assert solid_motor.propellantInitialMass == grainNumber * grain_mass
    assert solid_motor.exhaustVelocity == solid_motor.thrust.integral(0, burnOut) / (
        grainNumber * grain_mass
    )


def test_grain_geometry_progession_asserts_extreme_values(solid_motor):
    assert np.allclose(
        solid_motor.grainInnerRadius.getSource()[-1][-1], solid_motor.grainOuterRadius
    )
    assert (
        solid_motor.grainInnerRadius.getSource()[0][-1]
        < solid_motor.grainInnerRadius.getSource()[-1][-1]
    )
    assert (
        solid_motor.grainHeight.getSource()[0][-1]
        > solid_motor.grainHeight.getSource()[-1][-1]
    )


def test_mass_curve_asserts_extreme_values(solid_motor):
    grain_vol = grainInitialHeight * (
        np.pi * (grainOuterRadius**2 - grainInitialInnerRadius**2)
    )
    grain_mass = grain_vol * grainDensity

    assert np.allclose(solid_motor.mass.getSource()[-1][-1], 0)
    assert np.allclose(solid_motor.mass.getSource()[0][-1], grainNumber * grain_mass)


def test_burn_area_asserts_extreme_values(solid_motor):
    initial_burn_area = (
        2
        * np.pi
        * (
            grainOuterRadius**2
            - grainInitialInnerRadius**2
            + grainInitialInnerRadius * grainInitialHeight
        )
        * grainNumber
    )
    final_burn_area = (
        2
        * np.pi
        * (
            solid_motor.grainInnerRadius.getSource()[-1][-1]
            * solid_motor.grainHeight.getSource()[-1][-1]
        )
        * grainNumber
    )

    assert np.allclose(solid_motor.burnArea.getSource()[0][-1], initial_burn_area)
    assert np.allclose(solid_motor.burnArea.getSource()[-1][-1], final_burn_area)


def test_evaluate_inertia_I_asserts_extreme_values(solid_motor):
    grain_vol = grainInitialHeight * (
        np.pi * (grainOuterRadius**2 - grainInitialInnerRadius**2)
    )
    grain_mass = grain_vol * grainDensity

    grainInertiaI_initial = grain_mass * (
        (1 / 4) * (grainOuterRadius**2 + grainInitialInnerRadius**2)
        + (1 / 12) * grainInitialHeight**2
    )

    initialValue = (grainNumber - 1) / 2
    d = np.linspace(-initialValue, initialValue, grainNumber)
    d = d * (grainInitialHeight + grainSeparation)

    inertiaI_initial = grainNumber * grainInertiaI_initial + grain_mass * np.sum(d**2)

    assert np.allclose(
        solid_motor.inertiaI.getSource()[0][-1], inertiaI_initial, atol=0.01
    )
    assert np.allclose(solid_motor.inertiaI.getSource()[-1][-1], 0, atol=1e-16)


def test_evaluate_inertia_Z_asserts_extreme_values(solid_motor):
    grain_vol = grainInitialHeight * (
        np.pi * (grainOuterRadius**2 - grainInitialInnerRadius**2)
    )
    grain_mass = grain_vol * grainDensity

    grainInertiaZ_initial = (
        grain_mass * (1 / 2.0) * (grainInitialInnerRadius**2 + grainOuterRadius**2)
    )

    assert np.allclose(
        solid_motor.inertiaZ.getSource()[0][-1], grainInertiaZ_initial, atol=0.01
    )
    assert np.allclose(solid_motor.inertiaZ.getSource()[-1][-1], 0, atol=1e-16)


def tests_import_eng_asserts_read_values_correctly(solid_motor):
    comments, description, dataPoints = solid_motor.importEng(
        "tests/fixtures/motor/Cesaroni_M1670.eng"
    )

    assert comments == [";this motor is COTS", ";3.9 burnTime", ";"]
    assert description == ["M1670-BS", "75", "757", "0", "3.101", "5.231", "CTI"]
    assert dataPoints == [
        [0, 0],
        [0.055, 100.0],
        [0.092, 1500.0],
        [0.1, 2000.0],
        [0.15, 2200.0],
        [0.2, 1800.0],
        [0.5, 1950.0],
        [1.0, 2034.0],
        [1.5, 2000.0],
        [2.0, 1900.0],
        [2.5, 1760.0],
        [2.9, 1700.0],
        [3.0, 1650.0],
        [3.3, 530.0],
        [3.4, 350.0],
        [3.9, 0.0],
    ]


def tests_export_eng_asserts_exported_values_correct(solid_motor):
    grain_vol = 0.12 * (np.pi * (0.033**2 - 0.015**2))
    grain_mass = grain_vol * 1815 * 5

    solid_motor.exportEng(fileName="tests/solid_motor.eng", motorName="test_motor")
    comments, description, dataPoints = solid_motor.importEng("tests/solid_motor.eng")
    os.remove("tests/solid_motor.eng")

    assert comments == []
    assert description == [
        "test_motor",
        "{:3.1f}".format(2000 * grainOuterRadius),
        "{:3.1f}".format(1000 * 5 * (0.12 + 0.005)),
        "0",
        "{:2.3}".format(grain_mass),
        "{:2.3}".format(grain_mass),
        "RocketPy",
    ]

    assert dataPoints == [
        [0, 0],
        [0.055, 100.0],
        [0.092, 1500.0],
        [0.1, 2000.0],
        [0.15, 2200.0],
        [0.2, 1800.0],
        [0.5, 1950.0],
        [1.0, 2034.0],
        [1.5, 2000.0],
        [2.0, 1900.0],
        [2.5, 1760.0],
        [2.9, 1700.0],
        [3.0, 1650.0],
        [3.3, 530.0],
        [3.4, 350.0],
        [3.9, 0.0],
    ]


def test_reshape_thrust_curve_asserts_resultant_thrust_curve_correct():
    example_motor = SolidMotor(
        thrustSource="tests/fixtures/motor/Cesaroni_M1670_shifted.eng",
        burnOut=3.9,
        grainNumber=5,
        grainSeparation=5 / 1000,
        grainDensity=1815,
        grainOuterRadius=33 / 1000,
        grainInitialInnerRadius=15 / 1000,
        grainInitialHeight=120 / 1000,
        nozzleRadius=33 / 1000,
        throatRadius=11 / 1000,
        reshapeThrustCurve=(5, 3000),
        interpolationMethod="linear",
    )

    thrust_reshaped = example_motor.thrust.getSource()
    assert thrust_reshaped[1][0] == 0.155 * (5 / 4)
    assert thrust_reshaped[-1][0] == 5

    assert thrust_reshaped[1][1] == 100 * (3000 / 7539.1875)
    assert thrust_reshaped[7][1] == 2034 * (3000 / 7539.1875)
