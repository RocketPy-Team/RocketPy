import os
from unittest.mock import patch

import numpy as np
import pytest

from rocketpy import SolidMotor

burn_time = 3.9
grain_number = 5
grain_separation = 5 / 1000
grain_density = 1815
grain_outer_radius = 33 / 1000
grain_initial_inner_radius = 15 / 1000
grain_initial_height = 120 / 1000
nozzle_radius = 33 / 1000
throat_radius = 11 / 1000
grain_vol = 0.12 * (np.pi * (0.033**2 - 0.015**2))
grain_mass = grain_vol * 1815 * 5


@patch("matplotlib.pyplot.show")
def test_motor(mock_show, cesaroni_m1670):
    """Tests the SolidMotor.all_info() method.

    Parameters
    ----------
    mock_show : mock
        Mock of the matplotlib.pyplot.show function.
    cesaroni_m1670 : rocketpy.SolidMotor
        The SolidMotor object to be used in the tests.
    """
    assert cesaroni_m1670.all_info() == None


def test_evaluate_inertia_11_asserts_extreme_values(cesaroni_m1670):
    grain_vol = grain_initial_height * (
        np.pi * (grain_outer_radius**2 - grain_initial_inner_radius**2)
    )
    grain_mass = grain_vol * grain_density

    grainInertia_11_initial = grain_mass * (
        (1 / 4) * (grain_outer_radius**2 + grain_initial_inner_radius**2)
        + (1 / 12) * grain_initial_height**2
    )

    initial_value = (grain_number - 1) / 2
    d = np.linspace(-initial_value, initial_value, grain_number)
    d = d * (grain_initial_height + grain_separation)

    inertia_11_initial = grain_number * grainInertia_11_initial + grain_mass * np.sum(
        d**2
    )

    # not passing because I_33 is not discrete anymore
    assert np.allclose(
        cesaroni_m1670.propellant_I_11.get_source()[0][-1],
        inertia_11_initial,
        atol=0.01,
    )
    assert np.allclose(
        cesaroni_m1670.propellant_I_11.get_source()[-1][-1], 0, atol=1e-6
    )


def test_evaluate_inertia_33_asserts_extreme_values(cesaroni_m1670):
    grain_vol = grain_initial_height * (
        np.pi * (grain_outer_radius**2 - grain_initial_inner_radius**2)
    )
    grain_mass = grain_vol * grain_density

    grain_I_33_initial = (
        grain_mass
        * (1 / 2.0)
        * (grain_initial_inner_radius**2 + grain_outer_radius**2)
    )

    # not passing because I_33 is not discrete anymore
    assert np.allclose(
        cesaroni_m1670.propellant_I_33.get_source()[0][-1],
        grain_I_33_initial,
        atol=0.01,
    )
    assert np.allclose(
        cesaroni_m1670.propellant_I_33.get_source()[-1][-1], 0, atol=1e-6
    )


def tests_import_eng_asserts_read_values_correctly(cesaroni_m1670):
    """Tests the import_eng method. It checks whether the import operation
    extracts the values correctly.

    Parameters
    ----------
    cesaroni_m1670_shifted : rocketpy.SolidMotor
        The SolidMotor object to be used in the tests.
    """
    comments, description, data_points = cesaroni_m1670.import_eng(
        "tests/fixtures/motor/Cesaroni_M1670.eng"
    )

    assert comments == [";this motor is COTS", ";3.9 burnTime", ";"]
    assert description == ["M1670-BS", "75", "757", "0", "3.101", "5.231", "CTI"]
    assert data_points == [
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


def tests_export_eng_asserts_exported_values_correct(cesaroni_m1670):
    """Tests the export_eng method. It checks whether the exported values
    of the thrust curve still match data_points.

    Parameters
    ----------
    cesaroni_m1670_shifted : rocketpy.SolidMotor
        The SolidMotor object to be used in the tests.
    """

    cesaroni_m1670.export_eng(
        file_name="tests/cesaroni_m1670.eng", motor_name="test_motor"
    )
    comments, description, data_points = cesaroni_m1670.import_eng(
        "tests/cesaroni_m1670.eng"
    )
    os.remove("tests/cesaroni_m1670.eng")

    assert comments == []
    assert description == [
        "test_motor",
        "{:3.1f}".format(2000 * grain_outer_radius),
        "{:3.1f}".format(1000 * 5 * (0.12 + 0.005)),
        "0",
        "{:2.3}".format(grain_mass),
        "{:2.3}".format(grain_mass),
        "RocketPy",
    ]

    assert data_points == [
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
