import os
from unittest.mock import patch

import numpy as np
import pytest

from rocketpy import Function

BURN_TIME = 3.9
GRAIN_NUMBER = 5
GRAIN_SEPARATION = 5 / 1000
GRAIN_DENSITY = 1815
GRAIN_OUTER_RADIUS = 33 / 1000
GRAIN_INITIAL_INNER_RADIUS = 15 / 1000
GRAIN_INITIAL_HEIGHT = 120 / 1000
NOZZLE_RADIUS = 33 / 1000
THROAT_RADIUS = 11 / 1000
GRAIN_VOL = 0.12 * (np.pi * (0.033**2 - 0.015**2))
GRAIN_MASS = GRAIN_VOL * 1815 * 5


@patch("matplotlib.pyplot.show")
def test_motor(mock_show, cesaroni_m1670):  # pylint: disable=unused-argument
    """Tests the SolidMotor.all_info() method.

    Parameters
    ----------
    mock_show : mock
        Mock of the matplotlib.pyplot.show function.
    cesaroni_m1670 : rocketpy.SolidMotor
        The SolidMotor object to be used in the tests.
    """
    assert cesaroni_m1670.all_info() is None


def test_evaluate_inertia_11_asserts_extreme_values(cesaroni_m1670):
    grain_vol = GRAIN_INITIAL_HEIGHT * (
        np.pi * (GRAIN_OUTER_RADIUS**2 - GRAIN_INITIAL_INNER_RADIUS**2)
    )
    grain_mass = grain_vol * GRAIN_DENSITY

    grain_inertia_11_initial = grain_mass * (
        (1 / 4) * (GRAIN_OUTER_RADIUS**2 + GRAIN_INITIAL_INNER_RADIUS**2)
        + (1 / 12) * GRAIN_INITIAL_HEIGHT**2
    )

    initial_value = (GRAIN_NUMBER - 1) / 2
    d = np.linspace(-initial_value, initial_value, GRAIN_NUMBER)
    d = d * (GRAIN_INITIAL_HEIGHT + GRAIN_SEPARATION)

    inertia_11_initial = GRAIN_NUMBER * grain_inertia_11_initial + grain_mass * np.sum(
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
    grain_vol = GRAIN_INITIAL_HEIGHT * (
        np.pi * (GRAIN_OUTER_RADIUS**2 - GRAIN_INITIAL_INNER_RADIUS**2)
    )
    grain_mass = grain_vol * GRAIN_DENSITY

    grain_I_33_initial = (
        grain_mass * (1 / 2.0) * (GRAIN_INITIAL_INNER_RADIUS**2 + GRAIN_OUTER_RADIUS**2)
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
    _, description, data_points = cesaroni_m1670.import_eng(
        "data/motors/cesaroni/Cesaroni_M1670.eng"
    )

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
        f"{2000 * GRAIN_OUTER_RADIUS:3.1f}",
        f"{1000 * 5 * (0.12 + 0.005):3.1f}",
        "0",
        f"{GRAIN_MASS:2.3}",
        f"{GRAIN_MASS:2.3}",
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


def test_initialize_motor_asserts_dynamic_values(cesaroni_m1670):
    grain_vol = GRAIN_INITIAL_HEIGHT * (
        np.pi * (GRAIN_OUTER_RADIUS**2 - GRAIN_INITIAL_INNER_RADIUS**2)
    )
    grain_mass = grain_vol * GRAIN_DENSITY

    assert abs(cesaroni_m1670.max_thrust - 2200.0) < 1e-9
    assert abs(cesaroni_m1670.max_thrust_time - 0.15) < 1e-9
    assert abs(cesaroni_m1670.burn_time[1] - BURN_TIME) < 1e-9
    assert (
        abs(cesaroni_m1670.total_impulse - cesaroni_m1670.thrust.integral(0, BURN_TIME))
        < 1e-9
    )
    assert (
        cesaroni_m1670.average_thrust
        - cesaroni_m1670.thrust.integral(0, BURN_TIME) / BURN_TIME
    ) < 1e-9
    assert abs(cesaroni_m1670.grain_initial_volume - grain_vol) < 1e-9
    assert abs(cesaroni_m1670.grain_initial_mass - grain_mass) < 1e-9
    assert (
        abs(cesaroni_m1670.propellant_initial_mass - GRAIN_NUMBER * grain_mass) < 1e-9
    )
    assert (
        abs(
            cesaroni_m1670.exhaust_velocity(0)
            - cesaroni_m1670.thrust.integral(0, BURN_TIME) / (GRAIN_NUMBER * grain_mass)
        )
        < 1e-9
    )


def test_grain_geometry_progression_asserts_extreme_values(cesaroni_m1670):
    assert np.allclose(
        cesaroni_m1670.grain_inner_radius.get_source()[-1][-1],
        cesaroni_m1670.grain_outer_radius,
    )
    assert (
        cesaroni_m1670.grain_inner_radius.get_source()[0][-1]
        < cesaroni_m1670.grain_inner_radius.get_source()[-1][-1]
    )
    assert (
        cesaroni_m1670.grain_height.get_source()[0][-1]
        > cesaroni_m1670.grain_height.get_source()[-1][-1]
    )


def test_mass_curve_asserts_extreme_values(cesaroni_m1670):
    grain_vol = GRAIN_INITIAL_HEIGHT * (
        np.pi * (GRAIN_OUTER_RADIUS**2 - GRAIN_INITIAL_INNER_RADIUS**2)
    )
    grain_mass = grain_vol * GRAIN_DENSITY

    assert np.allclose(cesaroni_m1670.propellant_mass.get_source()[-1][-1], 0)
    assert np.allclose(
        cesaroni_m1670.propellant_mass.get_source()[0][-1], GRAIN_NUMBER * grain_mass
    )


def test_burn_area_asserts_extreme_values(cesaroni_m1670):
    initial_burn_area = (
        2
        * np.pi
        * (
            GRAIN_OUTER_RADIUS**2
            - GRAIN_INITIAL_INNER_RADIUS**2
            + GRAIN_INITIAL_INNER_RADIUS * GRAIN_INITIAL_HEIGHT
        )
        * GRAIN_NUMBER
    )
    final_burn_area = (
        2
        * np.pi
        * (
            cesaroni_m1670.grain_inner_radius.get_source()[-1][-1]
            * cesaroni_m1670.grain_height.get_source()[-1][-1]
        )
        * GRAIN_NUMBER
    )

    assert np.allclose(cesaroni_m1670.burn_area.get_source()[0][-1], initial_burn_area)
    assert np.allclose(
        cesaroni_m1670.burn_area.get_source()[-1][-1], final_burn_area, atol=1e-6
    )


@pytest.mark.parametrize("tuple_parametric", [(5, 3000)])
def test_reshape_thrust_curve_asserts_resultant_thrust_curve_correct(
    cesaroni_m1670_shifted, tuple_parametric, linear_func
):
    """Tests the reshape_thrust_curve. It checks whether the resultant
    thrust curve is correct when the user passes a certain tuple to the
    reshape_thrust_curve attribute. Also checking for the correct return
    data type.

    Parameters
    ----------
    cesaroni_m1670_shifted : rocketpy.SolidMotor
        The SolidMotor object to be used in the tests.
    tuple_parametric : tuple
        Tuple passed to the reshape_thrust_curve method.
    """

    assert isinstance(
        cesaroni_m1670_shifted.reshape_thrust_curve(linear_func, 1, 3000), Function
    )
    thrust_reshaped = cesaroni_m1670_shifted.thrust.get_source()

    assert thrust_reshaped[1][0] == 0.155 * (tuple_parametric[0] / 4)
    assert thrust_reshaped[-1][0] == tuple_parametric[0]

    assert thrust_reshaped[1][1] == 100 * (tuple_parametric[1] / 7539.1875)
    assert thrust_reshaped[7][1] == 2034 * (tuple_parametric[1] / 7539.1875)
