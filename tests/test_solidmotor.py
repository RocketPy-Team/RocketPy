from unittest.mock import patch
import os

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


def test_initialize_motor_asserts_dynamic_values(cesaroni_m1670):
    grain_vol = grain_initial_height * (
        np.pi * (grain_outer_radius**2 - grain_initial_inner_radius**2)
    )
    grain_mass = grain_vol * grain_density

    assert abs(cesaroni_m1670.max_thrust - 2200.0) < 1e-9
    assert abs(cesaroni_m1670.max_thrust_time - 0.15) < 1e-9
    assert abs(cesaroni_m1670.burn_time[1] - burn_time) < 1e-9
    assert (
        abs(cesaroni_m1670.total_impulse - cesaroni_m1670.thrust.integral(0, burn_time))
        < 1e-9
    )
    assert (
        cesaroni_m1670.average_thrust
        - cesaroni_m1670.thrust.integral(0, burn_time) / burn_time
    ) < 1e-9
    assert abs(cesaroni_m1670.grain_initial_volume - grain_vol) < 1e-9
    assert abs(cesaroni_m1670.grain_initial_mass - grain_mass) < 1e-9
    assert (
        abs(cesaroni_m1670.propellant_initial_mass - grain_number * grain_mass) < 1e-9
    )
    assert (
        abs(
            cesaroni_m1670.exhaust_velocity(0)
            - cesaroni_m1670.thrust.integral(0, burn_time) / (grain_number * grain_mass)
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
    grain_vol = grain_initial_height * (
        np.pi * (grain_outer_radius**2 - grain_initial_inner_radius**2)
    )
    grain_mass = grain_vol * grain_density

    assert np.allclose(cesaroni_m1670.propellant_mass.get_source()[-1][-1], 0)
    assert np.allclose(
        cesaroni_m1670.propellant_mass.get_source()[0][-1], grain_number * grain_mass
    )


def test_burn_area_asserts_extreme_values(cesaroni_m1670):
    initial_burn_area = (
        2
        * np.pi
        * (
            grain_outer_radius**2
            - grain_initial_inner_radius**2
            + grain_initial_inner_radius * grain_initial_height
        )
        * grain_number
    )
    final_burn_area = (
        2
        * np.pi
        * (
            cesaroni_m1670.grain_inner_radius.get_source()[-1][-1]
            * cesaroni_m1670.grain_height.get_source()[-1][-1]
        )
        * grain_number
    )

    assert np.allclose(cesaroni_m1670.burn_area.get_source()[0][-1], initial_burn_area)
    assert np.allclose(
        cesaroni_m1670.burn_area.get_source()[-1][-1], final_burn_area, atol=1e-6
    )


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
    grain_vol = 0.12 * (np.pi * (0.033**2 - 0.015**2))
    grain_mass = grain_vol * 1815 * 5

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


def test_reshape_thrust_curve_asserts_resultant_thrust_curve_correct():
    example_motor = SolidMotor(
        thrust_source="tests/fixtures/motor/Cesaroni_M1670_shifted.eng",
        burn_time=burn_time,
        dry_mass=1.815,
        dry_inertia=(0.125, 0.125, 0.002),
        center_of_dry_mass_position=0.317,
        nozzle_position=0,
        grain_number=grain_number,
        grain_density=grain_density,
        nozzle_radius=nozzle_radius,
        throat_radius=throat_radius,
        grain_separation=grain_separation,
        grain_outer_radius=grain_outer_radius,
        grain_initial_height=grain_initial_height,
        grains_center_of_mass_position=0.397,
        grain_initial_inner_radius=grain_initial_inner_radius,
        interpolation_method="linear",
        coordinate_system_orientation="nozzle_to_combustion_chamber",
        reshape_thrust_curve=(5, 3000),
    )

    thrust_reshaped = example_motor.thrust.get_source()
    assert thrust_reshaped[1][0] == 0.155 * (5 / 4)
    assert thrust_reshaped[-1][0] == 5

    assert thrust_reshaped[1][1] == 100 * (3000 / 7539.1875)
    assert thrust_reshaped[7][1] == 2034 * (3000 / 7539.1875)
