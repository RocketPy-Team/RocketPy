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
def test_motor(mock_show):
    example_motor = SolidMotor(
        thrust_source="data/motors/Cesaroni_M1670.eng",
        burn_time=3.9,
        dry_mass=1.815,
        dry_inertia=(0.125, 0.125, 0.002),
        center_of_dry_mass=0.317,
        nozzle_position=0,
        grain_number=5,
        grain_density=1815,
        nozzle_radius=33 / 1000,
        throat_radius=11 / 1000,
        grain_separation=5 / 1000,
        grain_outer_radius=33 / 1000,
        grain_initial_height=120 / 1000,
        grains_center_of_mass_position=0.397,
        grain_initial_inner_radius=15 / 1000,
        interpolation_method="linear",
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )

    assert example_motor.all_info() == None


def test_initialize_motor_asserts_dynamic_values(solid_motor):
    grain_vol = grain_initial_height * (
        np.pi * (grain_outer_radius**2 - grain_initial_inner_radius**2)
    )
    grain_mass = grain_vol * grain_density

    assert solid_motor.max_thrust == 2200.0
    assert solid_motor.max_thrust_time == 0.15
    assert solid_motor.burn_time[1] == burn_time
    assert solid_motor.total_impulse == solid_motor.thrust.integral(0, burn_time)
    assert (
        solid_motor.average_thrust
        == solid_motor.thrust.integral(0, burn_time) / burn_time
    )
    assert solid_motor.grainInitialVolume == grain_vol
    assert solid_motor.grainInitialMass == grain_mass
    assert solid_motor.propellant_initial_mass == grain_number * grain_mass
    assert solid_motor.exhaust_velocity(0) == solid_motor.thrust.integral(
        0, burn_time
    ) / (grain_number * grain_mass)


def test_grain_geometry_progession_asserts_extreme_values(solid_motor):
    assert np.allclose(
        solid_motor.grain_inner_radius.get_source()[-1][-1],
        solid_motor.grain_outer_radius,
    )
    assert (
        solid_motor.grain_inner_radius.get_source()[0][-1]
        < solid_motor.grain_inner_radius.get_source()[-1][-1]
    )
    assert (
        solid_motor.grain_height.get_source()[0][-1]
        > solid_motor.grain_height.get_source()[-1][-1]
    )


def test_mass_curve_asserts_extreme_values(solid_motor):
    grain_vol = grain_initial_height * (
        np.pi * (grain_outer_radius**2 - grain_initial_inner_radius**2)
    )
    grain_mass = grain_vol * grain_density

    assert np.allclose(solid_motor.propellant_mass.get_source()[-1][-1], 0)
    assert np.allclose(
        solid_motor.propellant_mass.get_source()[0][-1], grain_number * grain_mass
    )


def test_burn_area_asserts_extreme_values(solid_motor):
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
            solid_motor.grain_inner_radius.get_source()[-1][-1]
            * solid_motor.grain_height.get_source()[-1][-1]
        )
        * grain_number
    )

    assert np.allclose(solid_motor.burn_area.get_source()[0][-1], initial_burn_area)
    assert np.allclose(
        solid_motor.burn_area.get_source()[-1][-1], final_burn_area, atol=1e-6
    )


def test_evaluate_inertia_11_asserts_extreme_values(solid_motor):
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
        solid_motor.propellant_I_11.get_source()[0][-1], inertia_11_initial, atol=0.01
    )
    assert np.allclose(solid_motor.propellant_I_11.get_source()[-1][-1], 0, atol=1e-6)


def test_evaluate_inertia_33_asserts_extreme_values(solid_motor):
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
        solid_motor.propellant_I_33.get_source()[0][-1], grain_I_33_initial, atol=0.01
    )
    assert np.allclose(solid_motor.propellant_I_33.get_source()[-1][-1], 0, atol=1e-6)


def tests_import_eng_asserts_read_values_correctly(solid_motor):
    comments, description, data_points = solid_motor.import_eng(
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


def tests_export_eng_asserts_exported_values_correct(solid_motor):
    grain_vol = 0.12 * (np.pi * (0.033**2 - 0.015**2))
    grain_mass = grain_vol * 1815 * 5

    solid_motor.export_eng(file_name="tests/solid_motor.eng", motor_name="test_motor")
    comments, description, data_points = solid_motor.import_eng("tests/solid_motor.eng")
    os.remove("tests/solid_motor.eng")

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
        burn_time=3.9,
        dry_mass=1.815,
        dry_inertia=(0.125, 0.125, 0.002),
        center_of_dry_mass=0.317,
        nozzle_position=0,
        grain_number=5,
        grain_density=1815,
        nozzle_radius=33 / 1000,
        throat_radius=11 / 1000,
        grain_separation=5 / 1000,
        grain_outer_radius=33 / 1000,
        grain_initial_height=120 / 1000,
        grains_center_of_mass_position=0.397,
        grain_initial_inner_radius=15 / 1000,
        interpolation_method="linear",
        coordinate_system_orientation="nozzle_to_combustion_chamber",
        reshape_thrust_curve=(5, 3000),
    )

    thrust_reshaped = example_motor.thrust.get_source()
    assert thrust_reshaped[1][0] == 0.155 * (5 / 4)
    assert thrust_reshaped[-1][0] == 5

    assert thrust_reshaped[1][1] == 100 * (3000 / 7539.1875)
    assert thrust_reshaped[7][1] == 2034 * (3000 / 7539.1875)
