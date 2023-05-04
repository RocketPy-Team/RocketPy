from unittest.mock import patch
import os

import numpy as np
import pytest

from rocketpy import SolidMotor

burn_out = 3.9
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
        thrust_source="tests/fixtures/motor/Cesaroni_M1670.eng",
        burn_out=3.9,
        grain_number=5,
        grain_separation=5 / 1000,
        grain_density=1815,
        grain_outer_radius=33 / 1000,
        grain_initial_inner_radius=15 / 1000,
        grain_initial_height=120 / 1000,
        nozzle_radius=33 / 1000,
        throat_radius=11 / 1000,
        interpolation_method="linear",
        grains_center_of_mass_position=0.39796,
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )

    assert example_motor.allinfo() == None


def test_initialize_motor_asserts_dynamic_values(solid_motor):
    grain_vol = grain_initial_height * (
        np.pi * (grain_outer_radius**2 - grain_initial_inner_radius**2)
    )
    grain_mass = grain_vol * grain_density

    assert solid_motor.max_thrust == 2200.0
    assert solid_motor.max_thrust_time == 0.15
    assert solid_motor.burn_out_time == burn_out
    assert solid_motor.total_impulse == solid_motor.thrust.integral(0, burn_out)
    assert (
        solid_motor.average_thrust
        == solid_motor.thrust.integral(0, burn_out) / burn_out
    )
    assert solid_motor.grain_initial_volume == grain_vol
    assert solid_motor.grain_initial_mass == grain_mass
    assert solid_motor.propellant_initial_mass == grain_number * grain_mass
    assert solid_motor.exhaust_velocity == solid_motor.thrust.integral(0, burn_out) / (
        grain_number * grain_mass
    )


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

    assert np.allclose(solid_motor.mass.get_source()[-1][-1], 0)
    assert np.allclose(solid_motor.mass.get_source()[0][-1], grain_number * grain_mass)


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
    assert np.allclose(solid_motor.burn_area.get_source()[-1][-1], final_burn_area)


def test_evaluate_inertia_I_asserts_extreme_values(solid_motor):
    grain_vol = grain_initial_height * (
        np.pi * (grain_outer_radius**2 - grain_initial_inner_radius**2)
    )
    grain_mass = grain_vol * grain_density

    grainInertiaI_initial = grain_mass * (
        (1 / 4) * (grain_outer_radius**2 + grain_initial_inner_radius**2)
        + (1 / 12) * grain_initial_height**2
    )

    initialValue = (grain_number - 1) / 2
    d = np.linspace(-initialValue, initialValue, grain_number)
    d = d * (grain_initial_height + grain_separation)

    inertiaI_initial = grain_number * grainInertiaI_initial + grain_mass * np.sum(
        d**2
    )

    assert np.allclose(
        solid_motor.inertia_i.get_source()[0][-1], inertiaI_initial, atol=0.01
    )
    assert np.allclose(solid_motor.inertia_i.get_source()[-1][-1], 0, atol=1e-16)


def test_evaluate_inertia_Z_asserts_extreme_values(solid_motor):
    grain_vol = grain_initial_height * (
        np.pi * (grain_outer_radius**2 - grain_initial_inner_radius**2)
    )
    grain_mass = grain_vol * grain_density

    grainInertiaZ_initial = (
        grain_mass
        * (1 / 2.0)
        * (grain_initial_inner_radius**2 + grain_outer_radius**2)
    )

    assert np.allclose(
        solid_motor.inertia_z.get_source()[0][-1], grainInertiaZ_initial, atol=0.01
    )
    assert np.allclose(solid_motor.inertia_z.get_source()[-1][-1], 0, atol=1e-16)


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
        burn_out=3.9,
        grain_number=5,
        grain_separation=5 / 1000,
        grain_density=1815,
        grain_outer_radius=33 / 1000,
        grain_initial_inner_radius=15 / 1000,
        grain_initial_height=120 / 1000,
        nozzle_radius=33 / 1000,
        throat_radius=11 / 1000,
        reshape_thrust_curve=(5, 3000),
        interpolation_method="linear",
        grains_center_of_mass_position=0.39796,
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )

    thrust_reshaped = example_motor.thrust.get_source()
    assert thrust_reshaped[1][0] == 0.155 * (5 / 4)
    assert thrust_reshaped[-1][0] == 5

    assert thrust_reshaped[1][1] == 100 * (3000 / 7539.1875)
    assert thrust_reshaped[7][1] == 2034 * (3000 / 7539.1875)
