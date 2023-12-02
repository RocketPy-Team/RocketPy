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
