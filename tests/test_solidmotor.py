import numpy as np
import pytest

from rocketpy import Function

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
