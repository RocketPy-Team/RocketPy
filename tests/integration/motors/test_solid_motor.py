import numpy as np


def test_only_radial_burn_parameter_effect(cesaroni_m1670):
    """Tests the effect of the only_radial_burn parameter on burn area
    calculation. When enabled, the burn area should only account for
    the radial surface of the grains (no axial regression).

    Parameters
    ----------
    cesaroni_m1670 : rocketpy.SolidMotor
        The SolidMotor object used in the test.
    """
    motor = cesaroni_m1670
    motor.only_radial_burn = True
    assert motor.only_radial_burn

    # When only_radial_burn is active, burn_area should consider only radial area
    burn_area_radial = (
        2
        * np.pi
        * motor.grain_inner_radius(0)
        * motor.grain_height(0)
        * motor.grain_number
    )
    assert np.isclose(motor.burn_area(0), burn_area_radial, atol=1e-12)


def test_evaluate_geometry_updates_properties(cesaroni_m1670):
    """Tests if the grain geometry evaluation correctly updates SolidMotor
    properties after instantiation. It ensures that grain geometry
    functions are created and behave as expected.

    Parameters
    ----------
    cesaroni_m1670 : rocketpy.SolidMotor
        The SolidMotor object used in the test.
    """
    motor = cesaroni_m1670

    assert hasattr(motor, "grain_inner_radius")
    assert hasattr(motor, "grain_height")

    # Checks if the domain of grain_inner_radius function is consistent
    times = motor.grain_inner_radius.x_array
    values = motor.grain_inner_radius.y_array

    # expected initial time
    assert times[0] == 0

    # expected initial inner radius
    assert values[0] == motor.grain_initial_inner_radius

    # final inner radius should be less or equal than outer radius
    assert values[-1] <= motor.grain_outer_radius

    # evaluate at intermediate time
    assert isinstance(motor.grain_inner_radius(0.5), float)
