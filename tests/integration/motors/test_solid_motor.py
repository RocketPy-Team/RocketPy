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


def test_only_radial_burn_grain_geometry_time_evolution(cesaroni_m1670):
    """Regression for PR #944 (corrected Jacobian of the ``only_radial_burn``
    branch of ``evaluate_geometry``).

    With radial-only burning the integrated grain geometry over the full burn
    must have a constant grain height (no axial regression) and a monotonically
    growing inner radius, bounded by the initial inner radius and the grain
    outer radius. A wrong Jacobian corrupts this integration.
    """
    motor = cesaroni_m1670
    motor.only_radial_burn = True
    motor.evaluate_geometry()

    times = np.linspace(0, motor.grain_burn_out, 30)
    heights = np.array([motor.grain_height(t) for t in times])
    radii = np.array([motor.grain_inner_radius(t) for t in times])

    # Radial-only burn: grain height stays at its initial value the whole burn.
    assert np.allclose(heights, motor.grain_initial_height, atol=1e-9)

    # Inner radius grows monotonically (non-decreasing) and actually increases.
    assert np.all(np.diff(radii) >= -1e-12)
    assert radii[-1] > radii[0]

    # Physical bounds on the inner radius.
    assert np.isclose(radii[0], motor.grain_initial_inner_radius)
    assert radii[-1] <= motor.grain_outer_radius + 1e-12
