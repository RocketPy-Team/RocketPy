from unittest.mock import patch

import numpy as np


@patch("matplotlib.pyplot.show")
def test_hybrid_motor_info(mock_show, hybrid_motor):  # pylint: disable=unused-argument
    """Tests the HybridMotor.all_info() method.

    Parameters
    ----------
    mock_show : mock
        Mock of the matplotlib.pyplot.show function.
    hybrid_motor : rocketpy.HybridMotor
        The HybridMotor object to be used in the tests.
    """
    assert hybrid_motor.info() is None
    assert hybrid_motor.all_info() is None


def test_hybrid_motor_only_radial_burn_behavior(hybrid_motor):
    """
    Test if only_radial_burn flag in HybridMotor propagates to its SolidMotor
    and affects burn_area calculation.
    """
    motor = hybrid_motor

    # Activates the radial burning
    motor.solid.only_radial_burn = True
    assert motor.solid.only_radial_burn is True

    # Calculates the expected initial area
    burn_area_radial = (
        2
        * np.pi
        * (motor.solid.grain_inner_radius(0) * motor.solid.grain_height(0))
        * motor.solid.grain_number
    )

    assert np.isclose(motor.solid.burn_area(0), burn_area_radial, atol=1e-12)

    # Deactivates the radial burning and recalculate the geometry
    motor.solid.only_radial_burn = False
    motor.solid.evaluate_geometry()
    assert motor.solid.only_radial_burn is False

    # In this case the burning area also considers the bases of the grain
    inner_radius = motor.solid.grain_inner_radius(0)
    outer_radius = motor.solid.grain_outer_radius
    burn_area_total = (
        burn_area_radial
        + 2 * np.pi * (outer_radius**2 - inner_radius**2) * motor.solid.grain_number
    )
    assert np.isclose(motor.solid.burn_area(0), burn_area_total, atol=1e-12)
    assert motor.solid.burn_area(0) > burn_area_radial
