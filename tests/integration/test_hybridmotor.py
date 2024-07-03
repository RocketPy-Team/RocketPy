from unittest.mock import patch

import numpy as np

thrust_function = lambda t: 2000 - 100 * t
burn_time = 10
center_of_dry_mass = 0
dry_inertia = (4, 4, 0.1)
dry_mass = 8
grain_density = 1700
grain_number = 4
grain_initial_height = 0.1
grain_separation = 0
grain_initial_inner_radius = 0.04
grain_outer_radius = 0.1
nozzle_position = -0.4
nozzle_radius = 0.07
grains_center_of_mass_position = -0.1
oxidizer_tank_position = 0.3


@patch("matplotlib.pyplot.show")
def test_hybrid_motor_info(mock_show, hybrid_motor):
    """Tests the HybridMotor.all_info() method.

    Parameters
    ----------
    mock_show : mock
        Mock of the matplotlib.pyplot.show function.
    hybrid_motor : rocketpy.HybridMotor
        The HybridMotor object to be used in the tests.
    """
    assert hybrid_motor.info() == None
    assert hybrid_motor.all_info() == None
