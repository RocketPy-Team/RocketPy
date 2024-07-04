from unittest.mock import patch

import numpy as np
import pytest
import scipy.integrate

burn_time = (2, 7)
thrust_source = lambda t: 2000 - 100 * (t - 2)
chamber_height = 0.5
chamber_radius = 0.075
chamber_position = -0.25
propellant_initial_mass = 5.0
nozzle_position = -0.5
nozzle_radius = 0.075
dry_mass = 8.0
dry_inertia = (0.2, 0.2, 0.08)


@patch("matplotlib.pyplot.show")
def test_generic_motor_info(mock_show, generic_motor):
    """Tests the GenericMotor.all_info() method.

    Parameters
    ----------
    mock_show : mock
        Mock of the matplotlib.pyplot.show function.
    generic_motor : rocketpy.GenericMotor
        The GenericMotor object to be used in the tests.
    """
    assert generic_motor.info() == None
    assert generic_motor.all_info() == None
