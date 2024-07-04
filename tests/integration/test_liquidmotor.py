from unittest.mock import patch

import numpy as np
import pytest
import scipy.integrate

from rocketpy import Function

burn_time = (8, 20)
dry_mass = 10
dry_inertia = (5, 5, 0.2)
center_of_dry_mass = 0
nozzle_position = -1.364
nozzle_radius = 0.069 / 2
pressurant_tank_position = 2.007
fuel_tank_position = -1.048
oxidizer_tank_position = 0.711


@patch("matplotlib.pyplot.show")
def test_liquid_motor_info(mock_show, liquid_motor):
    """Tests the LiquidMotor.all_info() method.

    Parameters
    ----------
    mock_show : mock
        Mock of the matplotlib.pyplot.show function.
    liquid_motor : rocketpy.LiquidMotor
        The LiquidMotor object to be used in the tests.
    """
    assert liquid_motor.info() == None
    assert liquid_motor.all_info() == None
