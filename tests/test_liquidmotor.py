from unittest.mock import patch
import os

import numpy as np
import pytest

from rocketpy import LiquidMotor

burn_time = (8, 20)
dry_mass = 0
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
    assert liquid_motor.all_info() == None


def test_liquid_motor_basic_parameters(liquid_motor):
    """Tests the LiquidMotor class construction parameters.

    Parameters
    ----------
    liquid_motor : rocketpy.LiquidMotor
        The LiquidMotor object to be used in the tests.
    """
    assert liquid_motor.burn_time == burn_time
    assert liquid_motor.dry_mass == dry_mass
    assert liquid_motor.center_of_dry_mass == center_of_dry_mass
    assert liquid_motor.nozzle_position == nozzle_position
    assert liquid_motor.nozzle_radius == nozzle_radius
    assert liquid_motor.positioned_tanks[0]["position"] == pressurant_tank_position
    assert liquid_motor.positioned_tanks[1]["position"] == fuel_tank_position
    assert liquid_motor.positioned_tanks[2]["position"] == oxidizer_tank_position


def test_liquid_motor_center_of_mass(liquid_motor):
    """Tests the LiquidMotor class tanks flow and method values.

    Parameters
    ----------
    liquid_motor : rocketpy.LiquidMotor
        The LiquidMotor object to be used in the tests.
    """
    pressurant_tank = liquid_motor.positioned_tanks[0]["tank"]
    fuel_tank = liquid_motor.positioned_tanks[1]["tank"]
    oxidizer_tank = liquid_motor.positioned_tanks[2]["tank"]

    total_pressurant_tank_volume = 4 / 3 * np.pi
    expected_pressurant_mass = np.loadtxt("data/SEBLM/pressurantMassFiltered.csv")[:, 1]
    expected_fuel_volume = (
        np.loadtxt("data/SEBLM/test124_Propane_Volume.csv")[:, 1] * 1e-3
    )
    expected_oxidizer_volume = (
        np.loadtxt("data/SEBLM/test124_Lox_Volume.csv")[:, 1] * 1e-3
    )
