from unittest.mock import patch


@patch("matplotlib.pyplot.show")
def test_liquid_motor_info(mock_show, liquid_motor):  # pylint: disable=unused-argument
    """Tests the LiquidMotor.all_info() method.

    Parameters
    ----------
    mock_show : mock
        Mock of the matplotlib.pyplot.show function.
    liquid_motor : rocketpy.LiquidMotor
        The LiquidMotor object to be used in the tests.
    """
    assert liquid_motor.info() is None
    assert liquid_motor.all_info() is None
