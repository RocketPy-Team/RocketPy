from unittest.mock import patch


@patch("matplotlib.pyplot.show")
def test_empty_motor_info(mock_show, empty_motor):  # pylint: disable=unused-argument
    """Tests the LiquidMotor.all_info() method.

    Parameters
    ----------
    mock_show : mock
        Mock of the matplotlib.pyplot.show function.
    empty_motor : rocketpy.EmptyMotor
        The EmptyMotor object to be used in the tests.
    """
    assert empty_motor.info() is None
    assert empty_motor.all_info() is None
