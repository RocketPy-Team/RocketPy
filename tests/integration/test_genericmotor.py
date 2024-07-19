# pylint: disable=unused-argument
from unittest.mock import patch


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
    assert generic_motor.info() is None
    assert generic_motor.all_info() is None
