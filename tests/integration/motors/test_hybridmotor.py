# pylint: disable=unused-argument
from unittest.mock import patch


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
    assert hybrid_motor.info() is None
    assert hybrid_motor.all_info() is None
