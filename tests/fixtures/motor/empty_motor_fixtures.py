import pytest

from rocketpy.motors.empty_motor import EmptyMotor


@pytest.fixture
def empty_motor():
    """An example of an empty motor with zero thrust and mass.

    Returns
    -------
    rocketpy.EmptyMotor
    """
    return EmptyMotor()
