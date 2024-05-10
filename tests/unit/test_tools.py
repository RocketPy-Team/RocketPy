import numpy as np
import pytest

from rocketpy.tools import euler_to_quaternions


@pytest.mark.parametrize(
    "angles, expected_quaternions",
    [((0, 0, 0), (1, 0, 0, 0)), ((90, 90, 90), (0.7071068, 0, 0.7071068, 0))],
)
def test_euler_to_quaternions(angles, expected_quaternions):
    q0, q1, q2, q3 = euler_to_quaternions(*angles)
    assert round(q0, 7) == expected_quaternions[0]
    assert round(q1, 7) == expected_quaternions[1]
    assert round(q2, 7) == expected_quaternions[2]
    assert round(q3, 7) == expected_quaternions[3]
