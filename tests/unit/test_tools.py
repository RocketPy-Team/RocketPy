import numpy as np
import pytest

from rocketpy.tools import euler_to_quaternions


def test_euler_to_quaternions():
    q0, q1, q2, q3 = euler_to_quaternions(0, 0, 0)
    assert q0 == 1
    assert q1 == 0
    assert q2 == 0
    assert q3 == 0
    q0, q1, q2, q3 = euler_to_quaternions(90, 90, 90)
    assert round(q0, 7) == 0.7071068
    assert round(q1, 7) == 0
    assert round(q2, 7) == 0.7071068
    assert round(q3, 7) == 0
