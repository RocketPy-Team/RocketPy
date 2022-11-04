import pytest
from rocketpy import Function
import numpy as np

def test_integral_function():
    myFunc = Function(lambda x: 1, inputs="Time", outputs="X Squared")
    myIntegralFunc = myFunc.integralFunction(0, 8)

    assert np.allclose(myIntegralFunc.getSource(), [1, 2, 3, 4, 5, 6, 7, 8])

