import pytest
from rocketpy import Function
import numpy as np



def test_integral_function():
    test_func = lambda x: 1
    test_values = [[x * test_func(x)]*2 for x in np.linspace(0, 8, 100) ]
    
    myFunc = Function(test_func, inputs="Time", outputs="X Squared")
    myIntegralFunc = myFunc.integralFunction(0, 8, 100)

    assert myIntegralFunc == Function(test_values)

