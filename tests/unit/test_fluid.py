import numpy as np
from scipy.constants import atm, zero_Celsius

from rocketpy.mathutils.function import Function
from rocketpy.motors.fluid import Fluid


def test_constant_density_fluid_properties():
    """Test Fluid with constant density has correct properties."""
    water = Fluid(name="Water", density=1000.0)
    assert water.name == "Water"
    assert water.density == 1000.0
    assert isinstance(water.density_function, Function)
    assert np.isclose(water.density_function.get_value(zero_Celsius, atm), 1000.0)


def test_variable_density_fluid_properties():
    """Test Fluid with variable density function."""
    test_fluid = Fluid(name="TestFluid", density=lambda t, p: 44 * p / (8.314 * t))
    assert test_fluid.name == "TestFluid"
    assert callable(test_fluid.density)
    assert isinstance(test_fluid.density, Function)
    assert isinstance(test_fluid.density_function, Function)
    expected_density = 44 * atm / (8.314 * zero_Celsius)
    assert np.isclose(test_fluid.density_function(zero_Celsius, atm), expected_density)


def test_get_time_variable_density_with_callable_sources():
    """Test get_time_variable_density with callable temperature and pressure."""
    test_fluid = Fluid("TestFluid", lambda t, p: t + p)
    temperature = Function(lambda t: 300 + t)
    pressure = Function(lambda t: 100_000 + 10 * t)
    density_time_func = test_fluid.get_time_variable_density(temperature, pressure)
    assert np.isclose(density_time_func.get_value(0), 300 + 100_000)
    assert np.isclose(density_time_func.get_value(10), 310 + 100_100)


def test_get_time_variable_density_with_arrays():
    """Test get_time_variable_density with array sources."""
    fluid = Fluid("TestFluid", lambda t, p: t + p)
    times = np.array([0, 10])
    temperature = Function(np.column_stack((times, [300, 310])))
    pressure = Function(np.column_stack((times, [100_000, 100_100])))
    density_time_func = fluid.get_time_variable_density(temperature, pressure)
    expected = [300 + 100_000, 310 + 100_100]
    np.testing.assert_allclose(density_time_func(times), expected)


def test_get_time_variable_density_with_callable_and_arrays():
    """Test get_time_variable_density with mixed sources."""
    fluid = Fluid("TestFluid", lambda t, p: t + p)
    times = np.array([0, 10])
    temperature = Function(np.column_stack((times, [300, 310])))
    pressure = Function(lambda t: 100_000 + 10 * t)
    density_time_func = fluid.get_time_variable_density(temperature, pressure)
    expected = [300 + 100_000, 310 + 100_100]
    np.testing.assert_allclose(density_time_func(times), expected)
