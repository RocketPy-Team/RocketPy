import pytest

from rocketpy import Function


@pytest.fixture
def linear_func():
    """Create a linear function based on a list of points. The function
    represents y = x and may be used on different tests.

    Returns
    -------
    Function
        A linear function representing y = x.
    """
    return Function(
        [[0, 0], [1, 1], [2, 2], [3, 3]],
    )


@pytest.fixture
def linearly_interpolated_func():
    """Create a linearly interpolated function based on a list of points.

    Returns
    -------
    Function
        Linearly interpolated Function, with constant extrapolation
    """
    return Function(
        [[0, 0], [1, 7], [2, -3], [3, -1], [4, 3]],
        interpolation="linear",
        extrapolation="constant",
    )


@pytest.fixture
def spline_interpolated_func():
    """Create a spline interpolated function based on a list of points.

    Returns
    -------
    Function
        Spline interpolated, with natural extrapolation
    """
    return Function(
        [[0, 0], [1, 7], [2, -3], [3, -1], [4, 3]],
        interpolation="spline",
        extrapolation="natural",
    )


@pytest.fixture
def func_from_csv():
    """Create a function based on a csv file. The csv file contains the
    coordinates of the E473 airfoil at 10e6 degrees, but anything else could be
    used here as long as it is a csv file.

    Returns
    -------
    rocketpy.Function
        A function based on a csv file.
    """
    func = Function(
        source="data/airfoils/e473-10e6-degrees.csv",
    )
    return func


@pytest.fixture
def func_2d_from_csv():
    """Create a 2d function based on a csv file.

    Returns
    -------
    rocketpy.Function
        A function based on a csv file.
    """
    # Do not define any of the optional parameters so that the tests can check
    # if the defaults are being used correctly.
    func = Function(
        source="tests/fixtures/function/2d.csv",
    )
    return func


## Controller
@pytest.fixture
def controller_function():
    """Create a controller function that updates the air brakes deployment level
    based on the altitude and vertical velocity of the rocket. This is the same
    controller function that is used in the air brakes example in the
    documentation.

    Returns
    -------
    function
        A controller function
    """

    def controller_function(  # pylint: disable=unused-argument
        time, sampling_rate, state, state_history, observed_variables, air_brakes
    ):
        z = state[2]
        vz = state[5]
        previous_vz = state_history[-1][5]
        if time < 3.9:
            return None
        if z < 1500:
            air_brakes.deployment_level = 0
        else:
            new_deployment_level = (
                air_brakes.deployment_level + 0.1 * vz + 0.01 * previous_vz**2
            )
            if new_deployment_level > air_brakes.deployment_level + 0.2 / sampling_rate:
                new_deployment_level = air_brakes.deployment_level + 0.2 / sampling_rate
            elif (
                new_deployment_level < air_brakes.deployment_level - 0.2 / sampling_rate
            ):
                new_deployment_level = air_brakes.deployment_level - 0.2 / sampling_rate
            else:
                new_deployment_level = air_brakes.deployment_level
            air_brakes.deployment_level = new_deployment_level

    return controller_function


@pytest.fixture
def lambda_quad_func():
    """Create a lambda function based on a string.

    Returns
    -------
    Function
        A lambda function based on a string.
    """
    func = lambda x: x**2  # noqa: E731
    return Function(
        source=func,
    )
