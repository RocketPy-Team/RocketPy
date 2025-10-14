import pytest

from rocketpy import Fluid


@pytest.fixture
def nitrous_oxide_non_constant_fluid():
    """A nitrous_oxide fluid whose density varies with temperature
    and pressure.

    Returns
    -------
    rocketpy.Fluid
    """
    return Fluid(
        name="N2O",
        density="./data/motors/liquid_motor_example/n2o_fluid_parameters.csv",
    )
