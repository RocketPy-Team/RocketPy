# pylint: disable=unused-argument

from unittest.mock import patch

import pytest


@patch("matplotlib.pyplot.show")
@pytest.mark.parametrize(
    "fixture_name",
    [
        "sample_full_mass_flow_rate_tank",
        "sample_empty_mass_flow_rate_tank",
        "sample_full_ullage_tank",
        "sample_empty_ullage_tank",
        "sample_full_level_tank",
        "sample_empty_level_tank",
        "sample_full_mass_tank",
        "sample_empty_mass_tank",
        "real_mass_based_tank_seblm",
        "pressurant_tank",
        "fuel_tank",
        "oxidizer_tank",
        "spherical_oxidizer_tank",
        "cylindrical_variable_density_oxidizer_tank",
    ],
)
def test_tank_all_info(mock_show, fixture_name, request):
    tank = request.getfixturevalue(fixture_name)

    assert tank.info() is None
    assert tank.all_info() is None

    assert (tank.gas_height <= tank.geometry.top).all
    assert (tank.liquid_height <= tank.geometry.top).all
    assert (tank.fluid_volume <= tank.geometry.total_volume).all
