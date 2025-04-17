import pytest


@pytest.mark.parametrize(
    "fixture_name",
    [
        "stochastic_rail_buttons",
        "stochastic_main_parachute",
        "stochastic_environment",
        "stochastic_environment_custom_sampler",
        "stochastic_tail",
        "stochastic_calisto",
    ],
)
def test_visualize_attributes(request, fixture_name):
    """Tests the visualize_attributes method of the StochasticModel class. This
    test verifies if the method returns None, which means that the method is
    running without breaking.
    """
    fixture = request.getfixturevalue(fixture_name)
    assert fixture.visualize_attributes() is None
