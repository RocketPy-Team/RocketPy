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
    """Tests the visualize_attributes method of the StochasticModel class. It
    must run without breaking and return the formatted report string (which is
    also printed), so the report is never silently lost.
    """
    fixture = request.getfixturevalue(fixture_name)
    report = fixture.visualize_attributes()
    assert isinstance(report, str)
    assert report
