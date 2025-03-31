from unittest.mock import patch

import numpy as np
import pytest


@patch("matplotlib.pyplot.show")
def test_airfoil(
    mock_show,  # pylint: disable=unused-argument
    calisto,
    calisto_main_chute,
    calisto_drogue_chute,
    calisto_nose_cone,
    calisto_tail,
):
    test_rocket = calisto
    test_rocket.set_rail_buttons(0.082, -0.618)
    calisto.add_surfaces(calisto_nose_cone, 1.160)
    calisto.add_surfaces(calisto_tail, -1.313)

    test_rocket.add_trapezoidal_fins(
        2,
        span=0.100,
        root_chord=0.120,
        tip_chord=0.040,
        position=-1.168,
        airfoil=("data/airfoils/NACA0012-radians.txt", "radians"),
        name="NACA0012",
    )
    test_rocket.add_trapezoidal_fins(
        2,
        span=0.100,
        root_chord=0.120,
        tip_chord=0.040,
        position=-1.168,
        airfoil=("data/airfoils/e473-10e6-degrees.csv", "degrees"),
        name="E473",
    )
    calisto.parachutes.append(calisto_main_chute)
    calisto.parachutes.append(calisto_drogue_chute)

    static_margin = test_rocket.static_margin(0)

    assert test_rocket.all_info() is None or not abs(static_margin - 2.03) < 0.01


@patch("matplotlib.pyplot.show")
def test_air_brakes_clamp_on(mock_show, calisto_air_brakes_clamp_on):  # pylint: disable=unused-argument
    """Test the air brakes class with clamp on configuration. This test checks
    the basic attributes and the deployment_level setter. It also checks the
    all_info method.

    Parameters
    ----------
    mock_show : mock
        Mock of the matplotlib.pyplot.show method.
    calisto_air_brakes_clamp_on : Rocket instance
        A predefined instance of the calisto with air brakes in clamp on
        configuration.
    """
    air_brakes_clamp_on = calisto_air_brakes_clamp_on.air_brakes[0]

    # test basic attributes
    assert air_brakes_clamp_on.drag_coefficient.__dom_dim__ == 2
    assert (
        air_brakes_clamp_on.reference_area
        == calisto_air_brakes_clamp_on.radius**2 * np.pi
    )
    air_brakes_clamp_on.deployment_level = 0.5
    assert air_brakes_clamp_on.deployment_level == 0.5
    air_brakes_clamp_on.deployment_level = 1.5
    assert air_brakes_clamp_on.deployment_level == 1
    air_brakes_clamp_on.deployment_level = -1
    assert air_brakes_clamp_on.deployment_level == 0
    air_brakes_clamp_on.deployment_level = 0
    assert air_brakes_clamp_on.deployment_level == 0

    assert air_brakes_clamp_on.all_info() is None


@patch("matplotlib.pyplot.show")
def test_air_brakes_clamp_off(  # pylint: disable=unused-argument
    mock_show, calisto_air_brakes_clamp_off
):
    """Test the air brakes class with clamp off configuration. This test checks
    the basic attributes and the deployment_level setter. It also checks the
    all_info method.

    Parameters
    ----------
    mock_show : mock
        Mock of the matplotlib.pyplot.show method.
    calisto_air_brakes_clamp_off : Rocket instance
        A predefined instance of the calisto with air brakes in clamp off
        configuration.
    """
    air_brakes_clamp_off = calisto_air_brakes_clamp_off.air_brakes[0]

    # test basic attributes
    assert air_brakes_clamp_off.drag_coefficient.__dom_dim__ == 2
    assert (
        air_brakes_clamp_off.reference_area
        == calisto_air_brakes_clamp_off.radius**2 * np.pi
    )

    air_brakes_clamp_off.deployment_level = 0.5
    assert air_brakes_clamp_off.deployment_level == 0.5
    air_brakes_clamp_off.deployment_level = 1.5
    assert air_brakes_clamp_off.deployment_level == 1.5
    air_brakes_clamp_off.deployment_level = -1
    assert air_brakes_clamp_off.deployment_level == -1
    air_brakes_clamp_off.deployment_level = 0
    assert air_brakes_clamp_off.deployment_level == 0

    assert air_brakes_clamp_off.all_info() is None


@patch("matplotlib.pyplot.show")
def test_rocket(mock_show, calisto_robust):  # pylint: disable=unused-argument
    test_rocket = calisto_robust
    static_margin = test_rocket.static_margin(0)
    # Check if all_info and static_method methods are working properly
    assert test_rocket.all_info() is None or not abs(static_margin - 2.05) < 0.01


@patch("matplotlib.pyplot.show")
def test_aero_surfaces_infos(  # pylint: disable=unused-argument
    mock_show,
    calisto_nose_cone,
    calisto_tail,
    calisto_trapezoidal_fins,
    calisto_free_form_fins,
):
    assert calisto_nose_cone.all_info() is None
    assert calisto_trapezoidal_fins.all_info() is None
    assert calisto_tail.all_info() is None
    assert calisto_trapezoidal_fins.all_info() is None
    assert calisto_free_form_fins.all_info() is None


@pytest.mark.parametrize(
    "attribute, tolerance",
    [
        ("cpz", 1e-3),
        ("clalpha", 1e-3),
        ("Af", 1e-3),
        ("gamma_c", 1e-3),
        ("Yma", 1e-3),
        ("tau", 1e-3),
        ("roll_geometrical_constant", 1e-3),
        ("lift_interference_factor", 1e-3),
        ("roll_forcing_interference_factor", 1e-3),
        ("roll_damping_interference_factor", 1e-2),
    ],
)
def test_calisto_free_form_fins_equivalence(
    calisto_free_form_fins, calisto_trapezoidal_fins, attribute, tolerance
):
    """Test the equivalence of the free form fins with the same geometric
    characteristics as the trapezoidal fins, comparing cp, cnalpha, and
    geometrical parameters."""

    # Handle the 'clalpha' method comparison differently as it's a callable
    if attribute == "clalpha":
        free_form_value = calisto_free_form_fins.clalpha(0)
        trapezoidal_value = calisto_trapezoidal_fins.clalpha(0)
    else:
        free_form_value = getattr(calisto_free_form_fins, attribute)
        trapezoidal_value = getattr(calisto_trapezoidal_fins, attribute)

    assert abs(free_form_value - trapezoidal_value) < tolerance
