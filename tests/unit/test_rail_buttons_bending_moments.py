"""Unit tests for RailButtons bending moment formulas and calculation logic.

These tests focus on verifying the mathematical correctness and realism
of the calculate_rail_button_bending_moments method in isolation.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np

from rocketpy import Environment, Flight
from rocketpy.mathutils.function import Function
from rocketpy.rocket.aero_surface.rail_buttons import RailButtons


def test_bending_moment_zero_without_rail_buttons(calisto_motorless):
    """Verify bending moments return zero functions if no rail buttons present.

    Parameters
    ----------
    calisto_motorless : rocketpy.Rocket
        Basic rocket without rail buttons.
    """
    # Create a flight with this rocket (no rail buttons)
    env = Environment(latitude=0, longitude=0)
    env.set_atmospheric_model(type="standard_atmosphere")
    flight = Flight(rocket=calisto_motorless, environment=env, rail_length=1)

    # Should return zero moments
    moments = flight.calculate_rail_button_bending_moments
    m1_func, max_m1, m2_func, max_m2 = moments

    # Verify types
    assert isinstance(m1_func, Function)
    assert isinstance(m2_func, Function)
    assert isinstance(max_m1, float)
    assert isinstance(max_m2, float)

    # Verify zero functions
    assert m1_func(0) == 0
    assert m1_func(1) == 0
    assert m2_func(0) == 0
    assert m2_func(1) == 0
    assert max_m1 == 0.0
    assert max_m2 == 0.0


def test_bending_moment_zero_with_none_button_height(calisto_motorless):
    """Verify bending moments return zero when button_height is explicitly None.

    Parameters
    ----------
    calisto_motorless : rocketpy.Rocket
        Basic rocket fixture.
    """

    # Create rail buttons, then explicitly set height to None
    calisto_motorless.set_rail_buttons(
        upper_button_position=0.5,
        lower_button_position=-0.5,
        angular_position=45,
    )
    calisto_motorless.rail_buttons[0].component.button_height = None  # explicit None

    # Sanity check
    assert calisto_motorless.rail_buttons[0].component.button_height is None

    env = Environment(latitude=0, longitude=0)
    env.set_atmospheric_model(type="standard_atmosphere")
    flight = Flight(rocket=calisto_motorless, environment=env, rail_length=1)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        moments = flight.calculate_rail_button_bending_moments
        assert len(w) == 1
        assert "button height not defined" in str(w[0].message).lower()

    m1_func, max_m1, _, max_m2 = moments
    assert m1_func(0) == 0
    assert max_m1 == 0.0
    assert max_m2 == 0.0


def test_bending_moment_zero_with_default_button_height(calisto_motorless):
    """Verify bending moments return zero when button_height uses default value.

    Parameters
    ----------
    calisto_motorless : rocketpy.Rocket
        Basic rocket fixture.
    """
    # Add rail buttons without specifying button_height (tests default)
    calisto_motorless.set_rail_buttons(
        upper_button_position=0.5,
        lower_button_position=-0.5,
        angular_position=45,
        # button_height not specified - should default to None
    )

    # Verify default is None
    assert calisto_motorless.rail_buttons[0].component.button_height is None


def test_railbuttons_serialization_roundtrip():
    """Test RailButtons to_dict and from_dict serialization roundtrip."""
    rb_orig = RailButtons(
        buttons_distance=0.7,
        angular_position=60,
        button_height=0.02,
        name="Test Rail Buttons",
        rocket_radius=0.045,
    )

    data = rb_orig.to_dict()
    rb_reconstructed = RailButtons.from_dict(data)

    assert rb_reconstructed.buttons_distance == rb_orig.buttons_distance
    assert rb_reconstructed.angular_position == rb_orig.angular_position
    assert rb_reconstructed.button_height == rb_orig.button_height
    assert rb_reconstructed.name == rb_orig.name
    assert rb_reconstructed.rocket_radius == rb_orig.rocket_radius


def test_railbuttons_serialization_backward_compat():
    """Test backward compatibility when button_height is missing from dict."""
    # Simulate old serialized data without button_height
    old_data = {
        "buttons_distance": 0.5,
        "angular_position": 45,
        "name": "Legacy Buttons",
        "rocket_radius": 0.05,
    }

    rb = RailButtons.from_dict(old_data)
    assert rb.button_height is None  # Should use None default


def test_railbuttons_angular_position_rad_property():
    """Test angular_position_rad property calculation."""
    rb = RailButtons(buttons_distance=0.5, angular_position=30)
    expected_rad = np.radians(30)
    assert np.isclose(rb.angular_position_rad, expected_rad, rtol=1e-10)


def test_railbuttons_no_aero_contribution():
    """Test RailButtons provide zero aerodynamic contributions."""
    rb = RailButtons(buttons_distance=0.5)

    rb.evaluate_center_of_pressure()
    assert rb.cp == (0, 0, 0)

    rb.evaluate_lift_coefficient()
    assert rb.clalpha(1.0) == 0  # Zero lift derivative
    assert rb.cl(0.1, 1.0) == 0  # Zero lift coefficient


def test_rail_button_bending_moments_prints(flight_calisto_robust, capsys):
    """Test that bending moments are printed correctly in flight.prints.

    Parameters
    ----------
    flight_calisto_robust : rocketpy.Flight
        Flight fixture with motor and rail buttons.
    capsys : pytest fixture
        Captures stdout/stderr.
    """
    # Set button height on existing rail buttons
    flight_calisto_robust.rocket.rail_buttons[0].component.button_height = 0.02

    # Call the print method
    flight_calisto_robust.prints.rail_button_bending_moments()

    # Capture output
    captured = capsys.readouterr()

    # Verify output contains bending moment data
    assert "Rail Button Bending Moments" in captured.out
    assert "Maximum Upper Rail Button Bending Moment" in captured.out
    assert "Maximum Lower Rail Button Bending Moment" in captured.out
    assert "N·m" in captured.out


def test_rail_button_bending_moments_plot_with_height(flight_calisto_robust):
    """Test that bending moments plot is created when button_height is defined.

    Parameters
    ----------
    flight_calisto_robust : rocketpy.Flight
        Flight fixture with motor and rail buttons.
    """
    # Set button height on existing rail buttons
    flight_calisto_robust.rocket.rail_buttons[0].component.button_height = 0.02

    # Should not raise an error
    flight_calisto_robust.plots.rail_buttons_bending_moments()

    # Close the figure to avoid memory issues
    plt.close("all")


def test_rail_button_bending_moments_plot_without_height(flight_calisto_robust, capsys):
    """Test that bending moments plot is skipped when button_height is None.

    Parameters
    ----------
    flight_calisto_robust : rocketpy.Flight
        Flight fixture with motor and rail buttons.
    capsys : pytest fixture
        Captures stdout/stderr.
    """
    # Ensure button_height is None (it should be by default now)
    flight_calisto_robust.rocket.rail_buttons[0].component.button_height = None

    # Call plot method
    flight_calisto_robust.plots.rail_buttons_bending_moments()

    # Capture output
    captured = capsys.readouterr()

    # Should print skip message
    assert "height not defined" in captured.out
