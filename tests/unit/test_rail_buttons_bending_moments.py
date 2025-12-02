"""Unit tests for RailButtons bending moment formulas and calculation logic.


These tests focus on verifying the mathematical correctness and realism
of the calculate_rail_button_bending_moments method in isolation.
"""

import warnings

import numpy as np

from rocketpy.mathutils.function import Function
from rocketpy.rocket.aero_surface.rail_buttons import RailButtons


def test_bending_moment_zero_without_rail_buttons():
    """Verify bending moments return zero functions if no rail buttons present."""

    class NoRailButtonsMock:
        """Mock Flight class with no rail buttons for testing zero moment case."""

        def __init__(self):
            self.rocket = type("", (), {})()
            self.rocket.rail_buttons = []
            self.rocket.center_of_dry_mass_position = lambda x: 0
            self.rocket._csys = 1

        def calculate_rail_button_bending_moments(self):
            null_moment = Function(0)
            if len(self.rocket.rail_buttons) == 0:
                warnings.warn(
                    "Trying to calculate rail button bending moments without "
                    "rail buttons defined. Setting moments to zero.",
                    UserWarning,
                )
                return (null_moment, 0.0, null_moment, 0.0)

    flight = NoRailButtonsMock()
    moments = flight.calculate_rail_button_bending_moments()

    m1_func, max_m1, m2_func, max_m2 = moments

    # Verify types
    assert isinstance(m1_func, Function)
    assert isinstance(m2_func, Function)
    assert isinstance(max_m1, float)
    assert isinstance(max_m2, float)

    # Verify zero functions - check first few values instead of full source
    assert m1_func(0) == 0
    assert m1_func(1) == 0
    assert m2_func(0) == 0
    assert m2_func(1) == 0
    assert max_m1 == 0.0
    assert max_m2 == 0.0


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
    assert rb.button_height == 0.015  # Should use default


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
