"""Unit tests for the Parachute class, focusing on the radius and
drag_coefficient parameters introduced in PR #889."""

import numpy as np
import pytest

from rocketpy import Parachute


def _make_parachute(**kwargs):
    defaults = {
        "name": "test",
        "cd_s": 10.0,
        "trigger": "apogee",
        "sampling_rate": 100,
    }
    defaults.update(kwargs)
    return Parachute(**defaults)


class TestParachuteRadiusEstimation:
    """Tests for auto-computed radius from cd_s and drag_coefficient."""

    def test_radius_auto_computed_from_cd_s_default_drag_coefficient(self):
        """When radius is not provided the radius is estimated using the
        default drag_coefficient of 1.4 and the formula R = sqrt(cd_s / (Cd * pi))."""
        cd_s = 10.0
        parachute = _make_parachute(cd_s=cd_s)
        expected_radius = np.sqrt(cd_s / (1.4 * np.pi))
        assert parachute.radius == pytest.approx(expected_radius, rel=1e-9)

    def test_radius_auto_computed_uses_custom_drag_coefficient(self):
        """When drag_coefficient is provided and radius is not, the radius
        must be estimated using the given drag_coefficient."""
        cd_s = 10.0
        custom_cd = 0.75
        parachute = _make_parachute(cd_s=cd_s, drag_coefficient=custom_cd)
        expected_radius = np.sqrt(cd_s / (custom_cd * np.pi))
        assert parachute.radius == pytest.approx(expected_radius, rel=1e-9)

    def test_explicit_radius_overrides_estimation(self):
        """When radius is explicitly provided, it must be used directly and
        drag_coefficient must be ignored for the radius calculation."""
        explicit_radius = 2.5
        parachute = _make_parachute(radius=explicit_radius, drag_coefficient=0.5)
        assert parachute.radius == explicit_radius

    def test_drag_coefficient_stored_on_instance(self):
        """drag_coefficient must be stored as an attribute regardless of
        whether radius is provided or not."""
        parachute = _make_parachute(drag_coefficient=0.75)
        assert parachute.drag_coefficient == 0.75

    def test_drag_coefficient_default_is_1_4(self):
        """Default drag_coefficient must be 1.4 for backward compatibility."""
        parachute = _make_parachute()
        assert parachute.drag_coefficient == pytest.approx(1.4)

    def test_drogue_radius_smaller_than_main(self):
        """A drogue (cd_s=1.0) must have a smaller radius than a main (cd_s=10.0)
        when using the same drag_coefficient."""
        main = _make_parachute(cd_s=10.0)
        drogue = _make_parachute(cd_s=1.0)
        assert drogue.radius < main.radius

    def test_drogue_radius_approximately_0_48(self):
        """For cd_s=1.0 and drag_coefficient=1.4, the estimated radius
        must be approximately 0.48 m (fixes the previous hard-coded 1.5 m)."""
        drogue = _make_parachute(cd_s=1.0)
        assert drogue.radius == pytest.approx(0.476, abs=1e-3)

    def test_main_radius_approximately_1_51(self):
        """For cd_s=10.0 and drag_coefficient=1.4, the estimated radius
        must be approximately 1.51 m, matching the old hard-coded value."""
        main = _make_parachute(cd_s=10.0)
        assert main.radius == pytest.approx(1.508, abs=1e-3)


class TestParachuteSerialization:
    """Tests for to_dict / from_dict round-trip including drag_coefficient."""

    def test_to_dict_includes_drag_coefficient(self):
        """to_dict must include the drag_coefficient key."""
        parachute = _make_parachute(drag_coefficient=0.75)
        data = parachute.to_dict()
        assert "drag_coefficient" in data
        assert data["drag_coefficient"] == 0.75

    def test_from_dict_round_trip_preserves_drag_coefficient(self):
        """A Parachute serialized to dict and restored must have the same
        drag_coefficient."""
        original = _make_parachute(cd_s=5.0, drag_coefficient=0.75)
        data = original.to_dict()
        restored = Parachute.from_dict(data)
        assert restored.drag_coefficient == pytest.approx(0.75)
        assert restored.radius == pytest.approx(original.radius, rel=1e-9)

    def test_from_dict_defaults_drag_coefficient_to_1_4_when_absent(self):
        """Dicts serialized before drag_coefficient was added (no key) must
        fall back to 1.4 for backward compatibility."""
        data = {
            "name": "legacy",
            "cd_s": 10.0,
            "trigger": "apogee",
            "sampling_rate": 100,
            "lag": 0,
            "noise": (0, 0, 0),
            # no drag_coefficient key — simulates old serialized data
        }
        parachute = Parachute.from_dict(data)
        assert parachute.drag_coefficient == pytest.approx(1.4)


class TestParachuteOpeningShockForce:
    """Tests for the opening shock force estimation (issue #161)."""

    def test_opening_shock_force_matches_knacke_formula(self):
        """The estimate must equal Cx * q * cd_s with q = 0.5 * rho * v**2."""
        cd_s = 10.0
        cx = 1.6
        air_density = 1.05
        velocity = 30.0
        parachute = _make_parachute(cd_s=cd_s, opening_shock_coefficient=cx)

        force = parachute.evaluate_opening_shock_force(air_density, velocity)

        expected = cx * (0.5 * air_density * velocity**2) * cd_s
        assert force == pytest.approx(expected, rel=1e-12)

    def test_opening_shock_force_exceeds_steady_drag(self):
        """For a coefficient above 1, the peak load must exceed the steady
        drag force q * cd_s at the same condition."""
        cd_s = 10.0
        air_density = 1.05
        velocity = 30.0
        parachute = _make_parachute(cd_s=cd_s, opening_shock_coefficient=1.6)

        force = parachute.evaluate_opening_shock_force(air_density, velocity)

        steady_drag = (0.5 * air_density * velocity**2) * cd_s
        assert force > steady_drag

    def test_opening_shock_force_scales_with_velocity_squared(self):
        """Doubling the inflation velocity must quadruple the peak load."""
        parachute = _make_parachute(opening_shock_coefficient=1.5)

        force_low = parachute.evaluate_opening_shock_force(1.2, 20.0)
        force_high = parachute.evaluate_opening_shock_force(1.2, 40.0)

        assert force_high == pytest.approx(4.0 * force_low, rel=1e-12)

    def test_opening_shock_force_requires_coefficient(self):
        """Without an opening_shock_coefficient the estimate is disabled."""
        parachute = _make_parachute()  # opening_shock_coefficient defaults to None

        assert parachute.opening_shock_coefficient is None
        with pytest.raises(ValueError, match="opening_shock_coefficient"):
            parachute.evaluate_opening_shock_force(1.2, 30.0)

    def test_opening_shock_coefficient_survives_dict_round_trip(self):
        """to_dict/from_dict must preserve the opening_shock_coefficient."""
        parachute = _make_parachute(opening_shock_coefficient=1.7)

        restored = Parachute.from_dict(parachute.to_dict())

        assert restored.opening_shock_coefficient == 1.7
