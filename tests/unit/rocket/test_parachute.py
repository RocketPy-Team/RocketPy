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


@pytest.mark.parametrize(
    "trigger, expects_udot",
    [
        (lambda p, h, y: p < 900, False),
        (lambda p, h, y, sensors: p < 900, False),
        (lambda p, h, y, acceleration: acceleration is not None, True),
        (lambda p, h, y, sensors, u_dot: p < 900, True),
        (lambda *args: args[0] < 900, False),
    ],
)
def test_callable_trigger_arities_route_arguments(trigger, expects_udot):
    """Every supported trigger signature (3-arg, 4-arg sensors, 4-arg
    acceleration, 5-arg, and variadic ``*args``) builds a wrapper that runs
    without raising and sets ``_expects_udot`` correctly. Regression for the
    variadic trigger that raised ``TypeError`` in early v1.13."""
    parachute = _make_parachute(trigger=trigger)
    result = parachute.triggerfunc(800.0, 500.0, [0.0] * 6, [], [1.0] * 6)
    assert result is True
    assert parachute.triggerfunc._expects_udot is expects_udot


@pytest.mark.parametrize(
    "trigger",
    [800, 800.0, np.int64(800), np.int32(800), np.float64(800), np.float32(800)],
    ids=str,
)
def test_any_real_number_is_read_as_a_height(trigger):
    """A height is anything ``numbers.Real``, not just ``int`` and ``float``.

    The check used to be ``isinstance(trigger, (int, float))``. ``numpy.float64``
    subclasses ``float`` and passed, but ``numpy.int64`` and ``numpy.float32``
    subclass neither, so a height read out of a NumPy array raised even though
    it compares and arithmetics exactly like the value that worked."""
    parachute = _make_parachute(trigger=trigger)

    # Truthiness rather than `is True`: comparing against a NumPy scalar gives
    # back a numpy.bool_, which is not the `True` singleton.
    # falling (vz < 0) and below the trigger height
    assert parachute.triggerfunc(0.0, 700.0, [0.0] * 5 + [-1.0], [], None)
    # falling but still above it
    assert not parachute.triggerfunc(0.0, 900.0, [0.0] * 5 + [-1.0], [], None)
    # below it but still ascending
    assert not parachute.triggerfunc(0.0, 700.0, [0.0] * 5 + [1.0], [], None)


@pytest.mark.parametrize(
    "trigger",
    [True, False, np.bool_(True), complex(800), np.complex64(800), "banana", None, {}],
    ids=str,
)
def test_what_is_not_a_height_is_still_refused(trigger):
    """Widening to ``numbers.Real`` must not turn the check into "anything".

    ``bool`` is the one that has to be excluded by hand, because it *is* an
    ``int``: ``True`` would otherwise be accepted and read as a height of one
    metre, firing the parachute a metre above the ground. ``numpy.bool_`` and
    the complex types need no special case, since neither is ``Real``."""
    with pytest.raises(ValueError, match="Unable to set the trigger"):
        _make_parachute(trigger=trigger)


class TestParachuteTimeTrigger:
    """Fixed-time parachute triggers: ``("time", t_deploy)`` (#437)."""

    def test_time_trigger_fires_at_and_after_deploy_time(self):
        parachute = _make_parachute(trigger=("time", 5.0))
        state = [0.0] * 13

        parachute._eval_time = 4.999
        assert parachute.triggerfunc(101325.0, 1000.0, state, [], None) is False

        parachute._eval_time = 5.0
        assert parachute.triggerfunc(101325.0, 1000.0, state, [], None) is True

        parachute._eval_time = 7.5
        assert parachute.triggerfunc(101325.0, 1000.0, state, [], None) is True

    def test_time_trigger_list_form_and_case_insensitive_kind(self):
        parachute = _make_parachute(trigger=["TIME", 3])
        state = [0.0] * 13

        parachute._eval_time = 2.9
        assert parachute.triggerfunc(101325.0, 1000.0, state, [], None) is False
        parachute._eval_time = 3.0
        assert parachute.triggerfunc(101325.0, 1000.0, state, [], None) is True

    def test_time_trigger_does_not_require_descent_or_height(self):
        parachute = _make_parachute(trigger=("time", 1.0))
        assert parachute._trigger_falling_only is False
        assert parachute._trigger_needs_height is False

        # Ascending state at altitude well above any height trigger.
        ascending = [0.0, 0.0, 2000.0, 0.0, 0.0, 50.0] + [0.0] * 7
        parachute._eval_time = 1.0
        assert parachute.triggerfunc(101325.0, 2000.0, ascending, [], None) is True

    def test_time_trigger_false_when_eval_time_unset(self):
        parachute = _make_parachute(trigger=("time", 0.0))
        assert parachute.triggerfunc(101325.0, 0.0, [0.0] * 13, [], None) is False

    def test_time_trigger_accepts_numpy_scalar_delay(self):
        parachute = _make_parachute(trigger=("time", np.float64(2.5)))
        parachute._eval_time = 2.5
        assert parachute.triggerfunc(101325.0, 0.0, [0.0] * 13, [], None) is True

    @pytest.mark.parametrize(
        "trigger",
        [
            ("time", -1.0),
            ("time", True),
            ("time", "soon"),
            # float() would happily eat this one; the numeric boundary must not
            ("time", "3.0"),
            ("time",),
            ("burnout", 3.0),
            ("launch", 5.0),
        ],
        ids=str,
    )
    def test_invalid_time_triggers_are_refused(self, trigger):
        with pytest.raises(ValueError, match="Unable to set the trigger"):
            _make_parachute(trigger=trigger)

    def test_to_dict_round_trip_preserves_time_trigger(self):
        original = _make_parachute(trigger=("time", 4.0))
        restored = Parachute.from_dict(original.to_dict())
        assert restored.trigger == ("time", 4.0)
        restored._eval_time = 4.0
        assert restored.triggerfunc(101325.0, 0.0, [0.0] * 13, [], None) is True
