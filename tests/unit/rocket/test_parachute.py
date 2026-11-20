"""Unit tests for the Parachute class, focusing on the radius and
drag_coefficient parameters introduced in PR #889."""

import warnings

import numpy as np
import pytest

from rocketpy import Event, Parachute, Rocket
from rocketpy.simulation.events.event_context import EventContext


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
            # no drag_coefficient key — simulates old serialized data
        }
        parachute = Parachute.from_dict(data)
        assert parachute.drag_coefficient == pytest.approx(1.4)


def test_a_context_trigger_is_called_without_a_wrapper():
    """The trigger the user wrote is the one the event calls, with no hop."""

    def my_trigger(context):
        return context["height_agl"] < 800

    parachute = Parachute(
        name="Main", cd_s=10.0, trigger=my_trigger, sampling_rate=100, lag=0
    )
    assert parachute.event.trigger is my_trigger
    assert parachute.triggerfunc is my_trigger


def test_a_built_in_height_trigger_is_called_without_a_wrapper():
    parachute = Parachute(name="Main", cd_s=10.0, trigger=800, sampling_rate=100, lag=0)
    assert parachute.event.trigger is parachute.triggerfunc
    # and it still answers correctly
    assert (
        parachute.event.trigger({"canonical_state": [0.0] * 13, "height_agl": 700})
        is False
    )
    descending = [0.0] * 5 + [-1.0] + [0.0] * 7
    assert (
        parachute.event.trigger({"canonical_state": descending, "height_agl": 700})
        is True
    )
    assert (
        parachute.event.trigger({"canonical_state": descending, "height_agl": 900})
        is False
    )


def test_a_legacy_positional_trigger_still_goes_through_the_adapter():
    """The deprecated form needs the wrapper, so it keeps one."""
    with pytest.warns(DeprecationWarning, match="Positional-argument"):
        parachute = Parachute(
            name="Drogue",
            cd_s=1.0,
            trigger=lambda p, h, y: h < 800,
            sampling_rate=100,
            lag=0,
        )
    assert parachute.event.trigger is not parachute.triggerfunc


def test_noise_warns_visibly_and_generates_no_signal():
    """A noise different from zero is ignored, and the user is told so with a
    warning that Python shows by default."""
    with pytest.warns(FutureWarning, match="has no effect"):
        parachute = _make_parachute(noise=(0, 8.3, 0.5))

    assert parachute.noise == (0, 8.3, 0.5)
    assert parachute.noise_bias == 0
    assert parachute.noise_deviation == pytest.approx(8.3)
    assert parachute.noise_signal == []
    assert parachute.noisy_pressure_signal == []
    assert parachute.clean_pressure_signal == []
    assert not hasattr(parachute, "noise_function")


def test_noise_warning_is_raised_through_rocket_add_parachute():
    """The warning must also reach users who add the parachute through the
    rocket, which is where a DeprecationWarning used to be hidden."""
    rocket = Rocket(
        radius=0.0635,
        mass=14.4,
        inertia=(6.3, 6.3, 0.03),
        power_off_drag=0.5,
        power_on_drag=0.5,
        center_of_mass_without_motor=0,
    )
    with pytest.warns(FutureWarning, match="has no effect"):
        rocket.add_parachute(
            name="main",
            cd_s=10.0,
            trigger="apogee",
            sampling_rate=100,
            noise=(0, 8.3, 0.5),
        )


def test_default_noise_does_not_warn():
    """The default noise of zero is the supported case and stays silent."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _make_parachute()
        _make_parachute(noise=(0, 0, 0))


def test_parachute_does_not_draw_from_the_global_random_generator():
    """Creating or resetting a parachute must not move numpy's global
    generator, with or without a noise given."""
    np.random.seed(1234)
    _, keys_before, position_before, *_ = np.random.get_state()

    parachute = _make_parachute()
    parachute._reset_signals()
    with pytest.warns(FutureWarning):
        noisy = _make_parachute(noise=(0, 8.3, 0.5))
    noisy._reset_signals()

    _, keys_after, position_after, *_ = np.random.get_state()
    assert position_after == position_before
    assert np.array_equal(keys_after, keys_before)


def test_to_dict_outputs_carry_no_noise_entries():
    """Nothing about noise is recorded, so nothing about it is saved."""
    data = _make_parachute().to_dict(include_outputs=True)

    for key in (
        "noise",
        "noise_signal",
        "noise_function",
        "noisy_pressure_signal",
        "clean_pressure_signal",
    ):
        assert key not in data


def _check_apogee_trigger(parachute, time, vz, height=100.0):
    """Check the parachute's event once, as the flight does, with the rocket
    ``height`` above the ground and moving vertically at ``vz``. Returns
    whether the trigger fired."""
    state = [0.0] * 5 + [vz] + [0.0] * 7
    context = EventContext(
        time=time,
        canonical_state=state,
        raw_state=state,
        phase_names=(),
        height_agl=height,
    )
    return parachute.event(context, trigger_only=True)


def test_the_apogee_trigger_holds_once_the_climb_turns_into_a_descent():
    """The trigger holds from the apogee on. The parachute deploys once
    because its event fires only once."""
    parachute = Parachute(
        name="Drogue", cd_s=1.0, trigger="apogee", sampling_rate=100, lag=0
    )

    fired = [
        _check_apogee_trigger(parachute, time, vz)
        for time, vz in ((1.0, 5.0), (2.0, 1.0), (3.0, -1.0), (4.0, -5.0))
    ]

    assert fired == [False, False, True, True]
    assert parachute.event.trigger_only_once is True


def test_the_apogee_trigger_fires_at_once_for_a_rocket_already_descending():
    """A flight that starts past its apogee has no climb to compare against,
    so the first check of a descending rocket deploys the parachute."""
    parachute = Parachute(
        name="Drogue", cd_s=1.0, trigger="apogee", sampling_rate=100, lag=0
    )

    assert _check_apogee_trigger(parachute, 0.0, -3.0) is True


def test_the_apogee_trigger_fires_at_once_for_a_rocket_dropped_from_rest():
    """Released at rest above the ground, as from a helicopter or a balloon,
    the rocket is at its apogee at the first check."""
    parachute = Parachute(
        name="Drogue", cd_s=1.0, trigger="apogee", sampling_rate=100, lag=0
    )

    assert _check_apogee_trigger(parachute, 0.0, 0.0, height=1000.0) is True


def test_the_apogee_trigger_does_not_fire_for_a_rocket_at_rest_on_the_pad():
    """On the pad the rocket is not moving and not above the ground, and
    nothing deploys."""
    parachute = Parachute(
        name="Drogue", cd_s=1.0, trigger="apogee", sampling_rate=100, lag=0
    )

    assert _check_apogee_trigger(parachute, 0.0, 0.0, height=0.0) is False
    assert _check_apogee_trigger(parachute, 0.1, 2.0, height=0.1) is False


def test_the_apogee_trigger_is_the_one_an_event_uses():
    parachute = Parachute(
        name="Drogue", cd_s=1.0, trigger="apogee", sampling_rate=100, lag=0
    )

    assert (
        parachute.triggerfunc is Event(lambda context: None, trigger="apogee").trigger
    )
