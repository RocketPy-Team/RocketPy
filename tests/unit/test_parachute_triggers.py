import numpy as np

from rocketpy.rocket.parachute import Parachute
from rocketpy.simulation.flight import Flight


def test_trigger_receives_u_dot():
    def derivative_func(_t, _y):
        return np.array([0, 0, 0, 1.0, 2.0, 3.0, 0, 0, 0, 0, 0, 0, 0])

    recorded = {}

    def user_trigger(_p, _h, _y, u_dot):
        recorded["u_dot"] = np.array(u_dot)
        return True

    parachute = Parachute(
        name="test",
        cd_s=1.0,
        trigger=user_trigger,
        sampling_rate=100,
    )

    dummy = type("D", (), {})()

    res = Flight._evaluate_parachute_trigger(
        dummy,
        parachute,
        pressure=0.0,
        height=10.0,
        y=np.zeros(13),
        sensors=[],
        derivative_func=derivative_func,
        t=0.0,
    )

    assert res is True
    assert "u_dot" in recorded
    assert np.allclose(recorded["u_dot"][3:6], np.array([1.0, 2.0, 3.0]))


def test_trigger_with_u_dot_only():
    """Test trigger that only expects u_dot (no sensors)."""

    def derivative_func(_t, _y):
        return np.array([0, 0, 0, -1.0, -2.0, -3.0, 0, 0, 0, 0, 0, 0, 0])

    recorded = {}

    def user_trigger(_p, _h, _y, u_dot):
        recorded["u_dot"] = np.array(u_dot)
        return False

    parachute = Parachute(
        name="test_u_dot_only",
        cd_s=1.0,
        trigger=user_trigger,
        sampling_rate=100,
    )

    dummy = type("D", (), {})()

    res = Flight._evaluate_parachute_trigger(
        dummy,
        parachute,
        pressure=0.0,
        height=5.0,
        y=np.zeros(13),
        sensors=[],
        derivative_func=derivative_func,
        t=1.234,
    )

    assert res is False
    assert "u_dot" in recorded
    assert np.allclose(recorded["u_dot"][3:6], np.array([-1.0, -2.0, -3.0]))


def test_basic_trigger_does_not_compute_u_dot():
    def derivative_func(_t, _y):
        raise RuntimeError("derivative should not be called for legacy triggers")

    called = {}

    def basic_trigger(_p, _h, _y):
        called["ok"] = True
        return True

    parachute = Parachute(
        name="basic",
        cd_s=1.0,
        trigger=basic_trigger,
        sampling_rate=100,
    )

    dummy = type("D", (), {})()

    res = Flight._evaluate_parachute_trigger(
        dummy,
        parachute,
        pressure=0.0,
        height=0.0,
        y=np.zeros(13),
        sensors=[],
        derivative_func=derivative_func,
        t=0.0,
    )

    assert res is True
    assert called.get("ok", False) is True
