"""Atmosphere transfer uses actual reference-flight conditions, not ISA."""

from types import SimpleNamespace

import numpy as np
import pytest

from rocketpy.rocket.datcom import datcom_decks


def _flight(rocket, density=1.5):
    stop = rocket.motor.burn_time[1]
    return SimpleNamespace(
        rocket=rocket,
        apogee_time=2 * stop,
        mach_number=lambda t: np.where(t < stop, t / stop, 2 - t / stop),
        z=lambda t: 1000 + t * 100,
        partial_angle_of_attack=lambda t: np.zeros_like(t),
        env=SimpleNamespace(
            density=lambda z: np.full_like(z, density),
            speed_of_sound=lambda z: np.full_like(z, 300),
            dynamic_viscosity=lambda z: np.full_like(z, 2e-5),
            pressure=lambda z: np.full_like(z, 90000),
            temperature=lambda z: np.full_like(z, 280),
        ),
    )


@pytest.mark.parametrize("density", [0.8, 1.5])
def test_set_datcom_atmosphere_actual_environment(datcom_rocket, density):
    """Both native phases use rho*a*M/mu, including unreached Mach values."""
    # Arrange
    flight = _flight(datcom_rocket, density)
    grid = [0.1, 0.5, 2.0]
    # Act
    report = datcom_rocket.set_datcom_atmosphere(flight, mach=grid)
    decks, _, _, _ = datcom_decks(datcom_rocket)
    # Assert
    for state in ("power_on", "power_off"):
        assert report["reynolds"][state] == pytest.approx(
            density * 300 * np.array(grid) / 2e-5
        )
        assert report["temperature_k"][state] == pytest.approx([280] * 3)
        assert report["pressure_pa"][state] == pytest.approx([90000] * 3)
    assert all("REN =" in deck and "ALT =" not in deck for deck in decks.values())


def test_set_datcom_atmosphere_failure_preserves_options(datcom_rocket):
    """An invalid atmosphere cannot partially overwrite the native settings."""
    # Arrange
    fins = [
        s
        for s, _, _ in datcom_rocket.aerodynamic_surfaces
        if hasattr(s, "datcom_options")
    ]
    before = [dict(s.datcom_options) for s in fins]
    flight = _flight(datcom_rocket, density=-1)
    # Act / Assert
    with pytest.raises(ValueError, match="positive Reynolds"):
        datcom_rocket.set_datcom_atmosphere(flight)
    assert [s.datcom_options for s in fins] == before


def test_set_datcom_atmosphere_separates_power_phases(datcom_rocket):
    """The same Mach at different altitudes has different native Reynolds values."""
    # Arrange
    flight = _flight(datcom_rocket)
    flight.env.density = lambda z: 2 - z / 10000
    stop = datcom_rocket.motor.burn_time[1]
    # Act
    report = datcom_rocket.set_datcom_atmosphere(flight, mach=[0.25, 0.5], samples=1001)
    # Assert
    for phase, t in [("power_on", 0.5 * stop), ("power_off", 1.5 * stop)]:
        expected = (2 - (1000 + 100 * t) / 10000) * 300 * 0.5 / 2e-5
        assert report["reynolds"][phase][1] == pytest.approx(expected)
    assert report["reynolds"]["power_on"][1] > report["reynolds"]["power_off"][1]


def test_set_datcom_atmosphere_requires_both_phases(datcom_rocket):
    """A reference ending before burnout cannot supply a coast atmosphere."""
    # Arrange
    flight = _flight(datcom_rocket)
    flight.apogee_time = datcom_rocket.motor.burn_time[1] / 2
    # Act / Assert
    with pytest.raises(ValueError, match="no power_off samples"):
        datcom_rocket.set_datcom_atmosphere(flight)


@pytest.mark.parametrize("peak", [0.0, 0.2, 0.95, 1.5, 3.0, 8.0])
def test_adaptive_mach_grid_native_limit_and_regime(peak):
    """Every flight regime gets 20 distinct points, fixed anchors and headroom."""
    # Arrange
    from rocketpy.rocket.datcom_conditions import adaptive_mach_grid

    # Act
    grid = adaptive_mach_grid(peak)
    # Assert
    assert len(grid) == 20
    assert 0.1 in grid and 1.0 in grid
    assert grid[0] == pytest.approx(0.01)
    assert grid[-1] >= peak * 1.15
    assert np.diff(grid).min() >= 0.0099
    assert np.count_nonzero((grid >= 0.8) & (grid <= 1.2)) >= 9


def test_set_datcom_atmosphere_adapts_mach_and_alpha(datcom_rocket):
    """Updating from a hotter trajectory expands both grids and rebuilds REN."""
    # Arrange
    flight = _flight(datcom_rocket)
    original_mach = flight.mach_number
    flight.mach_number = lambda t: 3 * original_mach(t)
    flight.partial_angle_of_attack = lambda t: 40 * original_mach(t)
    # Act
    report = datcom_rocket.set_datcom_atmosphere(flight)
    decks, _, _, _ = datcom_decks(datcom_rocket)
    # Assert
    assert len(report["mach"]) == 20
    assert report["mach"][-1] >= 3.45
    assert 0.1 in report["mach"] and 1 in report["mach"]
    assert report["alpha"][-1] >= 48
    assert report["alpha"][0] == pytest.approx(-report["alpha"][-1])
    assert 0 in report["alpha"]
    for phase in ("power_on", "power_off"):
        assert len(report["reynolds"][phase]) == 20
        assert report["coverage"]["phases"][phase]["mach_max"] > 2.9
    assert all("NMACH = 20." in deck for deck in decks.values())


def test_set_datcom_atmosphere_ignores_unloaded_incidence(datcom_rocket):
    """An undefined pad-flow direction cannot force a tail-first alpha table."""
    # Arrange
    flight = _flight(datcom_rocket)
    flight.partial_angle_of_attack = lambda t: np.where(t == 0, 180, 3)
    # Act
    report = datcom_rocket.set_datcom_atmosphere(flight)
    # Assert
    assert report["alpha"][-1] == pytest.approx(20)
    assert report["coverage"]["excluded_low_q_samples"] > 0


def test_set_datcom_atmosphere_loaded_tail_first_rejected(datcom_rocket):
    """A tumbling ascent cannot silently fold loaded tail-first flow."""
    # Arrange
    flight = _flight(datcom_rocket)
    flight.mach_number = lambda t: np.full_like(t, 0.5)
    flight.partial_angle_of_attack = lambda t: np.full_like(t, 180)
    # Act / Assert
    with pytest.raises(ValueError, match="tail-first"):
        datcom_rocket.set_datcom_atmosphere(flight)


def test_set_datcom_atmosphere_explicit_alpha_preserved(datcom_rocket):
    """Explicit native grids remain available for controlled tunnel conditions."""
    # Arrange
    flight = _flight(datcom_rocket)
    # Act
    report = datcom_rocket.set_datcom_atmosphere(
        flight, mach=[0.1, 1.0], alpha=[-5, 0, 5]
    )
    # Assert
    assert report["alpha"] == [-5, 0, 5]
    assert report["mach"] == [0.1, 1.0]


def test_datcom_provisional_grid_uses_twenty_points():
    """Even before a reference flight, the default uses all native Mach slots."""
    # Arrange
    from rocketpy.rocket.datcom import _flight_conditions

    # Act
    mach, _, _, _ = _flight_conditions({})
    # Assert
    assert len(mach) == 20
    assert 0.1 in mach and 1.0 in mach
