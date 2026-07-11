"""Tests for the reworked stability model: the aerodynamic center, the
cp_position alias, the reconstructed nonlinear center of pressure, and the
aggregate aerodynamic coefficients."""

import numpy as np
import pytest


def test_cp_position_alias_matches_aerodynamic_center(calisto_robust):
    """``cp_position`` is a plain alias of ``aerodynamic_center`` (no warning)."""
    rocket = calisto_robust
    assert rocket.cp_position.get_value_opt(0.3) == pytest.approx(
        rocket.aerodynamic_center.get_value_opt(0.3)
    )


def test_reconstructed_center_of_pressure_converges_to_aerodynamic_center(
    calisto_robust,
):
    """The nonlinear center of pressure, reconstructed from the aggregate
    coefficients as ``x_cdm + csys * d * Cm / CN``, converges to the linear
    aerodynamic center as the angle of attack goes to zero. (The singular
    nonlinear CP is no longer a blessed method; this is its documented
    reconstruction path.)"""
    rocket = calisto_robust
    mach = 0.3
    aerodynamic_center = rocket.aerodynamic_center.get_value_opt(mach)
    csys = rocket._csys
    diameter = 2 * rocket.radius
    cdm = rocket.center_of_dry_mass_position

    coeffs = rocket.aerodynamic_coefficients_full(np.radians(0.1), 0.0, mach)
    reconstructed_cp = cdm + csys * diameter * coeffs["cm"] / coeffs["cL"]
    assert reconstructed_cp == pytest.approx(aerodynamic_center, abs=1e-3)


def test_aerodynamic_coefficients_normal_force_grows_with_alpha(calisto_robust):
    """Total normal-force coefficient increases with angle of attack and is zero
    at zero incidence; the returned dict exposes normal force and pitch moment."""
    rocket = calisto_robust
    coeffs = rocket.aerodynamic_coefficients(np.radians(5), 0.0, 0.3)
    assert set(coeffs) == {"normal_force", "pitch_moment"}

    cn_2 = rocket.aerodynamic_coefficients(np.radians(2), 0.0, 0.3)["normal_force"]
    cn_8 = rocket.aerodynamic_coefficients(np.radians(8), 0.0, 0.3)["normal_force"]
    assert cn_8 > cn_2 > 0


def test_axisymmetric_rocket_planes_coincide(calisto_robust):
    """An axisymmetric rocket has matching pitch and yaw aerodynamic centers."""
    rocket = calisto_robust
    assert rocket.is_axisymmetric
    for mach in (0.0, 0.5, 1.0):
        assert rocket.aerodynamic_center.get_value_opt(mach) == pytest.approx(
            rocket.aerodynamic_center_yaw.get_value_opt(mach)
        )


def test_aerodynamic_coefficients_full_signed_set(calisto_robust):
    """The full rocket coefficient set returns all six signed coefficients;
    lift grows with alpha, drag comes from the vehicle drag curve, and the pitch
    moment is restoring (negative) for a stable rocket."""
    rocket = calisto_robust
    coeffs = rocket.aerodynamic_coefficients_full(np.radians(5), 0.0, 0.3)
    assert set(coeffs) == {"cL", "cQ", "cD", "cm", "cn", "cl"}

    low = rocket.aerodynamic_coefficients_full(np.radians(2), 0.0, 0.3)
    assert coeffs["cL"] > low["cL"] > 0
    assert coeffs["cm"] < 0  # restoring pitch moment about the center of dry mass
    assert coeffs["cD"] == pytest.approx(
        rocket.power_off_drag_by_mach.get_value_opt(0.3)
    )


def test_add_vehicle_aerodynamic_surface(calisto_robust):
    """A supplied full-vehicle coefficient set is added as a single generic
    surface and contributes to the rocket aggregate (rocket-as-GenericSurface)."""
    rocket = calisto_robust
    base_cl = rocket.aerodynamic_coefficients_full(np.radians(5), 0.0, 0.3)["cL"]
    n_before = len(rocket.aerodynamic_surfaces)

    surface = rocket.add_vehicle_aerodynamic_surface(
        coefficients={"cL": lambda a, b, m, re, p, q, r: 2.0 * a}
    )

    assert len(rocket.aerodynamic_surfaces) == n_before + 1
    # The vehicle surface exposes the uniform coefficient accessors.
    assert surface.cL(np.radians(5), 0, 0.3, 0, 0, 0, 0) == pytest.approx(
        2.0 * np.radians(5)
    )
    # Its lift adds to the rocket aggregate.
    new_cl = rocket.aerodynamic_coefficients_full(np.radians(5), 0.0, 0.3)["cL"]
    assert new_cl > base_cl
