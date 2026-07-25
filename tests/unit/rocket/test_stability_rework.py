"""Tests for the reworked stability model: the aerodynamic center, the
cp_position alias, the reconstructed nonlinear center of pressure, and the
aggregate aerodynamic coefficients."""

import numpy as np
import pytest

from rocketpy import Function, GenericSurface, LinearGenericSurface, Rocket


def _full_body_surface(rocket, coefficients, name="Full Body Aerodynamics", **kwargs):
    """Build a full-body GenericSurface referenced to the rocket dimensions,
    the way a user would before handing it to ``add_full_body_aerodynamics``."""
    return GenericSurface(
        reference_area=rocket.area,
        reference_length=2 * rocket.radius,
        coefficients=coefficients,
        name=name,
        **kwargs,
    )


def test_cp_position_alias_matches_aerodynamic_center(calisto_robust):
    """``cp_position`` is a plain alias of ``aerodynamic_center`` (no warning)."""
    rocket = calisto_robust
    assert rocket.cp_position.get_value_opt(0.3) == pytest.approx(
        rocket.aerodynamic_center.get_value_opt(0.3)
    )


def test_length_spans_nose_tip_to_aft_surface(calisto_robust):
    """The overall length runs from the nose tip to the aft-most surface, and
    does not depend on the coordinate-system orientation."""
    rocket = calisto_robust
    # Nose tip at z = 1.160; tail base at z = -1.313 - 0.060 = -1.373.
    assert rocket.length == pytest.approx(1.160 - (-1.373), abs=1e-9)


def test_length_orientation_independent(calisto_robust, calisto_nose_to_tail):
    """The same physical rocket has the same length in either orientation."""
    # Mirror of calisto_robust: nose tip at z=0, then every surface reference
    # sits at the same distance from the nose tip as in calisto_robust, so the
    # physical rocket (and its length) is identical. The aft-most point is the
    # tail base at z = 2.473 + 0.060 = 2.533.
    calisto_nose_to_tail.add_nose(length=0.55829, kind="vonKarman", position=0.0)
    calisto_nose_to_tail.add_tail(
        top_radius=0.0635, bottom_radius=0.0435, length=0.060, position=2.473
    )
    calisto_nose_to_tail.add_trapezoidal_fins(
        n=4, root_chord=0.120, tip_chord=0.040, span=0.100, position=2.328
    )
    assert calisto_nose_to_tail.length == pytest.approx(calisto_robust.length, abs=1e-9)


def test_length_extends_to_nozzle_past_surfaces(calisto_robust, cesaroni_m1670):
    """When the motor nozzle extends aft of the last aerodynamic surface, the
    length runs from the nose tip to the nozzle rather than to the surface."""
    rocket = calisto_robust
    # Move the motor aft so its nozzle (at the motor origin, z = -2.0 in the
    # rocket frame) sits past the tail base at z = -1.373.
    rocket.add_motor(cesaroni_m1670, position=-2.0)
    assert rocket.nozzle_position == pytest.approx(-2.0, abs=1e-9)
    assert rocket.length == pytest.approx(1.160 - (-2.0), abs=1e-9)


def test_length_requires_a_surface(calisto):
    """A rocket with no aerodynamic surfaces has no defined length."""
    with pytest.raises(ValueError, match="at least one aerodynamic surface"):
        _ = calisto.length


def test_axisymmetric_rocket_planes_coincide(calisto_robust):
    """An axisymmetric rocket has matching pitch and yaw aerodynamic centers."""
    rocket = calisto_robust
    assert rocket.is_axisymmetric
    for mach in (0.0, 0.5, 1.0):
        assert rocket.aerodynamic_center.get_value_opt(mach) == pytest.approx(
            rocket.aerodynamic_center_yaw.get_value_opt(mach)
        )


def test_add_full_body_aerodynamics(calisto_robust):
    """A prebuilt full-body surface is added and contributes to the rocket
    aggregate (rocket-as-GenericSurface)."""
    rocket = calisto_robust
    base_slope = rocket.total_lift_coeff_der.get_value_opt(0.3)
    n_before = len(rocket.aerodynamic_surfaces)

    surface = _full_body_surface(rocket, {"cN": lambda a, b, m, re, p, q, r: 2.0 * a})
    returned = rocket.add_full_body_aerodynamics(surface)

    assert returned is surface
    assert len(rocket.aerodynamic_surfaces) == n_before + 1
    # A single surface is active during the whole flight by default.
    assert surface.active_during == "always"
    # The full-body surface exposes the uniform coefficient accessors.
    assert surface.cN(np.radians(5), 0, 0.3, 0, 0, 0, 0) == pytest.approx(
        2.0 * np.radians(5)
    )
    # Its normal-force slope adds to the rocket aggregate lift-curve slope.
    assert rocket.total_lift_coeff_der.get_value_opt(0.3) > base_slope


def test_add_full_body_aerodynamics_linear_surface_with_damping(calisto_robust):
    """A LinearGenericSurface stability-derivative set (with pitch and roll
    damping) is accepted as a full-body model, and its damping derivatives
    stay inspectable as named attributes."""
    rocket = calisto_robust
    surface = LinearGenericSurface(
        reference_area=rocket.area,
        reference_length=2 * rocket.radius,
        coefficients={
            "cN_alpha": 2.0,
            "cm_alpha": -1.0,
            "cm_q": -50.0,
            "cl_p": -5.0,
        },
        name="Full body derivatives",
    )
    # overwrite clears the built-in drag even though this surface carries none.
    with pytest.warns(UserWarning, match="were cleared"):
        created = rocket.add_full_body_aerodynamics(surface, overwrite=True)

    assert created is surface
    assert len(rocket.aerodynamic_surfaces) == 1
    assert surface.cm_q.get_value_opt(0, 0, 0.3, 0, 0, 0, 0) == pytest.approx(-50.0)
    # The surface has no drag coefficient, so the rocket now has no drag at all.
    assert rocket.power_off_drag_by_mach.get_value_opt(0.3) == pytest.approx(0.0)
    assert rocket.power_on_drag_by_mach.get_value_opt(0.3) == pytest.approx(0.0)


def test_to_surface_round_trips(calisto_robust, cesaroni_m1670):
    """``to_surface`` lumps the whole rocket into a power-off/power-on pair of
    :class:`LinearGenericSurface` (carrying the pitch/yaw/roll damping and each
    phase's drag) that reproduces the rocket's stability when added to a bare
    rocket -- the inverse of ``add_full_body_aerodynamics``."""
    surfaces = calisto_robust.to_surface()
    assert isinstance(surfaces, list) and len(surfaces) == 2
    power_off, power_on = surfaces
    assert all(isinstance(s, LinearGenericSurface) for s in surfaces)
    assert power_off.active_during == "power_off"
    assert power_on.active_during == "power_on"
    # The rate damping is captured as named, inspectable derivatives.
    assert power_off.cm_q.get_value_opt(0, 0, 0.3, 0, 0, 0, 0) < 0  # pitch damping
    assert power_off.cl_p.get_value_opt(0, 0, 0.3, 0, 0, 0, 0) < 0  # roll damping
    # Each surface carries its own phase's drag as the axial coefficient.
    assert power_off.cA.get_value_opt(0, 0, 0.3, 0, 0, 0, 0) == pytest.approx(
        calisto_robust.power_off_drag_by_mach.get_value_opt(0.3), rel=1e-6
    )
    assert power_on.cA.get_value_opt(0, 0, 0.3, 0, 0, 0, 0) == pytest.approx(
        calisto_robust.power_on_drag_by_mach.get_value_opt(0.3), rel=1e-6
    )

    # A bare rocket (same body and motor) carrying only the lumped pair
    # reproduces the modeled rocket's static margin and aerodynamic center. The
    # surfaces carry the drag, so overwrite clears the bare rocket's curves.
    bare = Rocket(
        radius=0.0635,
        mass=14.426,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag="data/rockets/calisto/powerOffDragCurve.csv",
        power_on_drag="data/rockets/calisto/powerOnDragCurve.csv",
        center_of_mass_without_motor=0,
        coordinate_system_orientation="tail_to_nose",
    )
    bare.add_motor(cesaroni_m1670, position=-1.373)
    with pytest.warns(UserWarning, match="were cleared"):
        bare.add_full_body_aerodynamics(surfaces, overwrite=True)

    assert bare.power_off_drag_by_mach.get_value_opt(0.3) == pytest.approx(0.0)
    assert bare.power_on_drag_by_mach.get_value_opt(0.3) == pytest.approx(0.0)
    assert bare.static_margin(0) == pytest.approx(
        calisto_robust.static_margin(0), rel=1e-3
    )
    for mach in (0.3, 0.8, 1.5):
        assert bare.aerodynamic_center.get_value_opt(mach) == pytest.approx(
            calisto_robust.aerodynamic_center.get_value_opt(mach), abs=1e-3
        )


def test_to_coefficients_returns_phase_pair(calisto_robust):
    """``to_coefficients`` returns a ``power_off``/``power_on`` pair of coefficient
    sets, each a dict of Mach curves. The stability derivatives are the same in
    both; only the drag ``cA_0`` differs by motor phase."""
    coeffs = calisto_robust.to_coefficients()
    assert set(coeffs) == {"power_off", "power_on"}
    body_keys = {
        "cN_alpha",
        "cm_alpha",
        "cN_q",
        "cm_q",
        "cY_beta",
        "cn_beta",
        "cY_r",
        "cn_r",
        "cl_p",
        "cA_0",
    }
    assert set(coeffs["power_off"]) == body_keys
    assert set(coeffs["power_on"]) == body_keys
    assert all(isinstance(c, Function) for c in coeffs["power_off"].values())

    # Stability derivatives are identical between phases; only drag differs.
    assert coeffs["power_on"]["cN_alpha"].get_value_opt(0.3) == pytest.approx(
        coeffs["power_off"]["cN_alpha"].get_value_opt(0.3)
    )
    assert coeffs["power_off"]["cA_0"].get_value_opt(0.3) == pytest.approx(
        calisto_robust.power_off_drag_by_mach.get_value_opt(0.3), rel=1e-6
    )
    assert coeffs["power_on"]["cA_0"].get_value_opt(0.3) == pytest.approx(
        calisto_robust.power_on_drag_by_mach.get_value_opt(0.3), rel=1e-6
    )

    # It is exactly what ``to_surface`` wraps into surfaces.
    power_off, _ = calisto_robust.to_surface()
    assert power_off.cN_alpha.get_value_opt(0, 0, 0.3, 0, 0, 0, 0) == pytest.approx(
        coeffs["power_off"]["cN_alpha"].get_value_opt(0.3)
    )

    # Wind naming flows through to each phase set.
    wind = calisto_robust.to_coefficients(force_convention="wind")
    assert "cL_alpha" in wind["power_off"] and "cD_0" in wind["power_off"]
    with pytest.raises(ValueError, match="force_convention"):
        calisto_robust.to_coefficients(force_convention="bogus")


def test_to_surface_force_convention(calisto_robust):
    """``force_convention`` selects the coefficient naming (body ``cN``/``cA`` vs
    wind ``cL``/``cD``) while yielding an equivalent surface pair."""
    args = (0, 0, 0.3, 0, 0, 0, 0)
    drag = calisto_robust.power_off_drag_by_mach.get_value_opt(0.3)

    body_off, _ = calisto_robust.to_surface(force_convention="body")
    wind_off, _ = calisto_robust.to_surface(force_convention="wind")
    assert body_off.force_convention == "body"
    assert wind_off.force_convention == "wind"
    # Same underlying model either way (both store body-frame derivatives).
    assert wind_off.cN_alpha.get_value_opt(*args) == pytest.approx(
        body_off.cN_alpha.get_value_opt(*args)
    )
    # The body axial and wind drag both equal the phase's drag.
    assert body_off.cA.get_value_opt(*args) == pytest.approx(drag, rel=1e-6)
    assert wind_off.cD.get_value_opt(*args) == pytest.approx(drag, rel=1e-6)

    with pytest.raises(ValueError, match="force_convention"):
        calisto_robust.to_surface(force_convention="bogus")


def test_add_full_body_aerodynamics_power_on_off(calisto_robust):
    """A power-on/power-off pair, passed as a list, adds two phase-gated
    full-body surfaces."""
    rocket = calisto_robust
    n_before = len(rocket.aerodynamic_surfaces)

    power_on = _full_body_surface(
        rocket, {"cD": 0.3}, name="Full body (power on)", active_during="power_on"
    )
    power_off = _full_body_surface(
        rocket, {"cD": 0.5}, name="Full body (power off)", active_during="power_off"
    )
    created = rocket.add_full_body_aerodynamics([power_on, power_off])

    assert len(created) == 2
    assert len(rocket.aerodynamic_surfaces) == n_before + 2
    assert created[0].active_during == "power_on"
    assert created[1].active_during == "power_off"


def test_add_full_body_aerodynamics_single_phase(calisto_robust):
    """A single phase-gated surface (e.g. base drag after burnout) may be added
    on its own."""
    rocket = calisto_robust
    surface = _full_body_surface(rocket, {"cD": 0.5}, active_during="power_off")
    created = rocket.add_full_body_aerodynamics(surface)
    assert created is surface
    assert surface.active_during == "power_off"


def test_add_full_body_aerodynamics_overwrite_replaces_surfaces_and_drag(
    calisto_robust,
):
    """overwrite=True removes existing surfaces and clears both built-in drag
    curves; the supplied surface then provides the only drag."""
    rocket = calisto_robust
    assert len(rocket.aerodynamic_surfaces) > 1
    assert rocket.power_off_drag_by_mach.get_value_opt(0.3) > 0

    surface = _full_body_surface(
        rocket, {"cA": 0.4, "cN": lambda a, b, m, re, p, q, r: 2.0 * a}
    )
    with pytest.warns(UserWarning, match="were cleared"):
        rocket.add_full_body_aerodynamics(surface, overwrite=True)

    # Only the full-body surface remains.
    assert len(rocket.aerodynamic_surfaces) == 1
    assert rocket.aerodynamic_surfaces[0].component is surface
    # Both built-in drag curves were cleared (the surface carries the drag now).
    assert rocket.power_off_drag_by_mach.get_value_opt(0.3) == pytest.approx(0.0)
    assert rocket.power_on_drag_by_mach.get_value_opt(0.3) == pytest.approx(0.0)


def test_add_full_body_aerodynamics_overwrite_always_clears_drag(calisto_robust):
    """overwrite=True clears both built-in drag curves unconditionally, even for
    a phase-gated surface that carries no drag of its own."""
    rocket = calisto_robust
    assert rocket.power_off_drag_by_mach.get_value_opt(0.3) > 0
    assert rocket.power_on_drag_by_mach.get_value_opt(0.3) > 0

    power_on = _full_body_surface(rocket, {"cD": 0.3}, active_during="power_on")
    power_off = _full_body_surface(rocket, {"cN": 1.0}, active_during="power_off")
    with pytest.warns(UserWarning, match="were cleared"):
        rocket.add_full_body_aerodynamics([power_on, power_off], overwrite=True)

    # Both curves are cleared regardless of phase or whether a surface carries
    # drag; the supplied surfaces are the complete aerodynamics.
    assert rocket.power_on_drag_by_mach.get_value_opt(0.3) == pytest.approx(0.0)
    assert rocket.power_off_drag_by_mach.get_value_opt(0.3) == pytest.approx(0.0)


def test_add_full_body_aerodynamics_overwrite_warns_on_later_add(calisto_robust):
    """After an overwrite, adding another surface warns that it stacks on the
    full-body model."""
    rocket = calisto_robust
    surface = _full_body_surface(rocket, {"cD": 0.4})
    with pytest.warns(UserWarning):  # the overwrite itself warns about drag
        rocket.add_full_body_aerodynamics(surface, overwrite=True)

    later = _full_body_surface(rocket, {"cN": 1.0})
    with pytest.warns(UserWarning, match="stacks on|summed on top"):
        rocket.add_full_body_aerodynamics(later)
