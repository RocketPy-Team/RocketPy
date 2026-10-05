"""Tests for the reworked stability model: the aerodynamic center, the
cp_position alias, the reconstructed nonlinear center of pressure, and the
aggregate aerodynamic coefficients."""

import json
import math
import warnings
from unittest.mock import patch

import numpy as np
import pytest

from rocketpy import (
    ControllableGenericSurface,
    Function,
    GenericSurface,
    LinearGenericSurface,
    PointMassRocket,
    Rocket,
)
from rocketpy._encoders import RocketPyDecoder, RocketPyEncoder
from rocketpy.rocket.aero_surface.aero_coefficient import AeroCoefficient
from rocketpy.rocket._helpers import (
    aerodynamic_damping,
    corrective_and_damping_moments,
    lateral_inertia_and_rate,
    stability_margin_and_slope,
    stability_surfaces,
    damping_derivative,
    is_incidence_linear,
    neutral_point_and_slope,
    summed_force_and_moment,
)
from rocketpy.rocket.aero_surface.fins.trapezoidal_fin import TrapezoidalFin


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


def test_length_needs_both_ends(calisto, calisto_nose_cone, calisto_tail):
    """The length is only measured when a nose cone marks the front and a
    tail, a fin set or a motor nozzle marks the back. A generic surface marks
    neither. Otherwise it is ``None``, unless it was given."""
    bare = Rocket(
        radius=0.0635,
        mass=14.426,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag=0.5,
        power_on_drag=0.5,
        center_of_mass_without_motor=0,
    )
    assert bare.length is None
    bare.add_surfaces(
        GenericSurface(bare.area, 2 * bare.radius, {"cN": lambda alpha: 2 * alpha}),
        (0, 0, 0),
    )
    assert bare.length is None
    bare.add_surfaces(calisto_tail, -1.313)
    assert bare.length is None  # no nose cone
    bare.add_surfaces(calisto_nose_cone, 1.160)
    assert bare.length == pytest.approx(1.160 - (-1.373), abs=1e-9)

    # A motor and a generic surface alone: the nozzle is known, the nose is not
    calisto.add_surfaces(
        GenericSurface(
            calisto.area, 2 * calisto.radius, {"cN": lambda alpha: 2 * alpha}
        ),
        (0, 0, 0),
    )
    assert calisto.length is None
    # The nose cone and the motor nozzle are enough
    calisto.add_surfaces(calisto_nose_cone, 1.160)
    assert calisto.length == pytest.approx(1.160 - calisto.nozzle_position)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calisto.prints.rocket_aerodynamics_quantities()


def test_given_length_is_used_without_surfaces(capsys):
    """A length given at construction is reported even when the rocket has no
    aerodynamic surface to measure it from."""
    rocket = Rocket(
        radius=0.0635,
        mass=14.426,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag=0.5,
        power_on_drag=0.5,
        center_of_mass_without_motor=0,
        length=2.5,
    )
    rocket.prints.rocket_aerodynamics_quantities()
    assert "Rocket Length: 2.500 m" in capsys.readouterr().out


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


@pytest.mark.parametrize("orientation", ["tail_to_nose", "nose_to_tail"])
@pytest.mark.parametrize("cp_z", [-0.3, 0.3])
def test_generic_surface_cp_offset_matches_flight_neutral_point(orientation, cp_z):
    """A generic surface's ``center_of_pressure`` must move the aerodynamic
    center to the same point the flight applies the force at (the neutral
    point), in both planes and for both coordinate system orientations."""
    rocket = Rocket(
        radius=0.05,
        mass=10,
        inertia=(5, 5, 0.02),
        power_off_drag=0.5,
        power_on_drag=0.5,
        center_of_mass_without_motor=0,
        coordinate_system_orientation=orientation,
    )
    surface = LinearGenericSurface(
        reference_area=rocket.area,
        reference_length=2 * rocket.radius,
        coefficients={"cN_alpha": 2.0, "cY_beta": -2.0},
        center_of_pressure=(0, 0, cp_z),
    )
    rocket.add_surfaces(surface, 0.1)

    # The center of pressure is given along the body z-axis (toward the nose)
    csys = 1 if orientation == "tail_to_nose" else -1
    expected = 0.1 + csys * cp_z
    assert rocket.aerodynamic_center(0.3) == pytest.approx(expected)
    assert rocket.aerodynamic_center_yaw(0.3) == pytest.approx(expected)
    assert rocket.neutral_point(0.0, 0.3) == pytest.approx(expected)
    assert rocket.neutral_point_yaw(0.0, 0.3) == pytest.approx(expected)


def _rocket_with(surface, position):
    """A small rocket of built-in surfaces plus one generic surface."""
    rocket = Rocket(
        radius=0.0635,
        mass=14.426,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag=0.5,
        power_on_drag=0.5,
        center_of_mass_without_motor=0,
        coordinate_system_orientation="tail_to_nose",
    )
    rocket.add_nose(length=0.55829, kind="vonkarman", position=1.278)
    rocket.add_trapezoidal_fins(
        n=4, span=0.100, root_chord=0.120, tip_chord=0.040, position=-1.04956
    )
    rocket.add_surfaces(surface, position)
    return rocket


def _margin_at_the_true_state(rocket, alpha, beta, plane):
    """Margin of one plane with both flow angles at their real values."""
    neutral_point, _ = neutral_point_and_slope(rocket, alpha, beta, 0.3, plane)
    center_of_mass = rocket.center_of_mass.get_value_opt(0.0)
    return rocket._csys * (center_of_mass - neutral_point) / (2 * rocket.radius)


@pytest.mark.parametrize(
    "surface, position",
    [
        # A pure couple: a moment slope with no force slope
        (
            LinearGenericSurface(math.pi * 0.0635**2, 0.127, {"cm_alpha": -1.5}),
            0.5,
        ),
        (
            LinearGenericSurface(math.pi * 0.0635**2, 0.127, {"cn_beta": 1.5}),
            0.5,
        ),
        # An axial force that changes with the angle, at a sideways offset
        (
            LinearGenericSurface(
                math.pi * 0.0635**2,
                0.127,
                {"cN_alpha": 1.0, "cA_alpha": 3.0},
                center_of_pressure=(0, 0.05, 0),
            ),
            (0, 0, 0.5),
        ),
        (
            LinearGenericSurface(
                math.pi * 0.0635**2,
                0.127,
                {"cY_beta": -1.0, "cA_beta": 3.0},
                center_of_pressure=(0.05, 0, 0),
            ),
            (0, 0, 0.5),
        ),
        # A canted individual fin: part of its force acts along the axis
        (
            TrapezoidalFin(
                angular_position=30,
                span=0.06,
                root_chord=0.08,
                tip_chord=0.04,
                rocket_radius=0.0635,
                cant_angle=3,
            ),
            0.9,
        ),
    ],
)
def test_aerodynamic_center_counts_the_whole_moment_of_each_surface(surface, position):
    """The aerodynamic center is the neutral point at zero angle, taken from the
    rocket's summed forces and moments, even for a surface whose moment is not
    its force slope times an arm along the axis."""
    rocket = _rocket_with(surface, position)
    for mach in (0.0, 0.3, 0.9):
        for plane, center in (
            ("pitch", rocket.aerodynamic_center),
            ("yaw", rocket.aerodynamic_center_yaw),
        ):
            expected, _ = neutral_point_and_slope(rocket, 0.0, 0.0, mach, plane)
            assert center.get_value_opt(mach) == pytest.approx(expected, abs=1e-6)


def test_margin_of_a_non_axisymmetric_rocket_follows_the_angle_of_its_plane():
    """A rocket that is not axisymmetric is analysed one plane at a time: its
    pitch margin goes with the angle of attack and its yaw margin with the
    sideslip angle. Reading both at the total angle of attack is wrong: in a
    pure sideslip it reports the pitch plane of this rocket as unstable while
    it is stable."""
    area, diameter = math.pi * 0.0635**2, 2 * 0.0635
    canards = GenericSurface(
        area,
        diameter,
        {"cN": lambda alpha: 3.0 * alpha + 60.0 * alpha**3, "cY": 0},
        force_convention="body",
    )
    rocket = _rocket_with(canards, 0.8)
    assert not rocket.is_axisymmetric

    alpha, beta = 0.0, math.radians(10.0)  # a pure sideslip
    reference = _margin_at_the_true_state(rocket, alpha, beta, "pitch")
    at_its_own_angle = stability_margin_and_slope(rocket, alpha, 0.0, 0.3, 0.0)[0]
    at_the_total_angle = stability_margin_and_slope(rocket, beta, 0.0, 0.3, 0.0)[0]

    assert at_its_own_angle == pytest.approx(reference, abs=0.05)
    assert reference > 1.0 > 0.0 > at_the_total_angle

    # the yaw plane of this rocket is linear, whatever the sideslip
    yaw_reference = _margin_at_the_true_state(rocket, alpha, beta, "yaw")
    yaw_margin = stability_margin_and_slope(rocket, 0.0, beta, 0.3, 0.0, "yaw")[0]
    assert yaw_margin == pytest.approx(yaw_reference, abs=1e-6)


def test_margin_of_an_axisymmetric_rocket_follows_the_total_angle():
    """An axisymmetric rocket behaves the same in every plane, so it has one
    margin, taken in the plane of the wind at the total angle of attack: the
    same value whether the wind comes as an angle of attack or as a sideslip."""
    area, diameter = math.pi * 0.0635**2, 2 * 0.0635
    body_lift = GenericSurface(
        area,
        diameter,
        {"cN": lambda alpha_total: 12.0 * math.sin(alpha_total) ** 2},
    )
    rocket = _rocket_with(body_lift, 0.3)
    assert rocket.is_axisymmetric and not rocket.is_incidence_linear

    angle = math.radians(8.0)
    in_the_wind_plane = stability_margin_and_slope(rocket, angle, 0.0, 0.3, 0.0)[0]
    assert in_the_wind_plane == pytest.approx(
        _margin_at_the_true_state(rocket, angle, 0.0, "pitch"), abs=1e-6
    )
    assert in_the_wind_plane == pytest.approx(
        _margin_at_the_true_state(rocket, 0.0, angle, "yaw"), abs=1e-6
    )
    across = stability_margin_and_slope(rocket, 0.0, angle, 0.3, 0.0, "yaw")[0]
    assert across == pytest.approx(in_the_wind_plane, abs=1e-6)


def test_restoring_slope_is_positive_in_both_planes():
    """The corrective moment ``q A slope margin d`` is a stiffness: positive for
    a stable rocket in either plane. The slope handed to it is therefore
    positive in yaw too, where a restoring side force has a negative
    ``dCY/dbeta``, and an axisymmetric rocket gives the same value in both
    planes, whether or not one of its surfaces is nonlinear."""
    area, diameter = math.pi * 0.0635**2, 2 * 0.0635
    linear = _rocket_with(GenericSurface(area, diameter, {"cN": 0}), 0.3)
    body_lift = GenericSurface(
        area,
        diameter,
        {"cN": lambda alpha_total: 12.0 * math.sin(alpha_total) ** 2},
    )
    nonlinear = _rocket_with(body_lift, 0.3)
    assert linear.is_incidence_linear and not nonlinear.is_incidence_linear

    for rocket in (linear, nonlinear):
        margin, slope = stability_margin_and_slope(rocket, 0.05, 0.0, 0.3, 0.0, "pitch")
        margin_yaw, slope_yaw = stability_margin_and_slope(
            rocket, 0.0, 0.05, 0.3, 0.0, "yaw"
        )
        assert slope > 0
        assert slope_yaw == pytest.approx(slope, rel=1e-6)
        assert margin_yaw == pytest.approx(margin, rel=1e-6)


def _table_that_stalls():
    """``cN = 2 alpha`` up to 12 degrees, flat beyond, as a table of points."""
    limit = math.radians(12)
    return [
        [angle, 2.0 * max(-limit, min(limit, angle))]
        for angle in np.radians(np.arange(-20, 21, 1.0))
    ]


@pytest.mark.parametrize(
    "coefficients, force_convention, linear",
    [
        ({"cN": lambda alpha: 2 * alpha, "cY": lambda beta: -2 * beta}, "body", True),
        (
            {
                "cN": lambda alpha, mach: (2 + mach) * alpha,
                "cm": lambda alpha: alpha / 3,
            },
            "body",
            True,
        ),
        ({"cN": lambda alpha: 3 * alpha + 60 * alpha**3}, "body", False),
        # Nonlinear only past the 5 degrees the check used to stop at
        (
            {"cN": lambda alpha: 2 * alpha + 400 * max(abs(alpha) - 0.14, 0) ** 2},
            "body",
            False,
        ),
        ({"cN": (_table_that_stalls(), ["alpha"])}, "body", False),
        # Nonlinear only at high Mach
        (
            {
                "cN": lambda alpha, mach: (
                    2 * alpha + (50 * alpha**3 if mach > 1.5 else 0)
                )
            },
            "body",
            False,
        ),
        # The force is linear but the moment is not
        (
            {
                "cN": lambda alpha: 2 * alpha,
                "cm": lambda alpha: alpha / 5 + 8 * alpha**3,
            },
            "body",
            False,
        ),
    ],
)
def test_is_incidence_linear_reads_the_whole_range(
    coefficients, force_convention, linear
):
    """The rocket is linear only when its neutral point stays put over the angles
    and Mach numbers a flight sees, not just at a single angle and Mach."""
    area, diameter = math.pi * 0.0635**2, 2 * 0.0635
    surface = GenericSurface(
        area, diameter, coefficients, force_convention=force_convention
    )
    rocket = _rocket_with(surface, 0.5)
    assert rocket.is_incidence_linear is linear

    # The answer must match the neutral point found from the flight forces
    moved = 0.0
    for mach in (0.3, 2.0):
        at_zero = neutral_point_and_slope(rocket, 0.0, 0.0, mach, "pitch")[0]
        for degrees in (3, 9, 14):
            angle = math.radians(degrees)
            point = neutral_point_and_slope(rocket, angle, 0.0, mach, "pitch")[0]
            moved = max(moved, abs(point - at_zero))
    assert (moved <= 1e-6) == linear


def test_a_single_nonlinear_surface_cannot_move_the_neutral_point():
    """A rocket whose whole lift comes from one surface with a fixed center of
    pressure has a neutral point that cannot move, however nonlinear its force.
    It keeps the fast, linear path."""
    area, diameter = math.pi * 0.0635**2, 2 * 0.0635
    rocket = Rocket(
        radius=0.0635,
        mass=14.426,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag=0.5,
        power_on_drag=0.5,
        center_of_mass_without_motor=0,
        coordinate_system_orientation="tail_to_nose",
    )
    whole_body = GenericSurface(
        area,
        diameter,
        {"cN": lambda alpha_total: alpha_total + 12 * math.sin(alpha_total) ** 2},
    )
    rocket.add_surfaces(whole_body, -0.4)
    assert rocket.is_incidence_linear


def test_is_incidence_linear_costs_nothing_for_the_built_in_surfaces(calisto_robust):
    """The built-in surfaces are linear whatever their data, so none of their
    coefficients is read."""
    with patch.object(
        AeroCoefficient, "get_value_opt", side_effect=AssertionError("read")
    ):
        assert is_incidence_linear(calisto_robust)


# Aerodynamic damping (C2 of the flight oscillator)


def _linear_damping(rocket, mach, plane):
    """The textbook sum 0.5 A sum((A_i / A) slope_i arm_i**2) at zero angle."""
    center_of_mass = rocket.center_of_mass.get_value_opt(0.0)
    total = 0.0
    for surface, position in rocket.aerodynamic_surfaces:
        if plane == "yaw":
            slope = -surface.cY_beta.get_value_opt(0, 0, mach, 0, 0, 0, 0)
            cp_z = surface.aerodynamic_center_yaw.get_value_opt(mach)
        else:
            slope = surface.cN_alpha.get_value_opt(0, 0, mach, 0, 0, 0, 0)
            cp_z = surface.aerodynamic_center.get_value_opt(mach)
        arm = position.z + rocket._csys * cp_z - center_of_mass
        total += surface.reference_area / rocket.area * slope * arm**2
    return 0.5 * rocket.area * total


@pytest.mark.parametrize("plane", ["pitch", "yaw"])
def test_damping_derivative_matches_the_linear_sum(calisto_robust, plane):
    """For a rocket of built-in surfaces the rate derivative of the summed
    moment about the center of mass equals the textbook slope-times-arm-squared
    sum, so the two paths of ``aerodynamic_damping`` agree."""
    rocket = calisto_robust
    assert rocket.is_incidence_linear
    closed_form = aerodynamic_damping(rocket, 0.0, 0.0, 0.3, 0.0, plane)
    assert closed_form == pytest.approx(_linear_damping(rocket, 0.3, plane))

    reference_z = -rocket.com_to_cdm_function.get_value_opt(0.0)
    derivative = damping_derivative(rocket, 0.0, 0.0, 0.3, plane, reference_z)
    assert derivative == pytest.approx(closed_form, rel=1e-5)
    assert derivative > 0


def test_boat_tail_takes_damping_away(calisto_robust):
    """A surface with a negative force slope (a boat tail) reduces the
    damping; it must not be counted as if it added to it."""
    rocket = calisto_robust
    tail = next(s for s, _ in rocket.aerodynamic_surfaces if "Tail" in type(s).__name__)
    assert tail.cN_alpha.get_value_opt(0, 0, 0.3, 0, 0, 0, 0) < 0
    with_tail = aerodynamic_damping(rocket, 0.0, 0.0, 0.3, 0.0, "pitch")
    rocket.aerodynamic_surfaces.remove(tail)
    without_tail = aerodynamic_damping(rocket, 0.0, 0.0, 0.3, 0.0, "pitch")
    assert with_tail < without_tail


def test_damping_reads_a_user_rate_coefficient():
    """A generic surface whose pitch moment depends on the pitch rate adds
    exactly that damping, ``0.25 A d**2 |cm_q|`` per unit density and speed,
    which the zero-angle slope sum alone would miss."""
    area, diameter = math.pi * 0.0635**2, 2 * 0.0635
    cm_q = -50.0
    surface = LinearGenericSurface(
        area, diameter, {"cN_alpha": 0.0, "cm_q": cm_q}, name="Rate damping"
    )
    rocket = _rocket_with(surface, 0.0)
    assert rocket.is_incidence_linear  # refreshes the flags
    assert rocket._uses_rate_coefficients
    without = _linear_damping(rocket, 0.3, "pitch")
    expected = without + 0.25 * area * diameter**2 * abs(cm_q)
    assert aerodynamic_damping(rocket, 0.0, 0.0, 0.3, 0.0, "pitch") == pytest.approx(
        expected, rel=1e-4
    )


def _phase_gated_rocket():
    """A rocket carrying one full-body surface per motor phase, the pair the
    documentation recommends, with a different center of pressure in each."""
    rocket = Rocket(
        radius=0.0635,
        mass=14.426,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag=0.5,
        power_on_drag=0.6,
        center_of_mass_without_motor=0,
    )
    surfaces = [
        LinearGenericSurface(
            rocket.area,
            2 * rocket.radius,
            {"cN_alpha": 2.0, "cY_beta": -2.0},
            center_of_pressure=(0, 0, cp_z),
            active_during=phase,
        )
        for cp_z, phase in ((-0.5, "power_off"), (-0.3, "power_on"))
    ]
    rocket.add_full_body_aerodynamics(surfaces, position=0.0)
    return rocket


def test_stability_counts_only_the_surfaces_of_its_phase():
    """A surface active during one motor phase only enters the stability
    analysis of that phase: coasting by default, the powered phase when
    ``stability_phase`` says so. Summing both would double the lift slope
    and average the two centers of pressure."""
    rocket = _phase_gated_rocket()
    with pytest.warns(UserWarning, match="power_off"):
        assert rocket.aerodynamic_center(0.3) == pytest.approx(-0.5)
    assert rocket.total_lift_coeff_der(0.3) == pytest.approx(2.0)
    assert rocket.aerodynamic_center_yaw(0.3) == pytest.approx(-0.5)
    assert rocket.neutral_point(0.0, 0.3) == pytest.approx(-0.5)

    rocket.stability_phase = "power_on"
    with pytest.warns(UserWarning, match="power_on"):
        assert rocket.aerodynamic_center(0.3) == pytest.approx(-0.3)
    assert rocket.total_lift_coeff_der(0.3) == pytest.approx(2.0)
    assert rocket.neutral_point(0.0, 0.3) == pytest.approx(-0.3)

    rocket.stability_phase = "coasting"
    with pytest.raises(ValueError, match="stability_phase"):
        rocket.aerodynamic_center(0.3)


def test_to_coefficients_lumps_each_phase_with_its_own_surfaces():
    """The lumped power-on and power-off sets each come from the surfaces
    active in that phase, whatever ``stability_phase`` is set to."""
    rocket = _phase_gated_rocket()
    coefficients = rocket.to_coefficients(machs=[0.3, 0.9])
    diameter = 2 * rocket.radius
    for phase, cp_z, drag in (("power_off", -0.5, 0.5), ("power_on", -0.3, 0.6)):
        phase_set = coefficients[phase]
        assert phase_set["cN_alpha"](0.3) == pytest.approx(2.0, rel=1e-6)
        # a force of slope 2 at cp_z gives a pitch moment slope of 2 cp_z / d
        assert phase_set["cm_alpha"](0.3) == pytest.approx(
            2.0 * cp_z / diameter, rel=1e-6
        )
        assert phase_set["cA_0"](0.3) == pytest.approx(drag)


def test_ungated_rocket_has_the_same_stability_in_both_phases(calisto_robust):
    """``stability_phase`` changes nothing for a rocket whose surfaces are all
    always active, and no phase warning is shown."""
    rocket = calisto_robust
    coasting = rocket.aerodynamic_center(0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rocket.stability_phase = "power_on"
        assert rocket.aerodynamic_center(0.5) == coasting
        assert rocket.to_coefficients(machs=[0.5, 1.0])["power_on"]["cN_alpha"](
            0.5
        ) == pytest.approx(rocket.total_lift_coeff_der(0.5), rel=1e-6)


def _lumped_twin(rocket, force_convention="body"):
    """A bare rocket carrying only the coasting surface of ``to_surface``."""
    twin = Rocket(
        radius=rocket.radius,
        mass=rocket.mass,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag=0,
        power_on_drag=0,
        center_of_mass_without_motor=rocket.center_of_mass_without_motor,
    )
    power_off, _ = rocket.to_surface(
        machs=[0.2, 0.3, 0.4], force_convention=force_convention
    )
    twin.add_full_body_aerodynamics(
        power_off, position=rocket.center_of_dry_mass_position
    )
    return twin


def _canted_calisto():
    rocket = Rocket(
        radius=0.0635,
        mass=14.426,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag=0.5,
        power_on_drag=0.5,
        center_of_mass_without_motor=0,
    )
    rocket.add_nose(length=0.55829, kind="vonkarman", position=1.278)
    rocket.add_trapezoidal_fins(
        n=4,
        span=0.1,
        root_chord=0.12,
        tip_chord=0.04,
        position=-1.04956,
        cant_angle=2.0,
    )
    return rocket


def test_lumped_coefficients_keep_the_roll_forcing_of_canted_fins():
    """Canted fins roll the rocket at zero angle of attack. The lumped set
    keeps that as ``cl_0`` and the lumped rocket rolls the same way."""
    rocket = _canted_calisto()
    coefficients = rocket.to_coefficients(machs=[0.3, 0.9])["power_off"]
    assert coefficients["cl_0"](0.3) < 0
    assert "cN_beta" not in coefficients  # still axisymmetric: no cross terms
    twin = _lumped_twin(rocket)
    original = summed_force_and_moment(rocket, 0.0, 0.0, 0.3, (0, 0, 0.5))
    lumped = summed_force_and_moment(twin, 0.0, 0.0, 0.3, (0, 0, 0.5))
    assert lumped[5] == pytest.approx(original[5], rel=1e-6)


@pytest.mark.parametrize("force_convention", ["body", "wind"])
def test_lumped_coefficients_keep_the_cross_terms_of_an_asymmetric_rocket(
    force_convention,
):
    """A single canted fin off the body axes couples the two planes and has a
    force and a moment at zero angle. The lumped rocket reproduces the
    original's force and moment to first order in the angles and rates, in
    either frame."""
    rocket = Rocket(
        radius=0.0635,
        mass=14.426,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag=0.5,
        power_on_drag=0.5,
        center_of_mass_without_motor=0,
    )
    rocket.add_nose(length=0.55829, kind="vonkarman", position=1.278)
    rocket.add_surfaces(
        TrapezoidalFin(45, 0.12, 0.04, 0.1, 0.0635, cant_angle=2.0), -1.04956
    )
    coefficients = rocket.to_coefficients(machs=[0.3, 0.9])["power_off"]
    for name in ("cN_beta", "cm_beta", "cn_alpha", "cN_0", "cm_0", "cl_0", "cl_alpha"):
        assert name in coefficients
    twin = _lumped_twin(rocket, force_convention)
    for alpha, beta, omega in (
        (0.0, 0.0, (0, 0, 0)),
        (0.01, 0.0, (0, 0, 0)),
        (0.0, 0.01, (0, 0, 0)),
        (0.01, 0.01, (0.1, 0.1, 0.1)),
    ):
        original = summed_force_and_moment(rocket, alpha, beta, 0.3, omega, speed=100)
        lumped = summed_force_and_moment(twin, alpha, beta, 0.3, omega, speed=100)
        # the twin's cA_0 holds the rocket's own drag curve on top of the fin
        lumped[2] += 0.5 * 100**2 * rocket.area * 0.5
        np.testing.assert_allclose(lumped, original, rtol=2e-3, atol=1e-6)


def test_to_coefficients_accepts_a_single_mach():
    rocket = _canted_calisto()
    coefficients = rocket.to_coefficients(machs=[0.3])["power_off"]
    assert coefficients["cN_alpha"](0.3) == pytest.approx(
        rocket.total_lift_coeff_der(0.3), rel=1e-6
    )


def _source_coefficients(rocket, alpha, mach, reynolds=None):
    """``(cN, cm)`` of the rocket's own surfaces, summed directly."""
    rocket.evaluate_surfaces_cp_to_cdm()
    forces = summed_force_and_moment(
        rocket, alpha, 0.0, mach, (0, 0, 0), reynolds=reynolds
    )
    dynamic_pressure_area = 0.5 * rocket.area
    return (
        -forces[1] / dynamic_pressure_area,
        forces[3] / (dynamic_pressure_area * 2 * rocket.radius),
    )


def _reynolds_rocket():
    """A rocket with a surface whose lift grows with the Reynolds number."""
    area, diameter = math.pi * 0.0635**2, 2 * 0.0635

    def gain(reynolds):
        return 1 + math.log10(max(reynolds, 1.0))

    surface = GenericSurface(
        area,
        diameter,
        {
            "cN": lambda alpha, reynolds: alpha * gain(reynolds),
            "cY": lambda beta, reynolds: -beta * gain(reynolds),
        },
    )
    return _rocket_with(surface, 0.3)


@pytest.mark.parametrize("model", ["linear", "table"])
def test_lumping_follows_the_reynolds_number(model):
    """A single Reynolds number sets the value the coefficients are read at;
    a list makes them (and the surface built from them) follow it. The
    Reynolds number is the rocket's, based on its diameter."""
    rocket = _reynolds_rocket()
    machs, angles = [0.2, 0.5, 0.8], np.radians([-10, -5, 0, 5, 10])
    alpha = math.radians(5)

    at_zero = rocket.to_coefficients(machs=machs)["power_off"]["cN_alpha"](0.5)
    fixed = rocket.to_coefficients(machs=machs, reynolds=1e6)["power_off"]
    assert fixed["cN_alpha"](0.5) == pytest.approx(at_zero + 6.0, rel=1e-6)
    swept = rocket.to_coefficients(machs=machs, reynolds=[1e5, 1e6, 1e7])
    assert swept["power_off"]["cN_alpha"](0.5, 1e7) == pytest.approx(
        at_zero + 7.0, rel=1e-6
    )

    power_off, _ = rocket.to_surface(
        machs=machs, model=model, angles=angles, reynolds=[1e5, 1e6, 1e7]
    )
    for reynolds in (1e5, 1e6, 1e7):
        expected = _source_coefficients(rocket, alpha, 0.5, reynolds)[0]
        lumped = power_off.cN(alpha, 0, 0.5, reynolds, 0, 0, 0)
        assert lumped == pytest.approx(expected, rel=1e-6)


def _canard(name):
    area, diameter = math.pi * 0.0635**2, 2 * 0.0635
    return ControllableGenericSurface(
        area,
        diameter,
        {
            "cN": lambda alpha, deflection: 1.5 * (alpha + deflection),
            "cY": lambda beta: -1.5 * beta,
        },
        name=name,
    )


def test_lumping_keeps_a_control():
    """A control listed in ``controls`` becomes an input of the coefficients,
    the surface built from them is controllable and matches the rocket at a
    deflection, and the rocket's own control is left where it was."""
    canard = _canard("Canard")
    rocket = _rocket_with(canard, 0.8)
    canard.set_control("deflection", 0.02)
    machs, angles = [0.2, 0.5, 0.8], np.radians([-10, 0, 10])
    controls = {"deflection": np.radians([-10, 0, 10])}

    linear = rocket.to_coefficients(machs=machs, controls=controls)["power_off"]
    assert linear["cN_0"](0.5, math.radians(10)) == pytest.approx(
        1.5 * math.radians(10)
    )
    assert canard.get_control("deflection") == 0.02

    power_off, _ = rocket.to_surface(
        machs=machs, model="table", angles=angles, controls=controls
    )
    assert isinstance(power_off, ControllableGenericSurface)
    assert power_off.control_variables == ["deflection"]
    alpha, deflection = math.radians(3), math.radians(5)
    power_off.set_control("deflection", deflection)
    canard.set_control("deflection", deflection)
    args = power_off._coefficient_arguments(alpha, 0, 0.5, 0, 0, 0, 0)
    expected = _source_coefficients(rocket, alpha, 0.5)
    assert power_off.cN.get_value_opt(*args) == pytest.approx(expected[0])
    assert power_off.cm.get_value_opt(*args) == pytest.approx(expected[1])

    with pytest.raises(ValueError, match="needs model='table'"):
        rocket.to_surface(machs=machs, controls=controls)
    with pytest.raises(ValueError, match="no control named 'flap'"):
        rocket.to_coefficients(machs=machs, controls={"flap": [0, 1]})
    with pytest.raises(ValueError, match="at least two different values"):
        rocket.to_coefficients(machs=machs, controls={"deflection": [0.1]})


def test_lumping_keeps_same_named_controls_apart():
    """Two surfaces with a control of the same name give two inputs, named
    after their surfaces; asking for the shared name warns and sweeps both."""
    rocket = _rocket_with(_canard("Canard A"), 0.8)
    rocket.add_surfaces(_canard("Canard B"), 0.6)
    machs = [0.2, 0.5]
    names = ["mach", "canard_a_deflection", "canard_b_deflection"]

    with pytest.warns(UserWarning, match="used by 2 surfaces"):
        shared = rocket.to_coefficients(machs=machs, controls={"deflection": [0, 0.1]})
    assert shared["power_off"]["cN_0"].__inputs__ == names

    apart = rocket.to_coefficients(
        machs=machs,
        controls={"canard_a_deflection": [0, 0.1], "canard_b_deflection": [0, 0.2]},
    )["power_off"]
    assert apart["cN_0"].__inputs__ == names
    assert apart["cN_0"](0.5, 0.1, 0.2) == pytest.approx(1.5 * 0.3)


def test_table_rates_can_be_read_at_each_angle():
    """With ``rates="at_each_angle"`` the rate terms are tables over the
    angles, and the surface built from them has the damping of the rocket at
    an angle, which the zero-angle rate terms miss."""
    rocket = _body_lift_rocket()
    machs, angles = [0.2, 0.5, 0.8], np.radians([-10, -5, 0, 5, 10])
    theta = math.radians(10)

    coefficients = rocket.to_coefficients(
        machs=machs, model="table", angles=angles, rates="at_each_angle"
    )["power_off"]
    assert coefficients["cm_q"].__inputs__ == ["alpha", "beta", "mach"]

    def twin_damping(rates):
        twin = _table_twin(rocket, angles=angles, rates=rates)
        return damping_derivative(twin, theta, 0.0, 0.3)

    expected = damping_derivative(rocket, theta, 0.0, 0.3)
    assert twin_damping("at_each_angle") == pytest.approx(expected, rel=1e-4)
    assert twin_damping(True) != pytest.approx(expected, rel=1e-2)

    with pytest.raises(ValueError, match="needs model='table'"):
        rocket.to_coefficients(machs=machs, rates="at_each_angle")
    with pytest.raises(ValueError, match="rates must be"):
        rocket.to_coefficients(machs=machs, rates="sometimes")


def _table_twin(rocket, **kwargs):
    """A bare rocket carrying only the coasting table surface of ``to_surface``."""
    twin = Rocket(
        radius=rocket.radius,
        mass=rocket.mass,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag=0,
        power_on_drag=0,
        center_of_mass_without_motor=rocket.center_of_mass_without_motor,
    )
    power_off, _ = rocket.to_surface(model="table", machs=[0.2, 0.3, 0.4], **kwargs)
    twin.add_full_body_aerodynamics(
        power_off, position=rocket.center_of_dry_mass_position
    )
    return twin


def _assert_twin_matches(rocket, twin, states, rtol):
    """The twin's summed force and moment equal the original's at each
    ``(alpha, beta, omega)`` state, once the rocket's own drag, which the
    twin carries in its axial coefficient, is put back."""
    for alpha, beta, omega in states:
        original = summed_force_and_moment(rocket, alpha, beta, 0.3, omega, speed=100)
        lumped = summed_force_and_moment(twin, alpha, beta, 0.3, omega, speed=100)
        lumped[2] += 0.5 * 100**2 * rocket.area * rocket.power_off_drag(0.3)
        np.testing.assert_allclose(
            lumped, original, rtol=rtol, atol=rtol * np.abs(original).max()
        )


def test_table_model_keeps_the_curve_of_an_axisymmetric_rocket():
    """An axisymmetric rocket with a body-lift term that is far from linear
    lumps to a 2-D table over the total angle of attack. The lumped rocket
    matches the original at every angle of the table, from either side and at
    combined angles, where the linear model is exact only near zero."""
    rocket = _canted_calisto()
    body_lift = GenericSurface(
        rocket.area,
        2 * rocket.radius,
        {"cN": lambda alpha_total: 12.0 * math.sin(alpha_total) ** 2},
    )
    rocket.add_surfaces(body_lift, 0.3)
    assert rocket.is_axisymmetric
    angles = np.radians(np.arange(-40, 41, 2))
    coefficients = rocket.to_coefficients(
        model="table", machs=[0.2, 0.3, 0.4], angles=angles
    )
    tables = coefficients["power_off"]
    assert set(tables) >= {"cN", "cA", "cm", "cl", "cm_q", "cl_p"}
    assert "cY" not in tables and "cn" not in tables
    assert tables["cN"].__inputs__ == ["alpha_total", "mach"]

    twin = _table_twin(rocket, angles=angles)
    deg = math.radians
    static = [
        (0.0, 0.0, (0, 0, 0)),
        (deg(10), 0.0, (0, 0, 0)),
        (0.0, deg(-25), (0, 0, 0)),
        (deg(20), deg(20), (0, 0, 0)),
        (deg(40), 0.0, (0, 0, 0)),
    ]
    _assert_twin_matches(rocket, twin, static, rtol=1e-3)
    # the rate terms are those of the linear model, read at zero angle
    _assert_twin_matches(rocket, twin, [(deg(5), deg(3), (0.3, 0.2, 0.5))], rtol=1e-2)


def test_table_model_keeps_the_coupling_of_a_lopsided_rocket():
    """A single canted fin off the body axes plus a nonlinear canard in one
    plane lump to 3-D tables over both angles that reproduce the original at
    combined angles, cross coupling included."""
    rocket = Rocket(
        radius=0.0635,
        mass=14.426,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag=0.5,
        power_on_drag=0.5,
        center_of_mass_without_motor=0,
    )
    rocket.add_nose(length=0.55829, kind="vonkarman", position=1.278)
    rocket.add_surfaces(
        TrapezoidalFin(45, 0.12, 0.04, 0.1, 0.0635, cant_angle=2.0), -1.04956
    )
    canard = GenericSurface(
        rocket.area,
        2 * rocket.radius,
        {"cN": lambda alpha: 1.5 * math.sin(2 * alpha) + 20 * alpha**3, "cY": 0},
    )
    rocket.add_surfaces(canard, 0.8)
    assert not rocket.is_axisymmetric
    angles = np.radians(np.arange(-30, 31, 5))
    tables = rocket.to_coefficients(
        model="table", machs=[0.2, 0.3, 0.4], angles=angles
    )["power_off"]
    assert set(tables) >= {"cN", "cY", "cA", "cm", "cn", "cl"}
    assert tables["cN"].__inputs__ == ["alpha", "beta", "mach"]

    twin = _table_twin(rocket, angles=angles)
    deg = math.radians
    static = [
        (0.0, 0.0, (0, 0, 0)),
        (deg(10), 0.0, (0, 0, 0)),
        (0.0, deg(25), (0, 0, 0)),
        (deg(20), deg(-20), (0, 0, 0)),
    ]
    _assert_twin_matches(rocket, twin, static, rtol=1e-3)
    _assert_twin_matches(rocket, twin, [(deg(5), deg(3), (0.3, 0.2, 0.5))], rtol=1e-2)


def test_lumping_without_rates_has_no_damping():
    """``rates=False`` leaves the rate terms out of both models: the linear
    set has no ``_p``/``_q``/``_r`` keys and the table twin feels nothing
    from a pure body rate."""
    rocket = _canted_calisto()
    linear = rocket.to_coefficients(machs=[0.3, 0.9], rates=False)["power_off"]
    assert not any(name[-2:] in ("_p", "_q", "_r") for name in linear)
    assert "cN_alpha" in linear and "cl_0" in linear
    twin = _table_twin(rocket, rates=False)
    moment = summed_force_and_moment(twin, 0.0, 0.0, 0.3, (1.0, 0, 0), speed=100)
    assert moment[3] == 0.0


def test_lumping_rejects_bad_options():
    rocket = _canted_calisto()
    with pytest.raises(ValueError, match="model"):
        rocket.to_coefficients(model="quadratic")
    with pytest.raises(ValueError, match="body-frame"):
        rocket.to_coefficients(model="table", force_convention="wind")
    with pytest.raises(ValueError, match="at least two"):
        rocket.to_coefficients(model="table", machs=[0.3])


def test_table_surface_saves_and_loads():
    """The table surfaces round trip through the RocketPy encoder."""
    rocket = _canted_calisto()
    power_off, _ = rocket.to_surface(
        model="table", machs=[0.2, 0.3], angles=np.radians([-10, 0, 10])
    )
    loaded = json.loads(json.dumps(power_off, cls=RocketPyEncoder), cls=RocketPyDecoder)
    args = (math.radians(6), math.radians(-4), 0.25, 0.0, 0.1, 0.0, 0.2)
    for name in ("cN", "cY", "cA", "cm", "cn", "cl"):
        assert getattr(loaded, name).get_value_opt(*args) == pytest.approx(
            getattr(power_off, name).get_value_opt(*args)
        )


def _rocket_of(*surfaces_and_positions):
    rocket = Rocket(
        radius=0.0635,
        mass=14.426,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag=0.5,
        power_on_drag=0.5,
        center_of_mass_without_motor=0,
    )
    rocket.add_nose(length=0.55829, kind="vonkarman", position=1.278)
    for surface, position in surfaces_and_positions:
        rocket.add_surfaces(surface, position)
    return rocket


def _fin(angular_position, cant_angle=0.0):
    return TrapezoidalFin(
        angular_position, 0.12, 0.04, 0.1, 0.0635, cant_angle=cant_angle
    )


def test_a_coupled_rocket_is_not_axisymmetric():
    """A single fin at 45 degrees pushes equally in pitch and yaw with the same
    center of pressure in both planes, so the two margins agree. It is still
    not axisymmetric: a pitch also yaws it, and along its own plane it does
    not push back at all."""
    rocket = _rocket_of((_fin(45), -1.04956))
    with pytest.warns(UserWarning, match="couple"):
        assert not rocket.is_axisymmetric
    assert rocket.aerodynamic_center(0.3) == pytest.approx(
        rocket.aerodynamic_center_yaw(0.3)
    )


def test_planes_of_equal_center_but_different_strength_are_not_axisymmetric():
    """Lift added to the pitch plane alone, at the aerodynamic center, leaves
    the two centers of pressure equal but the pitch plane stiffer."""
    area, diameter = math.pi * 0.0635**2, 2 * 0.0635
    both = LinearGenericSurface(area, diameter, {"cN_alpha": 4.0, "cY_beta": -4.0})
    pitch_only = LinearGenericSurface(area, diameter, {"cN_alpha": 2.0})
    rocket = Rocket(
        radius=0.0635,
        mass=14.426,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag=0.5,
        power_on_drag=0.5,
        center_of_mass_without_motor=0,
    )
    rocket.add_surfaces([both, pitch_only], [-1.0, -1.0])
    with pytest.warns(UserWarning, match="strength"):
        assert not rocket.is_axisymmetric
    assert rocket.aerodynamic_center(0.3) == pytest.approx(
        rocket.aerodynamic_center_yaw(0.3)
    )


@pytest.mark.parametrize(
    "angular_positions", [(0, 90, 180, 270), (30, 150, 270), (45, 135, 225, 315)]
)
def test_evenly_spaced_individual_fins_are_axisymmetric(angular_positions):
    """Three or more evenly spaced fins, canted or not, behave the same in
    every plane, whatever their orientation about the axis."""
    rocket = _rocket_of(
        *((_fin(angle, cant_angle=1.5), -1.04956) for angle in angular_positions)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert rocket.is_axisymmetric


# Both planes are read at one state (alpha, beta)


def _body_lift_rocket():
    """Calisto-like rocket plus a Galejs body-lift term against the total
    angle of attack: axisymmetric, nonlinear in the angle."""
    area, diameter = math.pi * 0.0635**2, 2 * 0.0635
    body_lift = GenericSurface(
        area,
        diameter,
        {
            "cN": lambda alpha_total: (
                12.0 * math.sin(alpha_total) ** 2 * math.cos(alpha_total)
            )
        },
    )
    return _rocket_with(body_lift, 0.3)


def test_center_of_pressure_of_a_linear_rocket_is_the_aerodynamic_center(
    calisto_robust,
):
    """For a rocket built from nose, fins and tail the point where the force
    acts does not move with the angle: it is ``cp_position`` at every angle,
    in both planes, and at zero angle where the force itself vanishes."""
    rocket = calisto_robust
    for mach in (0.0, 0.3, 0.9):
        expected = rocket.cp_position.get_value_opt(mach)
        for degrees in (0.0, 2.0, 10.0):
            angle = math.radians(degrees)
            assert rocket.center_of_pressure(angle, mach) == pytest.approx(expected)
            assert rocket.center_of_pressure_yaw(angle, mach) == pytest.approx(expected)


def test_center_of_pressure_of_a_nonlinear_rocket_moves_with_the_angle():
    """With a body-lift term the center of pressure ``Cm/CN`` leaves the
    aerodynamic center as the angle grows, differs from the in-plane neutral
    point, and is the across-the-wind neutral point. Turning the wind into
    the yaw plane gives the same point from the yaw accessor."""
    rocket = _body_lift_rocket()
    theta, mach = math.radians(10), 0.3

    center = rocket.center_of_pressure(theta, mach)
    assert center == pytest.approx(
        rocket.neutral_point_yaw(0.0, mach, alpha=theta), abs=1e-5
    )
    assert abs(center - rocket.neutral_point(theta, mach)) > 0.05
    assert abs(center - rocket.aerodynamic_center.get_value_opt(mach)) > 0.05
    assert rocket.center_of_pressure_yaw(theta, mach) == pytest.approx(center)
    # Toward zero angle it returns to the aerodynamic center
    assert rocket.center_of_pressure(0.0, mach) == pytest.approx(
        rocket.aerodynamic_center.get_value_opt(mach)
    )
    assert rocket.center_of_pressure(1e-4, mach) == pytest.approx(
        rocket.aerodynamic_center.get_value_opt(mach), abs=1e-3
    )


def test_axisymmetric_nonlinear_yaw_is_read_across_the_wind():
    """With the wind in the pitch plane, state ``(theta, 0)``, the pitch plane
    of an axisymmetric rocket nonlinear in the angle gives the in-plane
    (tangent) neutral point and the yaw plane the across-the-wind one, which
    is the classic center of pressure ``Cm/CN`` at that angle with a restoring
    slope ``CN / tan(theta)``. The two margins differ."""
    rocket = _body_lift_rocket()
    assert rocket.is_axisymmetric and not rocket.is_incidence_linear
    theta, mach = math.radians(10), 0.3
    diameter = 2 * rocket.radius

    r1, r2, _, m1, _, _ = summed_force_and_moment(rocket, theta, 0.0, mach, (0, 0, 0))
    dynamic_pressure_area = 0.5 * rocket.area
    cN = -r2 / dynamic_pressure_area
    cm = m1 / (dynamic_pressure_area * diameter)
    secant_point = rocket.center_of_dry_mass_position + rocket._csys * diameter * (
        cm / cN
    )

    across = rocket.neutral_point_yaw(0.0, mach, alpha=theta)
    in_plane = rocket.neutral_point(theta, mach)
    assert across == pytest.approx(secant_point, abs=1e-5)
    assert abs(in_plane - secant_point) > 0.05

    pitch_margin, pitch_slope = stability_margin_and_slope(
        rocket, theta, 0.0, mach, 0.0, "pitch"
    )
    yaw_margin, yaw_slope = stability_margin_and_slope(
        rocket, theta, 0.0, mach, 0.0, "yaw"
    )
    assert yaw_slope == pytest.approx(cN / math.tan(theta), rel=1e-5)
    assert pitch_slope > yaw_slope
    assert yaw_margin > pitch_margin + 0.5

    # Rotating the wind into the yaw plane swaps the roles
    assert rocket.neutral_point_yaw(theta, mach) == pytest.approx(in_plane)
    assert rocket.neutral_point(0.0, mach, beta=theta) == pytest.approx(across)


def test_rocket_level_margins_are_read_at_zero_angle():
    """``stability_margin(mach, time)`` and ``stability_margin_yaw(mach, time)``
    keep their two-argument form and read each plane at zero angle; by symmetry
    the two agree for an axisymmetric rocket, at zero angle and away from it."""
    rocket = _body_lift_rocket()
    at_zero = stability_margin_and_slope(rocket, 0.0, 0.0, 0.3, 0.0)[0]
    assert rocket.stability_margin(0.3, 0.0) == pytest.approx(at_zero, rel=1e-9)
    assert rocket.stability_margin_yaw(0.3, 0.0) == pytest.approx(at_zero, rel=1e-6)
    for angle in (0.1, 0.2):
        assert stability_margin_and_slope(rocket, angle, 0.0, 0.3, 0.0, "pitch")[
            0
        ] == pytest.approx(
            stability_margin_and_slope(rocket, 0.0, angle, 0.3, 0.0, "yaw")[0],
            rel=1e-6,
        )


def test_non_axisymmetric_nonlinear_plane_depends_on_the_other_angle():
    """Canards in one plane with a lift curve that stalls: the pitch-plane
    margin read at the rocket's actual state ``(alpha, beta)`` is what a
    direct linearization there gives, not the value at ``(alpha, 0)``."""
    area, diameter = math.pi * 0.0635**2, 2 * 0.0635
    limit = math.radians(12)
    table = [
        [angle, 2.0 * max(-limit, min(limit, angle))]
        for angle in np.radians(np.arange(-30, 31, 1.0))
    ]
    canards = GenericSurface(area, diameter, {"cN": (table, ["alpha"]), "cY": 0})
    rocket = _rocket_with(canards, 0.8)
    assert not rocket.is_axisymmetric and not rocket.is_incidence_linear
    alpha, beta, mach = math.radians(8), math.radians(8), 0.3

    margin, slope = stability_margin_and_slope(rocket, alpha, beta, mach, 0.0, "pitch")
    point, direct_slope = neutral_point_and_slope(rocket, alpha, beta, mach, "pitch")
    expected = (
        rocket._csys * (rocket.center_of_mass.get_value_opt(0.0) - point) / diameter
    )
    assert margin == pytest.approx(expected)
    assert slope == pytest.approx(direct_slope)
    # Past the stall the canards stop adding lift and the margin grows
    below = stability_margin_and_slope(rocket, 0.0, 0.0, mach, 0.0, "pitch")[0]
    beyond = stability_margin_and_slope(
        rocket, math.radians(15), 0.0, mach, 0.0, "pitch"
    )[0]
    assert beyond > below + 0.5


def test_closed_form_damping_follows_the_stability_phase():
    """A surface active only in the other motor phase is left out of the
    closed-form damping sum, as it is out of the aerodynamic center."""
    rocket = _phase_gated_rocket()
    off_only = aerodynamic_damping(rocket, 0.0, 0.0, 0.3, 0.0, "pitch")
    rocket.stability_phase = "power_on"
    on_only = aerodynamic_damping(rocket, 0.0, 0.0, 0.3, 0.0, "pitch")
    rocket.stability_phase = "power_off"
    # The two surfaces have different centers of pressure, so different arms
    assert off_only != pytest.approx(on_only)
    for phase, expected in (("power_off", off_only), ("power_on", on_only)):
        rocket.stability_phase = phase
        surfaces = stability_surfaces(rocket)
        arm_sum = sum(
            surface.reference_area
            / rocket.area
            * surface.cN_alpha.get_value_opt(0, 0, 0.3, 0, 0, 0, 0)
            * (
                position.z
                + rocket._csys * surface.aerodynamic_center.get_value_opt(0.3)
                - rocket.center_of_mass.get_value_opt(0.0)
            )
            ** 2
            for surface, position in surfaces
        )
        assert expected == pytest.approx(0.5 * rocket.area * arm_sum)


def test_a_surface_that_starts_switched_off_is_left_out_of_the_stability_sum():
    """It is not part of the rocket until an event switches it on."""
    area, length = math.pi * 0.0635**2, 0.127
    coefficients = {"cN_alpha": 2.0, "cY_beta": -2.0}
    switched_off = LinearGenericSurface(area, length, coefficients, active=False)
    switched_on = LinearGenericSurface(area, length, coefficients)

    without = stability_surfaces(_rocket_with(switched_off, -0.8))
    with_it = stability_surfaces(_rocket_with(switched_on, -0.8))

    assert switched_off not in [surface for surface, _ in without]
    assert switched_on in [surface for surface, _ in with_it]
    assert len(with_it) == len(without) + 1


def _oscillator_rocket():
    """A rocket of built-in surfaces with a generic surface that adds nothing."""
    nothing = LinearGenericSurface(math.pi * 0.0635**2, 0.127, {})
    return _rocket_with(nothing, 0.5)


def test_disturbance_response_is_the_damped_oscillation_of_the_rocket():
    """The response starts at the disturbance with no rate, and follows
    ``I theta'' + C2 theta' + C1 theta = 0`` with the rocket's coefficients at
    the chosen condition."""
    rocket = _oscillator_rocket()
    speed, density = 40.0, 1.1
    response = rocket.disturbance_response(speed, disturbance=4.0, density=density)

    inertia, inertia_rate = lateral_inertia_and_rate(rocket, rocket.I_11, 0.0)
    corrective, damping = corrective_and_damping_moments(
        rocket,
        0.0,
        0.0,
        speed / 340.29,
        0.0,
        speed,
        density,
        0.5 * density * speed**2,
        inertia_rate,
    )
    natural_frequency = math.sqrt(corrective / inertia)
    damping_ratio = damping / (2 * math.sqrt(corrective * inertia))
    assert 0 < damping_ratio < 1

    time = response.x_array
    damped_frequency = natural_frequency * math.sqrt(1 - damping_ratio**2)
    decay = damping_ratio * natural_frequency
    expected = (
        4.0
        * np.exp(-decay * time)
        * (
            np.cos(damped_frequency * time)
            + decay / damped_frequency * np.sin(damped_frequency * time)
        )
    )
    assert response.y_array == pytest.approx(expected, abs=1e-9)
    assert response.y_array[0] == pytest.approx(4.0)
    assert abs(response.y_array[-1]) < 0.1 * 4.0  # it settles
    assert f"{natural_frequency:.2f} rad/s" in response.title


def test_disturbance_response_grows_without_a_restoring_moment():
    """A rocket with its center of pressure ahead of its center of mass does
    not swing back: the angle grows."""
    rocket = Rocket(
        radius=0.0635,
        mass=14.426,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag=0.5,
        power_on_drag=0.5,
        center_of_mass_without_motor=0,
        coordinate_system_orientation="tail_to_nose",
    )
    rocket.add_nose(length=0.55829, kind="vonkarman", position=1.278)
    response = rocket.disturbance_response(speed=30)
    assert response.y_array[-1] > 3 * response.y_array[0]
    assert "no restoring moment" in response.title


def test_disturbance_response_of_a_point_mass_rocket_is_refused():
    rocket = PointMassRocket(
        radius=0.05,
        mass=5,
        center_of_mass_without_motor=0,
        power_off_drag=0.4,
        power_on_drag=0.4,
    )
    with pytest.raises(ValueError, match="point mass"):
        rocket.disturbance_response(speed=30)
