"""Regression tests for the GenericSurface-rooted aerodynamic hierarchy.

After the refactor, every aerodynamic surface (Barrowman or generic) is
described by the generic coefficient model and exposes the diagnostic accessors
``lift_coefficient_derivative`` and ``center_of_pressure_z`` used by the rocket's
center-of-pressure / stability-margin computation. These tests pin the
properties that the refactor is meant to guarantee.
"""

import warnings

import numpy as np
import pytest

from rocketpy import LinearGenericSurface, NoseCone, Tail, TrapezoidalFins


def test_barrowman_derived_cp_matches_geometric_cp():
    """The derived ``center_of_pressure_z`` must reproduce the geometric cp of
    each Barrowman surface (the moment is carried by ``cm`` but the diagnostic
    must recover the original location)."""
    nose = NoseCone(
        length=0.55829, kind="vonkarman", base_radius=0.0635, rocket_radius=0.0635
    )
    tail = Tail(
        top_radius=0.0635, bottom_radius=0.0435, length=0.060, rocket_radius=0.0635
    )
    fins = TrapezoidalFins(
        n=4, span=0.100, root_chord=0.120, tip_chord=0.040, rocket_radius=0.0635
    )

    for surface in (nose, tail, fins):
        for mach in (0.0, 0.5, 0.9):
            assert (
                pytest.approx(
                    surface.center_of_pressure_z.get_value_opt(mach), rel=1e-6, abs=1e-9
                )
                == surface.cpz
            )
        # The normal-force slope diagnostic must equal the Barrowman clalpha.
        assert pytest.approx(
            nose.lift_coefficient_derivative.get_value_opt(0.0)
        ) == nose.clalpha.get_value_opt(0.0)


def test_generic_surface_contributes_to_static_margin(calisto_motorless):
    """A generic surface must now contribute to the rocket center of pressure
    (previously generic surfaces were skipped, breaking stability margin)."""
    rocket = calisto_motorless
    rocket.add_nose(length=0.55829, kind="vonkarman", position=1.278)

    cp_without_generic = rocket.aerodynamic_center.get_value_opt(0.2)

    # A lifting generic surface placed aft should move the cp aft (more stable).
    generic = LinearGenericSurface(
        reference_area=rocket.area,
        reference_length=2 * rocket.radius,
        coefficients={
            "cL_alpha": lambda a, b, m, re, p, q, r: 2.0,
            "cm_alpha": lambda a, b, m, re, p, q, r: -1.0,
        },
        name="generic_fins",
    )
    rocket.add_surfaces(generic, positions=-1.0)

    cp_with_generic = rocket.aerodynamic_center.get_value_opt(0.2)
    assert cp_with_generic != pytest.approx(cp_without_generic)
    assert np.isfinite(cp_with_generic)


def test_zero_lift_surface_does_not_break_cp(calisto_motorless):
    """A surface with no normal-force slope must drop out of the lift-weighted
    cp average without producing NaNs."""
    rocket = calisto_motorless
    rocket.add_nose(length=0.55829, kind="vonkarman", position=1.278)
    cp_reference = rocket.aerodynamic_center.get_value_opt(0.2)

    drag_only = LinearGenericSurface(
        reference_area=rocket.area,
        reference_length=2 * rocket.radius,
        coefficients={"cD_0": lambda a, b, m, re, p, q, r: 0.5},
        name="drag_only",
    )
    rocket.add_surfaces(drag_only, positions=-1.0)

    cp_after = rocket.aerodynamic_center.get_value_opt(0.2)
    assert np.isfinite(cp_after)
    assert cp_after == pytest.approx(cp_reference)


def test_axisymmetric_rocket_pitch_equals_yaw_margin(calisto_motorless):
    """An axisymmetric rocket must have identical pitch and yaw margins and
    must not raise the asymmetry warning."""
    rocket = calisto_motorless
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # asymmetry warning would fail the test
        rocket.add_nose(length=0.55829, kind="vonkarman", position=1.278)
        rocket.add_trapezoidal_fins(
            n=4, span=0.100, root_chord=0.120, tip_chord=0.040, position=-1.04
        )
        rocket.add_tail(
            top_radius=0.0635, bottom_radius=0.0435, length=0.060, position=-1.194
        )

    for mach in (0.0, 0.5, 0.9):
        assert rocket.aerodynamic_center.get_value_opt(mach) == pytest.approx(
            rocket.aerodynamic_center_yaw.get_value_opt(mach), abs=1e-9
        )
    assert rocket.static_margin.get_value_opt(0) == pytest.approx(
        rocket.static_margin_yaw.get_value_opt(0), abs=1e-9
    )


def test_non_axisymmetric_rocket_splits_margins_and_warns(calisto_motorless):
    """A non-axisymmetric generic surface must yield distinct pitch/yaw margins
    and raise a warning that the scalar margin describes the pitch plane only.

    The advisory is emitted lazily -- on the first evaluation of the aerodynamic
    center, not eagerly at add time -- so adding the surface itself is silent."""
    rocket = calisto_motorless
    rocket.add_nose(length=0.55829, kind="vonkarman", position=1.278)
    asymmetric = LinearGenericSurface(
        reference_area=rocket.area,
        reference_length=2 * rocket.radius,
        coefficients={
            "cL_alpha": lambda a, b, m, re, p, q, r: 2.0,
            "cm_alpha": lambda a, b, m, re, p, q, r: -1.0,
            "cQ_beta": lambda a, b, m, re, p, q, r: -2.0,
            "cn_beta": lambda a, b, m, re, p, q, r: 2.0,
        },
        name="asym",
    )

    rocket.add_surfaces(asymmetric, positions=-1.0)

    # Warning fires once, on the first aerodynamic-center evaluation.
    with pytest.warns(UserWarning, match="not\\s+axisymmetric"):
        ac_pitch = rocket.aerodynamic_center.get_value_opt(0.2)

    assert ac_pitch != pytest.approx(rocket.aerodynamic_center_yaw.get_value_opt(0.2))
    assert rocket.static_margin.get_value_opt(0) != pytest.approx(
        rocket.static_margin_yaw.get_value_opt(0)
    )


def test_barrowman_surface_uses_generic_compute_path():
    """Barrowman surfaces must route through the shared generic
    ``compute_forces_and_moments`` (no bespoke override) and apply their force
    at the origin (moment carried by the coefficients)."""
    from rocketpy.rocket.aero_surface.generic_surface import GenericSurface

    nose = NoseCone(
        length=0.55829, kind="vonkarman", base_radius=0.0635, rocket_radius=0.0635
    )
    assert isinstance(nose, GenericSurface)
    # Force is applied at the origin; the cp offset lives in cm/cn.
    assert tuple(nose.force_application_point) == (0, 0, 0)
    assert (
        nose.compute_forces_and_moments.__func__
        is GenericSurface.compute_forces_and_moments
    )
