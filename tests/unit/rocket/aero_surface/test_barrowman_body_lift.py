"""Unit tests for the Galejs body-lift extension of ``_BarrowmanSurface``.

The body-lift hook adds a nonlinear ``sin²α`` term (Galejs, K = 1.1) to the
classic Barrowman normal force, applied at a blended centre of pressure
between the slender-body CP and the planform centroid. The tests pin down:

- the closed-form magnitude of the body-lift contribution,
- the low-speed / high-α (apogee) damping factor ``(M / 0.05)²``,
- the force-weighted CP blend between slender-body CP and planform centroid,
- the pure-tube limit where the slender-body term vanishes,
- backward compatibility: surfaces that do not opt in are unchanged.

Reference: Galejs, R., "Body Lift Extension to Barrowman's CP Calculation",
Apogee Rockets Newsletter, 2003; OpenRocket ``SymmetricComponentCalc``.
"""

import numpy as np
import pytest

from rocketpy import Function, NoseCone, Tail
from rocketpy.mathutils.vector_matrix import Vector
from rocketpy.rocket.aero_surface._barrowman_surface import _BarrowmanSurface

RHO = 1.225
SPEED = 100.0
MACH = 0.3


class _BodyLiftStub(_BarrowmanSurface):
    """Minimal Barrowman surface with constant clalpha and opt-in body lift.

    Defined only for exercising ``compute_forces_and_moments`` against
    closed-form expectations; mirrors how geometry-defined subclasses
    (nose cones, tails, future BodyTube) populate the planform attributes.
    """

    # pylint: disable=super-init-not-called
    def __init__(
        self,
        reference_area,
        clalpha_value=2.0,
        cpz_slender=0.0,
        planform_area=0.0,
        planform_centroid=0.0,
        cp_slender=0.0,
    ):
        self.name = "body lift stub"
        self.reference_area = reference_area
        self.reference_length = 2 * (reference_area / np.pi) ** 0.5
        self.clalpha = Function(lambda mach: clalpha_value)
        self.cpz = cpz_slender
        self._planform_area = planform_area
        self._planform_centroid = planform_centroid
        self._cp_slender = cp_slender

        # Attributes expected by GenericSurface.__init__ machinery.
        self._unsteady_aero = False
        self.control_variables = {}
        self.evaluate_coefficients()
        super().__init__(
            reference_area=reference_area,
            reference_length=self.reference_length,
            coefficients={},
            center_of_pressure=(0.0, 0.0, cpz_slender),
            name=self.name,
        )


def _velocity(alpha):
    """Stream velocity at a total angle of attack alpha, transverse along y."""
    return [0.0, SPEED * np.sin(alpha), -SPEED * np.cos(alpha)]


def _forces(surface, velocity, mach=MACH, cp=(0.0, 0.0, 0.0)):
    """Call compute_forces_and_moments with standard test conditions."""
    speed = float(np.linalg.norm(velocity))
    return surface.compute_forces_and_moments(
        Vector(velocity), speed, mach, RHO, Vector(cp), Vector([0, 0, 0])
    )


def test_default_surface_has_no_body_lift():
    """Surfaces that do not set a planform area must produce exactly the
    linear Barrowman force (backward compatibility)."""
    surface = _BodyLiftStub(reference_area=np.pi * 0.0635**2)

    _, r2, *_ = _forces(surface, _velocity(0.15))

    expected = 0.5 * RHO * SPEED**2 * surface.reference_area * 2.0 * 0.15
    assert r2 == pytest.approx(expected, rel=1e-12)


def test_body_lift_magnitude_matches_galejs_formula():
    """The total lift coefficient must equal the linear term plus
    K · (A_plan/A_ref) · sin²α."""
    ref_area = np.pi * 0.0635**2
    plan_area = 2 * 0.0635 * 0.5  # tube-like planform: diameter × length
    k = 1.1
    alpha = 0.35
    surface = _BodyLiftStub(reference_area=ref_area, planform_area=plan_area)

    _, r2, *_ = _forces(surface, _velocity(alpha))

    c_linear = 2.0 * alpha
    c_body = k * plan_area / ref_area * np.sin(alpha) ** 2
    expected = 0.5 * RHO * SPEED**2 * ref_area * (c_linear + c_body)

    assert r2 == pytest.approx(expected, rel=1e-12)


def test_low_speed_high_alpha_damping():
    """Below M = 0.05 with α > 45°, the body-lift term is damped by
    (M / 0.05)²."""
    ref_area = np.pi * 0.0635**2
    plan_area = 2 * 0.0635 * 0.5
    k = 1.1
    alpha = np.deg2rad(60)
    mach = 0.03
    slow_speed = mach * 340.0
    velocity = [0.0, slow_speed * np.sin(alpha), -slow_speed * np.cos(alpha)]
    surface = _BodyLiftStub(reference_area=ref_area, planform_area=plan_area)

    _, r2, *_ = _forces(surface, velocity, mach=mach)

    damping = (mach / 0.05) ** 2
    c_linear = 2.0 * alpha
    c_body = k * plan_area / ref_area * np.sin(alpha) ** 2 * damping
    expected = 0.5 * RHO * slow_speed**2 * ref_area * (c_linear + c_body)

    assert r2 == pytest.approx(expected, rel=1e-12)


def test_no_damping_below_45_degrees():
    """The damping factor applies only when α > 45°, even at very low Mach."""
    ref_area = np.pi * 0.0635**2
    plan_area = 2 * 0.0635 * 0.5
    k = 1.1
    alpha = np.deg2rad(30)
    mach = 0.03
    slow_speed = mach * 340.0
    velocity = [0.0, slow_speed * np.sin(alpha), -slow_speed * np.cos(alpha)]
    surface = _BodyLiftStub(reference_area=ref_area, planform_area=plan_area)

    _, r2, *_ = _forces(surface, velocity, mach=mach)

    c_linear = 2.0 * alpha
    c_body = k * plan_area / ref_area * np.sin(alpha) ** 2  # no damping
    expected = 0.5 * RHO * slow_speed**2 * ref_area * (c_linear + c_body)

    assert r2 == pytest.approx(expected, rel=1e-12)


def test_blended_cp_is_force_weighted_average():
    """The moment must match the force-weighted blend of the slender-body CP
    and the planform centroid."""
    ref_area = np.pi * 0.0635**2
    plan_area = 2 * 0.0635 * 0.5
    k = 1.1
    cp_slender = 0.10
    centroid = 0.25  # aft of the slender CP in local (nose→tail) z
    alpha = 0.30
    surface = _BodyLiftStub(
        reference_area=ref_area,
        planform_area=plan_area,
        planform_centroid=centroid,
        cp_slender=cp_slender,
    )

    # Pass the slender-body CP as the geometric application point, in the
    # body frame (the surface-local z runs nose→tail, hence the flip).
    forces = _forces(surface, _velocity(alpha), cp=(0.0, 0.0, -cp_slender))
    _, r2, _, m1, _, _ = forces

    c_linear = 2.0 * alpha
    c_body = k * plan_area / ref_area * np.sin(alpha) ** 2
    cp_blend = (c_linear * cp_slender + c_body * centroid) / (c_linear + c_body)
    # M1 = -(cp_z_body) × F2 transport about the origin.
    expected_m1 = cp_blend * r2

    assert m1 == pytest.approx(expected_m1, rel=1e-12)


def test_blended_cp_stays_between_the_two_contributions():
    """For any α the effective CP implied by the moment must lie between the
    slender-body CP and the planform centroid."""
    ref_area = np.pi * 0.0635**2
    plan_area = 2 * 0.0635 * 0.5
    cp_slender, centroid = 0.05, 0.40
    surface = _BodyLiftStub(
        reference_area=ref_area,
        planform_area=plan_area,
        planform_centroid=centroid,
        cp_slender=cp_slender,
    )

    lo, hi = sorted([cp_slender, centroid])
    for alpha_deg in (5, 15, 30, 60, 85):
        alpha = np.deg2rad(alpha_deg)
        forces = _forces(surface, _velocity(alpha), cp=(0.0, 0.0, -cp_slender))
        _, r2, _, m1, _, _ = forces

        c_linear = 2.0 * alpha
        c_body = 1.1 * plan_area / ref_area * np.sin(alpha) ** 2
        cp_blend = (
            c_linear * cp_slender + c_body * centroid
        ) / (c_linear + c_body)

        assert lo <= cp_blend <= hi
        assert m1 == pytest.approx(cp_blend * r2, rel=1e-12)


def test_pure_tube_no_nan_when_linear_term_vanishes():
    """A constant-radius tube has zero slender-body lift; the body-lift-only
    limit must not produce NaN/inf and must apply the force at the planform
    centroid."""
    cp_slender, centroid = 0.10, 0.25
    surface = _BodyLiftStub(
        reference_area=np.pi * 0.0635**2,
        clalpha_value=0.0,  # pure tube: no linear term at all
        planform_area=2 * 0.0635 * 0.5,
        planform_centroid=centroid,
        cp_slender=cp_slender,
    )
    alpha = np.deg2rad(80)

    forces = _forces(surface, _velocity(alpha), cp=(0.0, 0.0, -cp_slender))
    assert all(np.isfinite(f) for f in forces)

    # Force is purely from the Galejs term.
    _, r2, _, m1, _, _ = forces
    q_s = 0.5 * RHO * SPEED**2 * surface.reference_area
    c_body = 1.1 * (2 * 0.0635 * 0.5) / surface.reference_area * np.sin(alpha) ** 2
    assert r2 == pytest.approx(q_s * c_body, rel=1e-12)
    # Applied entirely at the planform centroid: M1 = -(cp_z_body) × R2
    # with the blended CP landing exactly on the centroid in the body frame.
    assert m1 == pytest.approx(centroid * r2, rel=1e-12)


def test_nose_cone_planform_matches_closed_form():
    """A real geometry subclass populates the planform attributes from its
    contour: for a conical nose, A_plan = R·L/2 with centroid at 2L/3 from
    the tip, and zeroing the planform recovers the legacy linear-only force."""
    nose = NoseCone(length=0.5, kind="conical", base_radius=0.05, rocket_radius=0.05)
    # The planform comes from trapezoidal integration of the contour (401
    # samples), so allow its discretization error (~1e-6 relative).
    assert nose._planform_area == pytest.approx(  # noqa: SLF001
        0.05 * 0.5 / 2, rel=1e-4
    )
    assert nose._planform_centroid == pytest.approx(  # noqa: SLF001
        2 * 0.5 / 3, rel=1e-4
    )
    assert nose._cp_slender == pytest.approx(nose.cpz)  # noqa: SLF001

    alpha = 0.20
    with_body = _forces(nose, _velocity(alpha))
    nose._planform_area = 0.0  # noqa: SLF001
    without_body = _forces(nose, _velocity(alpha))

    # Body lift adds force beyond the linear term at this alpha.
    assert with_body[1] > without_body[1]
    # And zeroing the planform reproduces the legacy linear Barrowman force.
    q_s = 0.5 * RHO * SPEED**2 * nose.reference_area
    assert without_body[1] == pytest.approx(
        q_s * 2 * (nose.radius_ratio**2) * alpha, rel=1e-12
    )


def test_tail_planform_matches_closed_form():
    """The tail's trapezoid planform is (r_top + r_bot)·L with centroid at
    L/3·(r_top + 2·r_bot)/(r_top + r_bot) from the top."""
    tail = Tail(
        top_radius=0.0635, bottom_radius=0.0435, length=0.06, rocket_radius=0.0635
    )
    assert tail._planform_area == pytest.approx(  # noqa: SLF001
        (0.0635 + 0.0435) * 0.06
    )
    expected_centroid = (
        0.06 / 3 * (0.0635 + 2 * 0.0435) / (0.0635 + 0.0435)
    )
    assert tail._planform_centroid == pytest.approx(expected_centroid)  # noqa: SLF001
    assert tail._cp_slender == pytest.approx(tail.cpz)  # noqa: SLF001
