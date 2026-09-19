"""Guard tests ensuring every concrete aerodynamic surface exposes the full
coefficient contract the rocket and flight code rely on.

Each surface must provide the six force/moment coefficients (``cL``, ``cQ``,
``cD``, ``cm``, ``cn``, ``cl``) and the four stability derivatives
(``cN_alpha``, ``cY_beta``, ``cm_alpha``, ``cn_beta``) as callables over its
independent-variable tuple, plus the pitch and yaw center-of-pressure accessors.
These tests catch a subclass silently omitting one.
"""

import numpy as np
import pytest

from rocketpy import (
    AirBrakes,
    ControllableGenericSurface,
    EllipticalFin,
    EllipticalFins,
    FreeFormFin,
    FreeFormFins,
    GenericSurface,
    LinearGenericSurface,
    NoseCone,
    Tail,
    TrapezoidalFin,
    TrapezoidalFins,
)

R = 0.0635  # a representative rocket radius, in meters
_SHAPE = [(0, 0), (0.08, 0.1), (0.12, 0.1), (0.12, 0)]


def _make_surfaces():
    """One instance of every concrete aerodynamic surface class."""
    area, length = np.pi * R**2, 2 * R
    return {
        "GenericSurface": GenericSurface(
            reference_area=area, reference_length=length, coefficients={"cL": 1.0}
        ),
        "LinearGenericSurface": LinearGenericSurface(
            reference_area=area,
            reference_length=length,
            coefficients={"cN_alpha": 2.0},
        ),
        "ControllableGenericSurface": ControllableGenericSurface(
            reference_area=area,
            reference_length=length,
            coefficients={"cL": lambda a, b, m, re, p, q, r, d: 1.0},
        ),
        "AirBrakes": AirBrakes(
            drag_coefficient_curve=lambda deployment, mach: 0.5,
            reference_area=area,
        ),
        "NoseCone": NoseCone(
            length=0.55829, kind="vonkarman", base_radius=R, rocket_radius=R
        ),
        "Tail": Tail(top_radius=R, bottom_radius=0.0435, length=0.06, rocket_radius=R),
        "TrapezoidalFins": TrapezoidalFins(
            n=4, span=0.1, root_chord=0.12, tip_chord=0.04, rocket_radius=R
        ),
        "EllipticalFins": EllipticalFins(
            n=4, span=0.1, root_chord=0.12, rocket_radius=R
        ),
        "FreeFormFins": FreeFormFins(n=4, shape_points=_SHAPE, rocket_radius=R),
        "TrapezoidalFin": TrapezoidalFin(
            angular_position=0,
            span=0.1,
            root_chord=0.12,
            tip_chord=0.04,
            rocket_radius=R,
        ),
        "EllipticalFin": EllipticalFin(
            angular_position=0, span=0.1, root_chord=0.12, rocket_radius=R
        ),
        "FreeFormFin": FreeFormFin(
            angular_position=0, shape_points=_SHAPE, rocket_radius=R
        ),
    }


SURFACES = _make_surfaces()

# All nine force/moment coefficients: the wind-frame forces (lift cL, side cQ,
# drag cD), the body-frame forces (normal cN, side cY, axial cA), and the moments
# (pitch cm, yaw cn, roll cl). The body-frame trio is derived from the wind trio
# (or vice versa) by the angle-of-attack/sideslip rotation. Note the
# case-sensitive distinction between ``cL`` (lift) and ``cl`` (roll).
FORCE_MOMENT_COEFFICIENTS = (
    "cL",
    "cQ",
    "cD",
    "cN",
    "cY",
    "cA",
    "cm",
    "cn",
    "cl",
)
STABILITY_DERIVATIVES = ("cN_alpha", "cY_beta", "cm_alpha", "cn_beta")


def _surface_params():
    return [pytest.param(name, id=name) for name in SURFACES]


def _coefficient_arguments(surface):
    """A representative independent-variable tuple for the surface: the seven
    base variables (alpha, beta, mach, reynolds, and the three rates) plus any
    control axes the surface adds, filled with zeros."""
    base = [0.05, 0.02, 0.5, 1e6, 0.0, 0.0, 0.0]
    extra = len(surface.independent_vars) - len(base)
    return tuple(base + [0.0] * max(0, extra))


@pytest.mark.parametrize("name", _surface_params())
@pytest.mark.parametrize("coefficient", FORCE_MOMENT_COEFFICIENTS)
def test_force_moment_coefficient_is_callable(name, coefficient):
    """Every surface exposes all nine force/moment coefficients (wind cL/cQ/cD,
    body cN/cY/cA, moments cm/cn/cl) as coefficients callable over the
    independent-variable tuple that return a finite value."""
    surface = SURFACES[name]
    coeff = getattr(surface, coefficient, None)
    assert coeff is not None, f"{name} is missing coefficient {coefficient}"
    value = coeff.get_value_opt(*_coefficient_arguments(surface))
    assert np.isfinite(value), f"{name}.{coefficient} returned {value}"


@pytest.mark.parametrize("name", _surface_params())
@pytest.mark.parametrize("derivative", STABILITY_DERIVATIVES)
def test_stability_derivative_is_callable(name, derivative):
    """Every surface exposes the stability derivatives cN_alpha, cY_beta,
    cm_alpha and cn_beta used by the rocket's center-of-pressure computation."""
    surface = SURFACES[name]
    coeff = getattr(surface, derivative, None)
    assert coeff is not None, f"{name} is missing derivative {derivative}"
    value = coeff.get_value_opt(*_coefficient_arguments(surface))
    assert np.isfinite(value), f"{name}.{derivative} returned {value}"


@pytest.mark.parametrize("name", _surface_params())
def test_center_of_pressure_accessors(name):
    """Every surface exposes the pitch and yaw center-of-pressure accessors used
    by the rocket's aerodynamic-center computation."""
    surface = SURFACES[name]
    for attr in ("center_of_pressure_z", "center_of_pressure_z_yaw"):
        accessor = getattr(surface, attr, None)
        assert accessor is not None, f"{name} is missing {attr}"
        assert np.isfinite(accessor.get_value_opt(0.5))


_ARGS = (0.15, 0.08, 0.5, 1e6, 0.0, 0.0, 0.0)


def test_body_input_is_recovered_by_body_accessors():
    """Coefficients supplied in the body frame are recovered by the body-frame
    accessors (they round-trip through the canonical wind-frame storage)."""
    from rocketpy import GenericSurface

    surface = GenericSurface(
        reference_area=0.01,
        reference_length=0.1,
        coefficients={
            "cN": lambda a, b, m, re, p, q, r: 2.0 * a,
            "cA": lambda a, b, m, re, p, q, r: 0.5,
            "cY": lambda a, b, m, re, p, q, r: 1.5 * b,
        },
    )
    assert surface.force_convention == "body"
    assert surface.cN.get_value_opt(*_ARGS) == pytest.approx(2.0 * _ARGS[0])
    assert surface.cA.get_value_opt(*_ARGS) == pytest.approx(0.5)
    assert surface.cY.get_value_opt(*_ARGS) == pytest.approx(1.5 * _ARGS[1])


def test_wind_and_body_input_agree_at_zero_angle():
    """cL == cN, cD == cA and cQ == cY at zero angle of attack and sideslip,
    regardless of the frame the coefficients were supplied in."""
    from rocketpy import GenericSurface

    wind = GenericSurface(
        reference_area=0.01,
        reference_length=0.1,
        coefficients={"cL": lambda a, b, m, re, p, q, r: 2.0},
        force_convention="wind",
    )
    body = GenericSurface(
        reference_area=0.01,
        reference_length=0.1,
        coefficients={"cN": lambda a, b, m, re, p, q, r: 2.0},
        force_convention="body",
    )
    zero = (0.0, 0.0, 0.5, 1e6, 0.0, 0.0, 0.0)
    assert wind.cN.get_value_opt(*zero) == pytest.approx(2.0)
    assert body.cL.get_value_opt(*zero) == pytest.approx(2.0)


def test_mixed_frame_input_raises():
    """Supplying both wind and body force coefficients without declaring the
    frame is rejected."""
    from rocketpy import GenericSurface

    with pytest.raises(ValueError, match="[Mm]ixed"):
        GenericSurface(
            reference_area=0.01,
            reference_length=0.1,
            coefficients={"cL": 1.0, "cN": 1.0},
        )
