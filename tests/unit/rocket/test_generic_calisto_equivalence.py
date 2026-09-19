"""Non-Flight equivalence checks for the generic-surface Calisto rockets.

``calisto_robust`` (Barrowman surfaces), ``calisto_linear_generic`` (per-surface
LinearGenericSurface), ``calisto_generic`` (per-surface GenericSurface) and
``calisto_full_aerodynamics`` (a single lumped full-body GenericSurface) all
describe the same Calisto aerodynamics. These tests pin that equivalence at the
coefficient/stability level -- without running a Flight -- so a regression in any
of the generic-surface code paths is caught cheaply. The full 6-DOF flight
comparison lives in ``tests/unit/simulation/test_flight.py``.
"""

import numpy as np
import pytest

GENERIC_FIXTURES = [
    "calisto_linear_generic",
    "calisto_generic",
    "calisto_full_aerodynamics",
]


@pytest.mark.parametrize("generic_name", GENERIC_FIXTURES)
def test_generic_calisto_matches_static_margin(request, calisto_robust, generic_name):
    """Each generic-surface Calisto reproduces the Barrowman rocket's static
    margin and Mach-dependent aerodynamic center."""
    generic = request.getfixturevalue(generic_name)
    assert generic.static_margin(0) == pytest.approx(
        calisto_robust.static_margin(0), rel=1e-3
    )
    for mach in (0.0, 0.3, 0.8, 1.2, 2.0):
        assert generic.aerodynamic_center.get_value_opt(mach) == pytest.approx(
            calisto_robust.aerodynamic_center.get_value_opt(mach), abs=1e-3
        )


@pytest.mark.parametrize("generic_name", GENERIC_FIXTURES)
def test_generic_calisto_matches_aggregate_slopes(
    request, calisto_robust, generic_name
):
    """Each generic-surface Calisto reproduces the Barrowman rocket's total
    normal-force and side-force curve slopes (the aggregate ``cN_alpha`` and
    ``cY_beta`` the whole rocket presents) across a range of Mach numbers."""
    generic = request.getfixturevalue(generic_name)
    for mach in (0.2, 0.6, 0.9, 1.5, 2.5):
        assert generic.total_lift_coeff_der.get_value_opt(mach) == pytest.approx(
            calisto_robust.total_lift_coeff_der.get_value_opt(mach), rel=2e-3
        )
        assert generic.total_side_coeff_der.get_value_opt(mach) == pytest.approx(
            calisto_robust.total_side_coeff_der.get_value_opt(mach), rel=2e-3
        )


def test_generic_and_linear_calisto_are_identical(
    calisto_generic, calisto_linear_generic
):
    """The per-surface GenericSurface and LinearGenericSurface Calistos express
    the same coefficients through different code paths, so their aggregate
    force-curve slopes and aerodynamic centers must agree to numerical
    precision."""
    for mach in (0.2, 0.7, 1.3, 2.5):
        for attr in (
            "total_lift_coeff_der",
            "total_side_coeff_der",
            "aerodynamic_center",
            "aerodynamic_center_yaw",
        ):
            generic = getattr(calisto_generic, attr).get_value_opt(mach)
            linear = getattr(calisto_linear_generic, attr).get_value_opt(mach)
            assert generic == pytest.approx(linear, rel=1e-9, abs=1e-12)
