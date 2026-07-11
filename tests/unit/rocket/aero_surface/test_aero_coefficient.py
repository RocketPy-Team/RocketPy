"""Unit tests for the AeroCoefficient minimal-dimension coefficient store."""

import pytest

from rocketpy import Function
from rocketpy.rocket.aero_surface.aero_coefficient import (
    AeroCoefficient,
    build_independent_vars,
)

IV = ["alpha", "beta", "mach", "reynolds", "pitch_rate", "yaw_rate", "roll_rate"]


# -- Construction & evaluation ------------------------------------------------


def test_constant_coefficient_is_zero_flagged():
    zero = AeroCoefficient(0, (), name="cD")
    assert zero.is_zero is True
    assert zero.is_zero_coefficient is True
    assert zero(0.1, 0.2, 0.3, 0, 0, 0, 0) == 0.0

    const = AeroCoefficient(0.7, (), name="cD")
    assert const.is_zero is False
    assert const(1, 2, 3, 4, 5, 6, 7) == 0.7
    assert const.get_value_opt(1, 2, 3, 4, 5, 6, 7) == 0.7


def test_call_is_get_value_opt():
    # __call__ is aliased to get_value_opt; both must behave identically.
    assert AeroCoefficient.__call__ is AeroCoefficient.get_value_opt


def test_mach_only_coefficient_maps_arguments():
    coeff = AeroCoefficient(lambda mach: 2 * mach, ("mach",), name="cL_alpha")
    assert coeff.depends_on == ("mach",)
    # Only the mach argument (index 2) should be used.
    assert coeff(99, 99, 0.3, 99, 99, 99, 99) == pytest.approx(0.6)
    assert coeff.get_value_opt(99, 99, 0.3, 99, 99, 99, 99) == pytest.approx(0.6)


def test_function_source_stored_directly():
    f = Function(lambda mach: mach**2, "mach", "cD")
    coeff = AeroCoefficient(f, ("mach",), name="cD")
    assert coeff.function is f
    assert coeff(0, 0, 0.5, 0, 0, 0, 0) == pytest.approx(0.25)


def test_depends_on_preserves_source_argument_order():
    # depends_on order must match the source's positional order, even when it
    # differs from the independent-variable order (e.g. shuffled CSV columns).
    coeff = AeroCoefficient(
        lambda mach, alpha: 10 * mach + alpha, ("mach", "alpha"), name="cL"
    )
    # full args: alpha=1 (idx0), mach=2 (idx2) -> source(mach=2, alpha=1) = 21
    assert coeff(1, 0, 2, 0, 0, 0, 0) == pytest.approx(21)


def test_unknown_dependency_raises():
    with pytest.raises(ValueError, match="unknown variable"):
        AeroCoefficient(lambda x: x, ("bogus",), name="cL")


def test_dom_dim_matches_full_arity():
    coeff = AeroCoefficient(0, (), name="cD")
    assert coeff.__dom_dim__ == len(IV)


def test_repr_constant_and_function():
    assert "0.5" in repr(AeroCoefficient(0.5, (), name="cD"))
    function_repr = repr(AeroCoefficient(lambda mach: mach, ("mach",), name="cL"))
    assert "depends_on" in function_repr and "mach" in function_repr


# -- Independent-variable axes (unsteady / control) ---------------------------


def test_build_independent_vars_base_unsteady_and_controls():
    assert build_independent_vars() == IV
    assert build_independent_vars(unsteady_aero=True) == IV + ["alpha_dot", "beta_dot"]
    assert build_independent_vars(control_variables=("defl",)) == IV + ["defl"]


def test_unsteady_aero_extends_independent_vars():
    coeff = AeroCoefficient(
        lambda alpha_dot: alpha_dot, ("alpha_dot",), unsteady_aero=True, name="cL"
    )
    assert coeff.independent_vars == tuple(IV + ["alpha_dot", "beta_dot"])
    assert coeff.__dom_dim__ == 9
    # alpha_dot is the 8th argument (index 7).
    assert coeff(0, 0, 0, 0, 0, 0, 0, 1.5, 0) == pytest.approx(1.5)


def test_control_variable_axis_is_appended():
    coeff = AeroCoefficient(
        lambda deflection: 2 * deflection,
        ("deflection",),
        control_variables=("deflection",),
        name="cL",
    )
    assert coeff.independent_vars[-1] == "deflection"
    assert coeff(0, 0, 0, 0, 0, 0, 0, 4) == pytest.approx(8)


# -- constructor inference: scalar -------------------------------------------------------


def test_from_input_scalar():
    coeff = AeroCoefficient(0, name="cm")
    assert coeff.is_zero is True


def test_from_input_non_numeric_raises():
    with pytest.raises(TypeError, match="must be a number"):
        AeroCoefficient(object(), name="cD")


# -- constructor inference: callable -----------------------------------------------------


def test_from_input_full_arity_callable():
    coeff = AeroCoefficient(lambda a, b, m, r, p, q, rr: a + m, name="cL")
    assert coeff.depends_on == tuple(IV)
    assert coeff(0.1, 0, 0.3, 0, 0, 0, 0) == pytest.approx(0.4)


def test_from_input_named_subset_callable():
    coeff = AeroCoefficient(lambda alpha, mach: alpha * mach, name="cL")
    assert coeff.depends_on == ("alpha", "mach")
    assert coeff(2, 0, 3, 0, 0, 0, 0) == pytest.approx(6)


def test_from_input_rejects_unmappable_callable():
    with pytest.raises(ValueError, match="callable must accept"):
        AeroCoefficient(lambda x, y, z: x, name="cL")


# -- constructor inference: Function -----------------------------------------------------


def test_from_input_full_dim_function():
    f = Function(lambda a, b, m, r, p, q, rr: a + m, IV, "cL")
    coeff = AeroCoefficient(f, name="cL")
    assert coeff.depends_on == tuple(IV)
    assert coeff(0.1, 0, 0.3, 0, 0, 0, 0) == pytest.approx(0.4)


def test_from_input_1d_function_infers_mach():
    f = Function(lambda mach: mach**2, "Mach", "cD")
    coeff = AeroCoefficient(f, name="cD")
    assert coeff.depends_on == ("mach",)
    assert coeff(0, 0, 0.5, 0, 0, 0, 0) == pytest.approx(0.25)


def test_from_input_function_with_bad_dimension_raises():
    f = Function(lambda a, b: a + b, ["alpha", "beta"], "cL")
    with pytest.raises(ValueError, match="must have 7 input arguments"):
        AeroCoefficient(f, name="cL")


# -- constructor inference: CSV path -----------------------------------------------------


def test_from_input_csv_loads_at_minimal_dimension(tmp_path):
    csv_file = tmp_path / "coeffs.csv"
    csv_file.write_text("mach,cD\n0.0,0.0\n1.0,3.0\n2.0,6.0\n")

    coeff = AeroCoefficient(str(csv_file), name="cD")
    assert coeff.depends_on == ("mach",)
    assert coeff(0, 0, 2, 0, 0, 0, 0) == pytest.approx(6)


def test_load_csv_rejects_unknown_column(tmp_path):
    csv_file = tmp_path / "coeffs.csv"
    csv_file.write_text("bogus,cD\n0.0,0.0\n1.0,3.0\n")

    with pytest.raises(ValueError, match="Invalid independent variable"):
        AeroCoefficient(str(csv_file), name="cD")


# -- constructor inference: AeroCoefficient round trip -----------------------------------


def test_roundtrip_callable_passthrough():
    original = AeroCoefficient(lambda alpha, mach: alpha + mach, name="cL")
    rebuilt = AeroCoefficient(original, name="cL")
    assert rebuilt.depends_on == original.depends_on
    assert rebuilt(0.5, 0, 0.3, 0, 0, 0, 0) == pytest.approx(
        original(0.5, 0, 0.3, 0, 0, 0, 0)
    )


def test_roundtrip_constant_passthrough():
    original = AeroCoefficient(0.9, name="cD")
    rebuilt = AeroCoefficient(original, name="cD")
    assert rebuilt._constant == pytest.approx(0.9)
    assert rebuilt(1, 2, 3, 4, 5, 6, 7) == pytest.approx(0.9)


def test_to_dict_from_dict_preserves_axes():
    original = AeroCoefficient(
        lambda deflection: deflection,
        ("deflection",),
        unsteady_aero=True,
        control_variables=("deflection",),
        name="cL",
    )
    rebuilt = AeroCoefficient.from_dict(original.to_dict())
    assert rebuilt.unsteady_aero is True
    assert rebuilt.control_variables == ("deflection",)
    assert rebuilt.independent_vars == original.independent_vars


# -- _infer_single_var fallbacks ----------------------------------------------


def test_infer_single_var_unmatched_label_defaults_to_first():
    f = Function(lambda gamma: gamma, "gamma", "cD")
    assert AeroCoefficient._infer_single_var(f, IV) == IV[0]


def test_infer_single_var_missing_inputs_defaults_to_first():
    class NoInputs:
        pass

    assert AeroCoefficient._infer_single_var(NoInputs(), IV) == IV[0]
