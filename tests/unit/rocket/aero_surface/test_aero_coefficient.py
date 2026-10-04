"""Unit tests for the AeroCoefficient minimal-dimension coefficient store."""

import functools

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


def test_repr_constant_and_function():
    assert "0.5" in repr(AeroCoefficient(0.5, (), name="cD"))
    function_repr = repr(AeroCoefficient(lambda mach: mach, ("mach",), name="cL"))
    assert "depends_on" in function_repr and "mach" in function_repr


# -- Independent-variable axes (control) --------------------------------------


def test_build_independent_vars_base_and_controls():
    assert build_independent_vars() == IV
    assert build_independent_vars(control_variables=("defl",)) == IV + ["defl"]


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
    with pytest.raises(ValueError, match="Cannot tell which variables"):
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
    f = Function(lambda a, b: a + b, ["x", "y"], "cL")
    with pytest.raises(ValueError, match="must have 7 input arguments"):
        AeroCoefficient(f, name="cL")


def test_from_input_function_with_named_inputs():
    """A Function whose inputs are named after the variables depends on them."""
    f = Function(lambda m, a: 10 * m + a, ["mach", "alpha"], "cL")
    coeff = AeroCoefficient(f, name="cL")
    assert coeff.depends_on == ("mach", "alpha")
    assert coeff(0.2, 0, 0.5, 0, 0, 0, 0) == pytest.approx(5.2)


def test_evaluator_takes_only_the_given_variables():
    coeff = AeroCoefficient(lambda mach, alpha: 10 * mach + alpha, name="cL")
    evaluate = coeff.evaluator(["alpha", "beta", "mach"])
    assert evaluate(0.2, 99, 0.5) == pytest.approx(5.2)
    assert AeroCoefficient(0.3).evaluator(["alpha"])(99) == 0.3


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
        control_variables=("deflection",),
        name="cL",
    )
    rebuilt = AeroCoefficient.from_dict(original.to_dict())
    assert rebuilt.control_variables == ("deflection",)
    assert rebuilt.independent_vars == original.independent_vars


# -- _infer_single_var fallbacks ----------------------------------------------


def test_infer_single_var_unmatched_label_gives_none():
    f = Function(lambda gamma: gamma, "gamma", "cD")
    assert AeroCoefficient._infer_single_var(f, IV) is None


def test_infer_single_var_missing_inputs_gives_none():
    class NoInputs:
        pass

    assert AeroCoefficient._infer_single_var(NoInputs(), IV) is None


@pytest.mark.parametrize(
    "label, expected",
    [
        ("mach", "mach"),
        ("Mach Number", "mach"),
        ("Pitch Rate", "pitch_rate"),
        ("Angle of attack alpha (rad)", "alpha"),
        ("Alphabet soup", None),
        ("machine", None),
        ("time (s)", None),
    ],
)
def test_infer_single_var_matches_whole_words_only(label, expected):
    f = Function([[0, 0.4], [1, 0.6]], label, "cD")
    assert AeroCoefficient._infer_single_var(f, IV) == expected


def _model_with_a_constant(alpha, mach, slope=2.0):
    return slope * alpha * (1 + mach)


def _seven_arguments_and_a_constant(a, b, m, re, q, r, p, gain=1.0):  # pylint: disable=unused-argument
    return gain * (a + m)


@pytest.mark.parametrize(
    "function, depends_on, expected",
    [
        (_model_with_a_constant, ("alpha", "mach"), 0.4),
        (functools.partial(_model_with_a_constant, slope=3.0), ("alpha", "mach"), 0.6),
        (lambda alpha, *, slope=2.0: slope * alpha, ("alpha",), 0.2),
        (lambda alpha, **options: 2 * alpha, ("alpha",), 0.2),
        (_seven_arguments_and_a_constant, tuple(IV), 1.1),
        (
            (lambda x, y, slope=2.0: slope * x * (1 + y), ["alpha", "mach"]),
            ("alpha", "mach"),
            0.4,
        ),
    ],
)
def test_function_arguments_with_a_default_are_not_variables(
    function, depends_on, expected
):
    """A function may carry constants of its own as arguments with a default
    value; only the arguments it must be given count as variables."""
    coeff = AeroCoefficient(function, name="cN")
    assert coeff.depends_on == depends_on
    assert coeff(0.1, 0, 1.0, 0, 0, 0, 0) == pytest.approx(expected)


@pytest.mark.parametrize("function", [lambda *args: 2 * args[0], lambda M: 0.5])
def test_function_that_names_no_variable_explains_what_to_do(function):
    with pytest.raises(ValueError, match="Cannot tell which variables"):
        AeroCoefficient(function, name="cN")


def test_slope_matches_the_analytic_derivative():
    coefficient = AeroCoefficient(lambda alpha, mach: 2 * alpha * (1 + mach) + alpha**3)
    slope = coefficient.slope("alpha", "mach", at={"alpha": 0.1})
    assert slope(0.5) == pytest.approx(2 * 1.5 + 3 * 0.1**2, rel=1e-6)
    assert coefficient.slope("alpha").get_value_opt(0) == pytest.approx(2.0)


@pytest.mark.parametrize(
    "call",
    [
        lambda c: c.slope("alfa", "mach"),
        lambda c: c.slope("alpha", "mahc"),
        lambda c: c.slope("alpha", "mach", at={"bta": 1.0}),
        lambda c: c.slice("alpha", at={"bta": 1.0}),
    ],
)
def test_slice_and_slope_reject_unknown_names_when_built(call):
    coefficient = AeroCoefficient(lambda alpha, mach: alpha * mach)
    with pytest.raises(ValueError, match="no independent variable"):
        call(coefficient)


def test_slice_rejects_a_repeated_name():
    coefficient = AeroCoefficient(lambda alpha, mach: alpha * mach)
    with pytest.raises(ValueError, match="more than once"):
        coefficient.slice("alpha", "alpha")
