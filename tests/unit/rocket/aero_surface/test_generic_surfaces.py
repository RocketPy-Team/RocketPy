import json
from types import SimpleNamespace

import pytest

from rocketpy import Function, GenericSurface, LinearGenericSurface
from rocketpy._encoders import RocketPyDecoder, RocketPyEncoder
from rocketpy.mathutils import Vector


def _rpy_round_trip(obj):
    """Encode ``obj`` and decode it back through the .rpy encoder/decoder."""
    return json.loads(json.dumps(obj, cls=RocketPyEncoder), cls=RocketPyDecoder)


REFERENCE_AREA = 1
REFERENCE_LENGTH = 1


@pytest.mark.parametrize(
    "coefficients",
    [
        "cN",
        {"invalid_name": 0},
        {"cN": "inexistent_file.csv"},
        {"cN": Function(lambda x1, x2, x3, x4, x5, x6: 0)},
        {"cN": lambda x1: 0},
        {"cN": {}},
    ],
)
def test_invalid_initialization(coefficients):
    """Checks if generic surface raises errors in initialization
    when coefficient argument is invalid"""

    with pytest.raises((ValueError, TypeError)):
        GenericSurface(
            reference_area=REFERENCE_AREA,
            reference_length=REFERENCE_LENGTH,
            coefficients=coefficients,
        )


def test_invalid_initialization_from_csv(filename_invalid_coeff):
    """Checks if generic surfaces raises errors when initialized incorrectly
    from a csv file"""
    with pytest.raises(ValueError):
        GenericSurface(
            reference_area=REFERENCE_AREA,
            reference_length=REFERENCE_LENGTH,
            coefficients={"cN": str(filename_invalid_coeff)},
        )


@pytest.mark.parametrize(
    "coefficients",
    [
        {},
        {"cN": 0},
        {
            "cN": 0,
            "cY": Function(lambda x1, x2, x3, x4, x5, x6, x7: 0),
            "cA": lambda x1, x2, x3, x4, x5, x6, x7: 0,
        },
    ],
)
def test_valid_initialization(coefficients):
    """Checks if generic surface initializes correctly when coefficient
    argument is valid"""

    GenericSurface(
        reference_area=REFERENCE_AREA,
        reference_length=REFERENCE_LENGTH,
        coefficients=coefficients,
    )


def test_valid_initialization_from_csv(filename_valid_coeff):
    """Checks if generic surfaces initializes correctly when
    coefficients is set from a csv file"""
    GenericSurface(
        reference_area=REFERENCE_AREA,
        reference_length=REFERENCE_LENGTH,
        coefficients={"cN": str(filename_valid_coeff)},
    )


def test_csv_independent_variables_accept_any_order(tmp_path):
    """Checks if GenericSurface correctly maps CSV columns by header names,
    regardless of independent variable column order."""
    filename = tmp_path / "valid_coefficients_shuffled_order.csv"
    filename.write_text(
        "mach,alpha,cN\n0,0,0\n0,1,10\n2,0,2\n2,1,12\n",
        encoding="utf-8",
    )

    generic_surface = GenericSurface(
        reference_area=REFERENCE_AREA,
        reference_length=REFERENCE_LENGTH,
        coefficients={"cN": str(filename)},
    )

    # The coefficient is stored at minimal dimension over its CSV columns, in
    # header order; AeroCoefficient maps the full argument tuple onto them.
    assert generic_surface.cN.depends_on == ("mach", "alpha")
    csv_function = generic_surface.cN.function

    assert generic_surface.cN(1, 0, 2, 0, 0, 0, 0) == pytest.approx(12)
    assert csv_function.get_interpolation_method() == "regular_grid"


POINTS = [[0, 0], [1, 1], [2, 4], [3, 9]]


def test_interpolation_extrapolation_scalar_applies_to_all():
    """A single interpolation/extrapolation string is applied to every
    tabulated coefficient."""
    gs = GenericSurface(
        reference_area=REFERENCE_AREA,
        reference_length=REFERENCE_LENGTH,
        coefficients={"cN": POINTS, "cA": POINTS},
        extrapolation="constant",
        interpolation="akima",
    )
    for coeff in (gs.cN, gs.cA):
        assert coeff.function.get_interpolation_method() == "akima"
        assert coeff.function.get_extrapolation_method() == "constant"


def test_interpolation_extrapolation_per_coefficient_dict():
    """A dict configures interpolation/extrapolation per coefficient; omitted
    coefficients keep the default."""
    gs = GenericSurface(
        reference_area=REFERENCE_AREA,
        reference_length=REFERENCE_LENGTH,
        coefficients={"cN": POINTS, "cA": POINTS},
        extrapolation={"cA": "constant"},
        interpolation={"cN": "akima"},
    )
    assert gs.cN.function.get_interpolation_method() == "akima"
    assert gs.cA.function.get_extrapolation_method() == "constant"
    # cN was not in the extrapolation dict, so it keeps the tabulated default.
    assert gs.cN.function.get_interpolation_method() == "akima"
    assert gs.cA.function.get_interpolation_method() == "linear"


def test_prebuilt_function_interpolation_left_unchanged():
    """A pre-built Function keeps its own interpolation/extrapolation when none
    is requested, and is copied (not mutated) when they are overridden."""
    source = Function(POINTS, interpolation="spline", extrapolation="zero")

    unchanged = GenericSurface(REFERENCE_AREA, REFERENCE_LENGTH, {"cN": source})
    assert unchanged.cN.function.get_interpolation_method() == "spline"
    assert unchanged.cN.function.get_extrapolation_method() == "zero"

    overridden = GenericSurface(
        REFERENCE_AREA,
        REFERENCE_LENGTH,
        {"cN": source},
        interpolation="linear",
        extrapolation="constant",
    )
    assert overridden.cN.function.get_interpolation_method() == "linear"
    # The original Function must not have been mutated in place.
    assert source.get_interpolation_method() == "spline"


def test_tabulated_coefficient_defaults_to_constant_extrapolation():
    """Tabulated coefficients default to constant extrapolation, so they do not
    run to non-physical values past their data."""
    gs = GenericSurface(
        reference_area=REFERENCE_AREA,
        reference_length=REFERENCE_LENGTH,
        coefficients={"cN": POINTS},
    )
    assert gs.cN.function.get_extrapolation_method() == "constant"


def _write_grid_csv(path):
    """A 4x4 (mach, alpha) Cartesian grid, nonlinear in alpha so interpolation
    methods produce distinguishable values. 4 points per axis lets "cubic" fit.
    """
    rows = ["mach,alpha,cN"]
    for mach in (0, 1, 2, 3):
        for alpha in (0.0, 0.1, 0.2, 0.3):
            rows.append(f"{mach},{alpha},{mach + 10 * alpha**2}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return str(path)


@pytest.mark.parametrize(
    "interpolation, expected_grid_method",
    [("linear", "linear"), ("spline", "cubic"), ("akima", "pchip")],
)
def test_grid_csv_interpolation_maps_to_scipy_method(
    tmp_path, interpolation, expected_grid_method
):
    """A gridded CSV honors the interpolation argument by mapping it onto the
    RegularGridInterpolator method (no silent fallback to shepard)."""
    filename = _write_grid_csv(tmp_path / "grid.csv")

    gs = GenericSurface(
        reference_area=REFERENCE_AREA,
        reference_length=REFERENCE_LENGTH,
        coefficients={"cN": filename},
        interpolation=interpolation,
    )
    function = gs.cN.function
    # The Function stays a regular grid (not clobbered to shepard) ...
    assert function.get_interpolation_method() == "regular_grid"
    # ... with the mapped scipy method threaded through.
    assert getattr(function, "_grid_method", "linear") == expected_grid_method


def test_grid_csv_cubic_differs_from_linear(tmp_path):
    """The mapped grid method actually changes interpolation off the grid nodes,
    confirming it is not ignored."""
    filename = _write_grid_csv(tmp_path / "grid.csv")

    linear = GenericSurface(
        REFERENCE_AREA, REFERENCE_LENGTH, {"cN": filename}, interpolation="linear"
    )
    cubic = GenericSurface(
        REFERENCE_AREA, REFERENCE_LENGTH, {"cN": filename}, interpolation="spline"
    )
    # Interior off-node point at alpha=0.15, mach=0.5 (argument order is
    # alpha, beta, mach, ...): the nonlinear-in-alpha grid makes cubic and
    # linear disagree there.
    args = (0.15, 0.0, 0.5, 0, 0, 0, 0)
    assert linear.cN(*args) != pytest.approx(cubic.cN(*args))


def test_compute_forces_and_moments():
    """Checks if there are not logical errors in
    compute forces and moments"""

    gs_object = GenericSurface(REFERENCE_AREA, REFERENCE_LENGTH, {})
    forces_and_moments = gs_object.compute_forces_and_moments(
        stream_velocity=Vector((0, 0, 0)),
        stream_speed=0,
        stream_mach=0,
        rho=0,
        cp=Vector((0, 0, 0)),
        omega=(0, 0, 0),
        density=Function(1.0),
        dynamic_viscosity=Function(1.0),
        z=0,
    )
    assert forces_and_moments == (0, 0, 0, 0, 0, 0)


def test_angular_rates_are_non_dimensionalized():
    """Coefficients receive the conventional reduced rate q* = q L_ref / (2 V),
    not the raw body rate in rad/s."""
    ref_area, ref_length = 2.0, 0.5
    # Roll-moment coefficient that simply returns the roll rate it is given, so
    # the resulting roll moment exposes which rate value reached the coefficient.
    gs = GenericSurface(ref_area, ref_length, {"cl": lambda roll_rate: roll_rate})

    rho, speed, raw_roll = 1.2, 10.0, 4.0
    *_, roll_moment = gs.compute_forces_and_moments(
        stream_velocity=Vector((0, 0, -speed)),  # along centerline -> alpha=beta=0
        stream_speed=speed,
        stream_mach=0,
        rho=rho,
        cp=Vector((0, 0, 0)),
        omega=(0, 0, raw_roll),  # raw body roll rate p, rad/s
        density=Function(1.0),
        dynamic_viscosity=Function(1.0),
        z=0,
    )

    reduced_roll = raw_roll * ref_length / (2 * speed)
    dyn_pressure_area_length = 0.5 * rho * speed**2 * ref_area * ref_length
    # The coefficient saw the reduced rate, ...
    assert roll_moment == pytest.approx(dyn_pressure_area_length * reduced_roll)
    # ... not the raw rad/s rate.
    assert roll_moment != pytest.approx(dyn_pressure_area_length * raw_roll)


class _ExplodingAtmosphere:
    """Stand-in for density/viscosity whose lookup raises, so a test can assert
    the Reynolds computation (and thus the lookup) is skipped."""

    def get_value_opt(self, z):
        raise AssertionError("atmosphere lookup should have been skipped")


def test_reynolds_length_defaults_to_reference_length():
    gs = GenericSurface(REFERENCE_AREA, 0.2, {"cN": 0})
    assert gs.reynolds_length == 0.2


def test_reynolds_length_override():
    gs = GenericSurface(REFERENCE_AREA, 0.2, {"cN": 0}, reynolds_length=4.0)
    assert gs.reynolds_length == 4.0
    # The moment/rate reference length is left untouched.
    assert gs.reference_length == 0.2


def test_needs_reynolds_reflects_coefficient_dependence():
    without = GenericSurface(
        REFERENCE_AREA, REFERENCE_LENGTH, {"cN": lambda mach: mach}
    )
    assert without._needs_reynolds is False

    with_re = GenericSurface(
        REFERENCE_AREA, REFERENCE_LENGTH, {"cN": lambda reynolds: reynolds}
    )
    assert with_re._needs_reynolds is True


def test_reynolds_computation_skipped_when_no_coefficient_uses_it():
    """A surface with no Reynolds-dependent coefficient must not perform the
    per-step atmosphere lookups (the exploding stand-ins would raise if it did)."""
    gs = GenericSurface(REFERENCE_AREA, REFERENCE_LENGTH, {"cN": lambda mach: mach})
    gs.compute_forces_and_moments(
        stream_velocity=Vector((0, 0, -100)),
        stream_speed=100,
        stream_mach=0.3,
        rho=1.0,
        cp=Vector((0, 0, 0)),
        omega=(0, 0, 0),
        density=_ExplodingAtmosphere(),
        dynamic_viscosity=_ExplodingAtmosphere(),
        z=0,
    )


def test_reynolds_uses_reynolds_length_not_reference_length():
    """The Reynolds number handed to the coefficients is built on
    ``reynolds_length``, not the (diameter) reference length."""
    ref_area, ref_length, re_length = 1.0, 0.2, 4.0
    rho_atm, mu, speed, rho = 1.2, 2.0e-5, 100.0, 1.0
    # cN returns the Reynolds number it is given, so the normal force exposes it.
    gs = GenericSurface(
        ref_area,
        ref_length,
        {"cN": lambda reynolds: reynolds},
        reynolds_length=re_length,
    )

    _, r2, *_ = gs.compute_forces_and_moments(
        stream_velocity=Vector((0, 0, -speed)),  # centerline -> alpha=beta=0
        stream_speed=speed,
        stream_mach=0.3,
        rho=rho,
        cp=Vector((0, 0, 0)),
        omega=(0, 0, 0),
        density=Function(rho_atm),
        dynamic_viscosity=Function(mu),
        z=0,
    )

    # R2 = -normal = -(0.5 rho V^2 A_ref) * Re_seen
    reynolds_seen = -r2 / (0.5 * rho * speed**2 * ref_area)
    assert reynolds_seen == pytest.approx(rho_atm * speed * re_length / mu)
    # ... which differs from the diameter-based value.
    assert reynolds_seen != pytest.approx(rho_atm * speed * ref_length / mu)


def _fake_flight(burn_out_time):
    """Minimal stand-in exposing only what ``is_active`` reads
    (``flight.rocket.motor.burn_out_time``), so no real Flight is built."""
    return SimpleNamespace(
        rocket=SimpleNamespace(motor=SimpleNamespace(burn_out_time=burn_out_time))
    )


def test_active_during_defaults_to_always():
    """By default a surface is active at every time."""
    gs = GenericSurface(REFERENCE_AREA, REFERENCE_LENGTH, {"cN": 1})
    flight = _fake_flight(burn_out_time=3.0)
    assert gs.active_during == "always"
    assert gs.is_active(0.0, flight) is True
    assert gs.is_active(5.0, flight) is True


def test_active_during_power_on_gates_at_burnout():
    """A power-on surface is active up to (not including) burnout."""
    gs = GenericSurface(
        REFERENCE_AREA, REFERENCE_LENGTH, {"cN": 1}, active_during="power_on"
    )
    flight = _fake_flight(burn_out_time=3.0)
    assert gs.is_active(2.999, flight) is True
    assert gs.is_active(3.0, flight) is False
    assert gs.is_active(4.0, flight) is False


def test_active_during_power_off_gates_at_burnout():
    """A power-off surface is active only from burnout onward."""
    gs = GenericSurface(
        REFERENCE_AREA, REFERENCE_LENGTH, {"cN": 1}, active_during="power_off"
    )
    flight = _fake_flight(burn_out_time=3.0)
    assert gs.is_active(2.999, flight) is False
    assert gs.is_active(3.0, flight) is True
    assert gs.is_active(4.0, flight) is True


def test_active_during_accepts_callable():
    """A custom predicate receives (t, flight) and drives activation."""
    seen = []

    def only_after_one_second(t, flight):
        seen.append((t, flight))
        return t > 1.0

    gs = GenericSurface(
        REFERENCE_AREA,
        REFERENCE_LENGTH,
        {"cN": 1},
        active_during=only_after_one_second,
    )
    flight = _fake_flight(burn_out_time=3.0)
    assert gs.is_active(0.5, flight) is False
    assert gs.is_active(2.0, flight) is True
    # The predicate was called with the time and the flight object.
    assert seen[0] == (0.5, flight)


def test_active_during_invalid_value_raises():
    """An unknown activation policy is rejected at construction."""
    with pytest.raises(ValueError, match="active_during"):
        GenericSurface(
            REFERENCE_AREA,
            REFERENCE_LENGTH,
            {"cN": 1},
            active_during="sometimes",
        )


def test_generic_surface_round_trips_through_encoder():
    """A GenericSurface survives the full .rpy encode/decode: coefficients,
    reynolds_length and a custom activation function are all restored."""
    gs = GenericSurface(
        reference_area=1.0,
        reference_length=0.2,
        coefficients={"cN": lambda mach: 2 * mach, "cm": 0.1},
        reynolds_length=4.0,
        active_during=lambda t, flight: t < 3.0,
    )
    restored = _rpy_round_trip(gs)

    assert isinstance(restored, GenericSurface)
    assert restored.reynolds_length == 4.0
    assert restored.cN(0, 0, 0.5, 0, 0, 0, 0) == pytest.approx(1.0)
    assert restored.cm(0, 0, 0, 0, 0, 0, 0) == pytest.approx(0.1)
    assert restored.active_during(1.0, None) is True
    assert restored.active_during(5.0, None) is False


def test_linear_generic_surface_round_trips_through_encoder():
    """A LinearGenericSurface restores its derivative coefficients and the
    Reynolds length through the .rpy encode/decode."""
    lgs = LinearGenericSurface(
        reference_area=1.0,
        reference_length=0.2,
        coefficients={"cN_alpha": 2.0, "cm_alpha": -0.5},
        reynolds_length=3.0,
    )
    restored = _rpy_round_trip(lgs)

    assert isinstance(restored, LinearGenericSurface)
    assert restored.reynolds_length == 3.0
    assert restored.cN_alpha(0, 0, 0, 0, 0, 0, 0) == pytest.approx(2.0)
    assert restored.cm_alpha(0, 0, 0, 0, 0, 0, 0) == pytest.approx(-0.5)


def test_generic_surface_preset_active_during_round_trips():
    """A preset activation policy round-trips as the plain string."""
    gs = GenericSurface(
        REFERENCE_AREA, REFERENCE_LENGTH, {"cN": 0}, active_during="power_on"
    )
    assert _rpy_round_trip(gs).active_during == "power_on"
