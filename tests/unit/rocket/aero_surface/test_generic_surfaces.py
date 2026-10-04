import json
import warnings
from types import SimpleNamespace

import numpy as np
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
    assert csv_function.is_regular_grid


# A one-input table, given with the name of its variable
POINTS = ([[0, 0], [1, 1], [2, 4], [3, 9]], ["mach"])


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
    source = Function(
        POINTS[0], "mach", "cN", interpolation="spline", extrapolation="zero"
    )

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
    assert function.is_regular_grid
    # ... interpolated with the mapped grid method.
    assert function.get_interpolation_method() == expected_grid_method


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


# Directions of the rocket's velocity relative to the air, in the body frame:
# small and large combined angles, tail-first flight and flow along one axis.
_FLOW_DIRECTIONS = [
    (0.0, 0.1, 1.0),
    (0.1, 0.0, 1.0),
    (0.1, 0.1, 1.0),
    (0.6, 0.6, 1.0),
    (-2.0, 3.0, 0.4),
    (0.0, 0.1, -1.0),
    (0.3, -0.2, -1.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
]


def _wind_input_force(coefficients, direction):
    """Body-frame force of a unit-area surface in a unit dynamic pressure flow."""
    surface = GenericSurface(1.0, 1.0, coefficients)
    velocity = Vector(direction)
    velocity = velocity / abs(velocity)
    force = surface.compute_forces_and_moments(
        -velocity,
        1.0,
        0.1,
        2.0,
        Vector([0, 0, 0]),
        (0, 0, 0),
        Function(1.0),
        Function(1.0),
        0,
    )[:3]
    return Vector(force), velocity


@pytest.mark.parametrize("direction", _FLOW_DIRECTIONS)
def test_wind_frame_drag_opposes_the_velocity(direction):
    """A drag coefficient given in the wind frame must give a force pointing
    exactly against the velocity, at any combination of angle of attack and
    sideslip, including tail-first flight."""
    force, velocity = _wind_input_force({"cD": 1.0}, direction)
    assert list(force) == pytest.approx(list(-velocity), abs=1e-12)


@pytest.mark.parametrize("direction", _FLOW_DIRECTIONS)
def test_wind_frame_lift_and_side_force_are_perpendicular_to_velocity(direction):
    """Lift and side force must be unit forces perpendicular to the velocity and
    to each other, with the lift lying in the body y-z plane."""
    lift, velocity = _wind_input_force({"cL": 1.0}, direction)
    side, _ = _wind_input_force({"cQ": 1.0}, direction)
    assert abs(lift) == pytest.approx(1.0)
    assert abs(side) == pytest.approx(1.0)
    assert lift @ velocity == pytest.approx(0.0, abs=1e-12)
    assert side @ velocity == pytest.approx(0.0, abs=1e-12)
    assert lift @ side == pytest.approx(0.0, abs=1e-12)
    assert lift[0] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    "alpha, beta", [(0.1, 0.0), (0.0, 0.2), (0.3, 0.2), (2.8, 3.0)]
)
def test_wind_frame_views_recover_the_wind_input(alpha, beta):
    """The ``cL``/``cD``/``cQ`` views of a surface built from wind-frame
    coefficients must give those coefficients back."""
    surface = GenericSurface(
        1.0,
        1.0,
        {
            "cL": lambda alpha, mach: 2 * alpha,
            "cD": 0.4,
            "cQ": lambda beta: -1.5 * beta,
        },
    )
    args = (alpha, beta, 0.5, 0, 0, 0, 0)
    assert surface.cL(*args) == pytest.approx(2 * alpha)
    assert surface.cD(*args) == pytest.approx(0.4)
    assert surface.cQ(*args) == pytest.approx(-1.5 * beta)


def test_wind_input_depends_only_on_what_it_uses():
    """A wind-frame input converted to the body frame depends on the angle of
    attack and sideslip plus whatever the input itself uses, so a constant drag
    does not make the surface look up the Reynolds number."""
    surface = GenericSurface(1.0, 1.0, {"cD": 0.5})
    assert surface.cA.depends_on == ("alpha", "beta")
    assert surface._needs_reynolds is False

    surface = GenericSurface(
        1.0,
        1.0,
        {
            "cD": ([[0, 0.4], [1, 0.6]], ["mach"]),
            "cL": lambda alpha, mach: 2 * alpha,
        },
    )
    assert surface.cN.depends_on == ("alpha", "beta", "mach")

    surface = GenericSurface(1.0, 1.0, {"cD": lambda reynolds: 1e-7 * reynolds})
    assert surface._needs_reynolds is True


def test_wind_input_honors_interpolation_and_extrapolation():
    """The interpolation and extrapolation given under a wind-frame coefficient's
    name must reach its table."""
    table = Function([[0.0, 0.4], [1.0, 0.6]], "mach", "cD", extrapolation="constant")
    held = GenericSurface(1.0, 1.0, {"cD": table})
    zeroed = GenericSurface(1.0, 1.0, {"cD": table}, extrapolation={"cD": "zero"})
    outside = (0.0, 0.0, 2.0, 0, 0, 0, 0)  # beyond the table
    assert held.cA(*outside) == pytest.approx(0.6)
    assert zeroed.cA(*outside) == pytest.approx(0.0)


def test_wind_views_are_built_once_and_follow_the_coefficients():
    surface = GenericSurface(1.0, 1.0, {"cN": lambda alpha: 2 * alpha, "cA": 0.5})
    view = surface.cD
    assert surface.cD is view
    surface.cA = surface._as_coefficient(0.8, "cA")
    assert surface.cD is not view
    assert surface.cD(0, 0, 0, 0, 0, 0, 0) == pytest.approx(0.8)


def test_one_input_table_needs_its_variable_named(tmp_path):
    """A one-input table must say which variable it is: a Mach curve must never
    be read against the angle of attack by default."""
    points = [[0, 0.4], [1, 0.6]]
    csv_file = tmp_path / "curve.csv"
    csv_file.write_text("0.0,0.4\n1.0,0.6\n")

    for source in (points, str(csv_file), Function(points)):
        with pytest.raises(ValueError, match="Cannot tell which variable"):
            GenericSurface(1.0, 1.0, {"cA": source})

    for source in (points, str(csv_file), Function(points)):
        surface = GenericSurface(1.0, 1.0, {"cA": (source, ["mach"])})
        assert surface.cA.depends_on == ("mach",)
        assert surface.cA(0.3, 0.0, 0.5, 0, 0, 0, 0) == pytest.approx(0.5)


def test_coefficient_given_with_the_names_of_its_variables():
    """``(source, variables)`` names the inputs of a table or of a function, in
    order, for any number of inputs."""
    points = [[a, m, 2 * a + m] for a in (-0.1, 0.0, 0.1) for m in (0.0, 1.0)]
    surface = GenericSurface(
        1.0,
        1.0,
        {
            "cN": (points, ["alpha", "mach"]),
            "cm": (lambda x, y: x - 3 * y, ["mach", "alpha"]),
            "cA": (0.5, []),
        },
    )
    args = (0.1, 0.0, 1.0, 0, 0, 0, 0)
    assert surface.cN.depends_on == ("alpha", "mach")
    assert surface.cN(*args) == pytest.approx(1.2)
    assert surface.cm.depends_on == ("mach", "alpha")
    assert surface.cm(*args) == pytest.approx(0.7)
    assert surface.cA(*args) == 0.5

    with pytest.raises(ValueError, match="1 input.*2 variable name"):
        GenericSurface(1.0, 1.0, {"cA": ([[0, 0.4], [1, 0.6]], ["alpha", "mach"])})
    with pytest.raises(ValueError, match="unknown variable"):
        GenericSurface(1.0, 1.0, {"cA": ([[0, 0.4], [1, 0.6]], ["speed"])})


def _grid_table():
    """A smooth coefficient sampled on an alpha-Mach grid, as rows and as a block
    of values."""
    alphas = np.radians(np.arange(-10, 11, 2.0))
    machs = np.arange(0, 3.01, 0.25)
    alpha_grid, mach_grid = np.meshgrid(alphas, machs, indexing="ij")
    values = (2 + 0.5 * mach_grid) * np.sin(alpha_grid)
    rows = np.column_stack([alpha_grid.ravel(), mach_grid.ravel(), values.ravel()])
    return alphas, machs, values, rows


def test_the_same_grid_gives_the_same_coefficient_in_every_form(tmp_path):
    """A table on a regular grid must be read the same way from a CSV file, a
    list, a numpy array (in any row order) and as axes with a block of values."""
    alphas, machs, values, rows = _grid_table()
    csv_file = tmp_path / "cN.csv"
    np.savetxt(csv_file, rows, delimiter=",", header="alpha,mach,cN", comments="")
    shuffled = np.random.default_rng(1).permutation(rows)
    sources = [
        str(csv_file),
        (rows.tolist(), ["alpha", "mach"]),
        (rows, ["alpha", "mach"]),
        (shuffled, ["alpha", "mach"]),
        ({"alpha": alphas, "mach": machs}, values),
        ({"mach": machs, "alpha": alphas}, values.T),
    ]
    results = []
    for source in sources:
        coefficient = GenericSurface(1.0, 1.0, {"cN": source}).cN
        assert coefficient.function.is_regular_grid
        results.append(coefficient(0.05, 0.0, 0.6, 0, 0, 0, 0))
    assert results == pytest.approx([results[0]] * len(results))
    assert results[0] == pytest.approx((2 + 0.5 * 0.6) * np.sin(0.05), rel=2e-3)


def test_incomplete_grid_and_one_variable_tables_stay_ordinary_tables():
    _, machs, _, rows = _grid_table()
    scattered = GenericSurface(1.0, 1.0, {"cN": (rows[:-1], ["alpha", "mach"])}).cN
    assert scattered.function.get_interpolation_method() == "linear"

    for source in (
        (np.column_stack([machs, 0.4 + 0.1 * machs]), ["mach"]),
        ({"mach": machs}, 0.4 + 0.1 * machs),
    ):
        curve = GenericSurface(1.0, 1.0, {"cA": source}).cA
        assert curve.depends_on == ("mach",)
        assert curve.function.get_interpolation_method() == "linear"
        assert curve(0.0, 0.0, 0.6, 0, 0, 0, 0) == pytest.approx(0.46)


def test_grid_values_must_match_their_axes():
    alphas, machs, values, _ = _grid_table()
    with pytest.raises(ValueError, match="The values of cN have shape"):
        GenericSurface(1.0, 1.0, {"cN": ({"alpha": alphas, "mach": machs}, values.T)})


def _degrees_table():
    """``cN = (2 + 0.5 * mach) * alpha`` tabulated against the angle of attack in
    degrees and Mach, as rows and as a block of values."""
    degrees = np.arange(-10, 11, 2.0)
    machs = np.array([0.0, 1.0, 2.0])
    degree_grid, mach_grid = np.meshgrid(degrees, machs, indexing="ij")
    values = (2 + 0.5 * mach_grid) * np.radians(degree_grid)
    rows = np.column_stack([degree_grid.ravel(), mach_grid.ravel(), values.ravel()])
    return degrees, machs, values, rows


def test_angle_in_degrees_in_every_form(tmp_path):
    """A source that names its angle ``alpha_deg`` is read in degrees, whichever
    form it is given in, while the coefficient is still called in radians."""
    degrees, machs, values, rows = _degrees_table()
    csv_file = tmp_path / "cN.csv"
    np.savetxt(csv_file, rows, delimiter=",", header="alpha_deg,mach,cN", comments="")
    sources = [
        str(csv_file),
        (rows, ["alpha_deg", "mach"]),
        ({"alpha_deg": degrees, "mach": machs}, values),
        lambda alpha_deg, mach: (2 + 0.5 * mach) * np.radians(alpha_deg),
    ]
    alpha = np.radians(3.0)
    for source in sources:
        coefficient = GenericSurface(1.0, 1.0, {"cN": source}).cN
        assert coefficient.depends_on == ("alpha", "mach")
        assert coefficient.in_degrees == ("alpha",)
        assert coefficient(alpha, 0.0, 1.0, 0, 0, 0, 0) == pytest.approx(2.5 * alpha)

    curve = Function(np.column_stack([degrees, np.radians(degrees)]), "beta_deg", "cY")
    side = GenericSurface(1.0, 1.0, {"cY": curve}).cY
    assert side.depends_on == ("beta",) and side.in_degrees == ("beta",)
    assert side(0.0, alpha, 0.0, 0, 0, 0, 0) == pytest.approx(alpha)


def test_angle_in_degrees_survives_slope_scaling_and_save_and_load():
    _, _, _, rows = _degrees_table()
    coefficient = GenericSurface(1.0, 1.0, {"cN": (rows, ["alpha_deg", "mach"])}).cN
    alpha = np.radians(3.0)
    args = (alpha, 0.0, 1.0, 0, 0, 0, 0)

    # The slope is per radian, whatever unit the table is in
    assert coefficient.slope("alpha", "mach")(1.0) == pytest.approx(2.5)
    assert (coefficient * 2)(*args) == pytest.approx(5.0 * alpha)
    restored = _rpy_round_trip(coefficient)
    assert restored.in_degrees == ("alpha",)
    assert restored(*args) == pytest.approx(2.5 * alpha)

    with pytest.raises(ValueError, match="more than once"):
        GenericSurface(1.0, 1.0, {"cN": (lambda a, b: a, ["alpha", "alpha_deg"])})


def test_angle_axis_in_degrees_read_as_radians_warns():
    """An angle axis that goes beyond what an angle in radians can be warns that
    the table looks like it is in degrees; a table in radians does not."""
    degrees, _, _, rows = _degrees_table()
    with pytest.warns(UserWarning, match="looks like degrees"):
        GenericSurface(1.0, 1.0, {"cN": (rows, ["alpha", "mach"])})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        in_radians = np.column_stack([np.radians(degrees), np.radians(degrees)])
        GenericSurface(1.0, 1.0, {"cN": (in_radians, ["alpha"])})
        GenericSurface(1.0, 1.0, {"cN": (rows, ["alpha_deg", "mach"])})
        # Other variables may be as large as they like
        GenericSurface(1.0, 1.0, {"cA": ([[0, 0.4], [10, 0.6]], ["mach"])})


def _write_multi_coefficient_csv(path, header, only_positive_angles=False):
    """One table with three coefficients against the angle of attack in degrees
    and Mach: ``cN = (2 + 0.5 mach) alpha``, ``cA = 0.4 + 0.1 mach`` and
    ``cm = -3 alpha``."""
    degrees = np.arange(0 if only_positive_angles else -10, 11, 2.0)
    machs = np.array([0.0, 1.0, 2.0])
    degree_grid, mach_grid = np.meshgrid(degrees, machs, indexing="ij")
    alpha_grid = np.radians(degree_grid)
    table = np.column_stack(
        [
            degree_grid.ravel(),
            mach_grid.ravel(),
            ((2 + 0.5 * mach_grid) * alpha_grid).ravel(),
            (0.4 + 0.1 * mach_grid).ravel(),
            (-3 * alpha_grid).ravel(),
        ]
    )
    np.savetxt(path, table, delimiter=",", header=header, comments="")


def test_from_csv_reads_several_coefficients_from_one_file(tmp_path):
    csv_file = tmp_path / "aero.csv"
    _write_multi_coefficient_csv(csv_file, "alpha_deg, mach, cN, cA, cm")

    surface = GenericSurface.from_csv(str(csv_file), 1.3, 0.7, name="One file")

    alpha = np.radians(3.0)
    args = (alpha, 0.0, 1.0, 0, 0, 0, 0)
    assert surface.name == "One file"
    assert surface.reference_area == 1.3 and surface.reference_length == 0.7
    assert surface.cN(*args) == pytest.approx(2.5 * alpha)
    assert surface.cA(*args) == pytest.approx(0.5)
    assert surface.cm(*args) == pytest.approx(-3 * alpha)
    assert surface.cY.is_zero
    for name in ("cN", "cA", "cm"):
        coefficient = getattr(surface, name)
        assert coefficient.depends_on == ("alpha", "mach")
        assert coefficient.function.is_regular_grid


def test_from_csv_translates_and_ignores_columns(tmp_path):
    """With ``columns``, a file written by another program loads as it is: the
    listed columns are translated and the others are ignored."""
    csv_file = tmp_path / "export.csv"
    _write_multi_coefficient_csv(csv_file, "Alpha,Mach,CN,CA Power-Off,CMY")
    columns = {"Alpha": "alpha_deg", "Mach": "mach", "CN": "cN", "CA Power-Off": "cA"}

    surface = GenericSurface.from_csv(
        str(csv_file), 1.0, 1.0, columns=columns, active_during="power_off"
    )

    alpha = np.radians(3.0)
    assert surface.cN(alpha, 0.0, 1.0, 0, 0, 0, 0) == pytest.approx(2.5 * alpha)
    assert surface.cm.is_zero  # CMY was not listed
    assert surface.active_during == "power_off"

    with pytest.raises(ValueError, match="neither a variable"):
        GenericSurface.from_csv(str(csv_file), 1.0, 1.0)
    with pytest.raises(ValueError, match="not found"):
        GenericSurface.from_csv(str(csv_file), 1.0, 1.0, columns={"Beta": "beta"})
    with pytest.raises(ValueError, match="at least one variable column"):
        GenericSurface.from_csv(str(csv_file), 1.0, 1.0, columns={"Mach": "mach"})


def test_table_with_only_positive_angles_warns(tmp_path):
    """A force table that starts at zero angle, where it is zero, would give no
    force at negative angles: warn. A drag table (not zero there) does not."""
    csv_file = tmp_path / "one_sided.csv"
    _write_multi_coefficient_csv(
        csv_file, "alpha_deg, mach, cN, cA, cm", only_positive_angles=True
    )
    with pytest.warns(UserWarning, match="no negative angles") as caught:
        GenericSurface.from_csv(str(csv_file), 1.0, 1.0)
    warned = {str(warning.message).split()[3] for warning in caught}
    assert warned == {"cN", "cm"}


def _total_angle_surface():
    """A surface with a nonlinear normal force, a pitch moment, an axial force
    and a roll moment, all given against the total angle of attack."""
    return GenericSurface(
        1.0,
        1.0,
        {
            "cN": lambda alpha_total, mach: (
                (2 + 0.5 * mach) * np.sin(alpha_total) + 1.5 * np.sin(alpha_total) ** 3
            ),
            "cm": lambda alpha_total: -3 * np.sin(alpha_total),
            "cA": lambda alpha_total: 0.4 + alpha_total**2,
            "cl": 0.01,
        },
    )


@pytest.mark.parametrize(
    "stream_velocity",
    [
        (3.0, -2.0, -100.0),
        (20.0, 35.0, -80.0),
        (-50.0, 10.0, -60.0),
        (5.0, 0.0, -100.0),
        (0.0, -7.0, -100.0),
        (0.0, 0.0, -100.0),
        (30.0, 30.0, 40.0),
    ],
)
def test_total_angle_convention_gives_the_force_along_the_crossflow(stream_velocity):
    """Coefficients against the total angle of attack: the normal force must act
    along the crossflow of the air and the pitch moment about the axis across
    it, whatever the direction of the wind, tail-first flow included."""
    surface = _total_angle_surface()
    mach, rho = 0.5, 2.0
    stream = Vector(stream_velocity)
    speed = abs(stream)
    r1, r2, r3, m1, m2, m3 = surface.compute_forces_and_moments(
        stream, speed, mach, rho, Vector([0, 0, 0]), (0, 0, 0), None, None, 0
    )

    # Built from the velocity directly, with no partial angles involved
    u_x, u_y, u_z = (-component / speed for component in stream_velocity)
    crossflow = np.hypot(u_x, u_y)
    alpha_total = np.arctan2(crossflow, u_z)
    scale = 0.5 * rho * speed**2
    normal = scale * (
        (2 + 0.5 * mach) * np.sin(alpha_total) + 1.5 * np.sin(alpha_total) ** 3
    )
    moment = scale * -3 * np.sin(alpha_total)
    direction = (u_x / crossflow, u_y / crossflow) if crossflow else (0.0, 0.0)

    assert r1 == pytest.approx(-normal * direction[0], abs=1e-9 * scale)
    assert r2 == pytest.approx(-normal * direction[1], abs=1e-9 * scale)
    assert r3 == pytest.approx(-scale * (0.4 + alpha_total**2))
    assert m1 == pytest.approx(moment * direction[1], abs=1e-9 * scale)
    assert m2 == pytest.approx(-moment * direction[0], abs=1e-9 * scale)
    assert m3 == pytest.approx(scale * 0.01)


def test_total_angle_convention_slopes_and_dependencies():
    surface = _total_angle_surface()
    assert surface.force_convention == "body"
    assert surface.cN.depends_on == ("alpha", "beta", "mach")
    assert surface.cm.depends_on == ("alpha", "beta")
    assert surface.cn.depends_on == ("alpha", "beta")
    assert surface._needs_reynolds is False
    # The same slope in both planes, with the side force sign of the body frame
    at_mach_1 = (0.0, 0.0, 1.0, 0, 0, 0, 0)
    assert surface.cN_alpha(*at_mach_1) == pytest.approx(2.5, rel=1e-6)
    assert surface.cY_beta(*at_mach_1) == pytest.approx(-2.5, rel=1e-6)
    assert surface.cm_alpha(*at_mach_1) == pytest.approx(-3.0, rel=1e-6)
    assert surface.cn_beta(*at_mach_1) == pytest.approx(3.0, rel=1e-6)


def test_total_angle_names_and_the_guard():
    """``alpha_total`` may be used by any coefficient. One with a direction acts
    in the plane of the wind and is split for you, unless it also takes the
    roll angle of the wind, which leaves the split to its source."""
    drag = GenericSurface(
        1.0, 1.0, {"cA": lambda alpha_total_deg: 0.4 + 0.01 * alpha_total_deg}
    )
    assert drag.cA.depends_on == ("alpha", "beta")
    assert drag.cA(np.radians(3), 0.0, 0, 0, 0, 0, 0) == pytest.approx(0.43)
    assert drag.cA(0.0, np.radians(-3), 0, 0, 0, 0, 0) == pytest.approx(0.43)

    by_hand = GenericSurface(
        1.0, 1.0, {"cN": lambda alpha_total, phi: 2 * alpha_total * np.sin(phi)}
    )
    assert by_hand.cN(-0.1, 0.0, 0, 0, 0, 0, 0) == pytest.approx(-0.2)

    # Named after the variable alone: no option is needed
    split = GenericSurface(1.0, 1.0, {"cN": lambda alpha_total: 2 * alpha_total})
    assert split.force_convention == "body"
    assert split.cN(-0.1, 0.0, 0, 0, 0, 0, 0) == pytest.approx(-0.2)
    assert split.cY(0.0, 0.1, 0, 0, 0, 0, 0) == pytest.approx(-0.2)

    # There is no side force or yaw moment in the plane of the wind
    for name in ("cY", "cn", "cQ"):
        with pytest.raises(ValueError, match="no side force or yaw moment"):
            GenericSurface(1.0, 1.0, {name: lambda alpha_total: alpha_total})
    # and the part in the other plane comes from the split
    with pytest.raises(ValueError, match="cY cannot be given together"):
        GenericSurface(
            1.0, 1.0, {"cN": lambda alpha_total: alpha_total, "cY": lambda beta: beta}
        )
    # A derivative only multiplies its own angle, so the total angle is just
    # another variable it may depend on
    linear = LinearGenericSurface(
        1.0, 1.0, {"cN_alpha": lambda alpha_total: 2.0 + alpha_total}
    )
    assert linear.cN(0.1, 0.0, 0, 0, 0, 0, 0) == pytest.approx(0.21)
    assert linear.cN(-0.1, 0.0, 0, 0, 0, 0, 0) == pytest.approx(-0.21)
    assert linear.cY(0.0, 0.1, 0, 0, 0, 0, 0) == pytest.approx(0.0)


def test_lift_and_drag_against_the_total_angle():
    """Lift and drag given against the total angle of attack give the same
    surface as the normal and axial forces they amount to."""

    def lift(alpha_total, mach):
        return (2 + 0.2 * mach) * alpha_total

    def drag(alpha_total):
        return 0.4 + alpha_total**2

    wind = GenericSurface(1.0, 1.0, {"cL": lift, "cD": drag})
    body = GenericSurface(
        1.0,
        1.0,
        {
            "cN": lambda alpha_total, mach: (
                lift(alpha_total, mach) * np.cos(alpha_total)
                + drag(alpha_total) * np.sin(alpha_total)
            ),
            "cA": lambda alpha_total, mach: (
                drag(alpha_total) * np.cos(alpha_total)
                - lift(alpha_total, mach) * np.sin(alpha_total)
            ),
        },
    )
    assert wind.force_convention == "wind"
    for alpha, beta in ((0.1, 0.0), (0.0, -0.2), (0.15, 0.1), (-0.3, 0.05)):
        state = (alpha, beta, 0.6, 0, 0, 0, 0)
        for name in ("cN", "cY", "cA", "cL", "cD"):
            assert getattr(wind, name)(*state) == pytest.approx(
                getattr(body, name)(*state), abs=1e-12
            )
    at_zero = (0.0, 0.0, 0.6, 0, 0, 0, 0)
    assert wind.cN_alpha(*at_zero) == pytest.approx(body.cN_alpha(*at_zero), rel=1e-6)
    # In the pitch plane the lift read back is the lift given
    assert wind.cL(0.2, 0.0, 0.6, 0, 0, 0, 0) == pytest.approx(lift(0.2, 0.6))

    with pytest.raises(ValueError, match="must be zero there"):
        GenericSurface(1.0, 1.0, {"cL": lambda alpha_total: 1.0})
    with pytest.raises(ValueError, match="cQ cannot be given together"):
        GenericSurface(1.0, 1.0, {"cL": lift, "cQ": lambda beta: beta})


def test_total_angle_table_from_one_file_and_save_and_load(tmp_path):
    """A table against the total angle of attack in degrees, which only has
    positive angles, loads from one file and needs no mirroring."""
    degrees = np.arange(0, 21, 2.0)
    machs = np.array([0.0, 1.0, 2.0])
    degree_grid, mach_grid = np.meshgrid(degrees, machs, indexing="ij")
    table = np.column_stack(
        [
            degree_grid.ravel(),
            mach_grid.ravel(),
            ((2 + 0.5 * mach_grid) * np.radians(degree_grid)).ravel(),
            (0.4 + 0.1 * mach_grid).ravel(),
        ]
    )
    csv_file = tmp_path / "export.csv"
    np.savetxt(csv_file, table, delimiter=",", header="Alpha,Mach,CN,CA", comments="")
    columns = {"Alpha": "alpha_total_deg", "Mach": "mach", "CN": "cN", "CA": "cA"}

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        surface = GenericSurface.from_csv(str(csv_file), 1.0, 1.0, columns=columns)

    angle = np.radians(3.0)
    assert surface.cN(-angle, 0.0, 1.0, 0, 0, 0, 0) == pytest.approx(-2.5 * angle)
    assert surface.cY(0.0, angle, 1.0, 0, 0, 0, 0) == pytest.approx(-2.5 * angle)
    assert surface.cA(-angle, 0.0, 1.0, 0, 0, 0, 0) == pytest.approx(0.5)

    # Saved in the body frame, so it loads back as an ordinary surface
    restored = _rpy_round_trip(surface)
    assert restored.cN(-angle, 0.0, 1.0, 0, 0, 0, 0) == pytest.approx(-2.5 * angle)


def test_total_angle_coefficient_must_vanish_at_zero_angle():
    """A normal force or pitch moment against the total angle of attack has no
    direction at zero angle, so it must be zero there. A table that starts
    above zero degrees holds its first value down to zero and is refused, as
    is a function with an intercept."""
    table = [[angle, mach, 0.05 * angle] for angle in (2, 4, 10, 20) for mach in (0, 1)]
    with pytest.raises(ValueError, match="cN is 0.1 at zero total angle"):
        GenericSurface(
            1.0,
            1.0,
            {"cN": (table, ["alpha_total_deg", "mach"])},
        )
    with pytest.raises(ValueError, match="cm is 0.02 at zero total angle"):
        GenericSurface(
            1.0,
            1.0,
            {
                "cN": lambda alpha_total: 2 * np.sin(alpha_total),
                "cm": lambda alpha_total: 0.02 - alpha_total,
            },
        )


def test_total_angle_table_from_zero_gives_the_right_slope():
    """The same table with its 0 degree row gives the tabulated slope, and a
    body-frame force that is continuous through zero angle."""
    table = [
        [angle, mach, 0.05 * angle] for angle in (0, 2, 4, 10, 20) for mach in (0, 1)
    ]
    surface = GenericSurface(
        1.0,
        1.0,
        {"cN": (table, ["alpha_total_deg", "mach"])},
    )
    assert surface.cN_alpha(0, 0, 0.5, 0, 0, 0, 0) == pytest.approx(
        np.degrees(0.05), rel=1e-6
    )
    small = 1e-4
    assert surface.cN(small, 0, 0.5, 0, 0, 0, 0) == pytest.approx(
        -surface.cN(-small, 0, 0.5, 0, 0, 0, 0)
    )
    assert abs(surface.cN(small, 0, 0.5, 0, 0, 0, 0)) < 1e-2


@pytest.mark.parametrize(
    "coefficients, wrong",
    [
        (
            {"cN": lambda alpha_total, alpha: alpha_total + alpha},
            "cN is given against alpha_total together with alpha",
        ),
        (
            {
                "cN": lambda alpha_total: alpha_total,
                "cm": lambda alpha_total, beta: -alpha_total * beta,
            },
            "cm is given against alpha_total together with beta",
        ),
    ],
)
def test_total_angle_refuses_signed_angles(coefficients, wrong):
    """Against the total angle of attack the split supplies the sign, so a
    normal force or pitch moment that also reads the signed alpha or beta
    would carry it twice."""
    with pytest.raises(ValueError, match=wrong):
        GenericSurface(1.0, 1.0, coefficients)


def test_total_angle_with_phi_is_split_by_its_source():
    """A force that depends on the roll angle of the wind is given against
    ``alpha_total`` and ``phi``, with the split along the crossflow written by
    its source; Mach, Reynolds and the rates stay valid."""
    surface = GenericSurface(
        1.0,
        1.0,
        {
            "cN": lambda alpha_total, phi, mach, reynolds, pitch_rate: (
                (2 + 0.1 * mach)
                * np.sin(alpha_total)
                * (1 + 0.1 * np.cos(2 * phi))
                * np.sin(phi)
            ),
            "cA": lambda alpha: 0.4 + alpha**2,
        },
    )
    assert surface.cN(0.1, 0, 0.5, 0, 0, 0, 0) == pytest.approx(
        -surface.cN(-0.1, 0, 0.5, 0, 0, 0, 0)
    )
    assert surface.cN(0.1, 0, 0.5, 0, 0, 0, 0) > 0


# Review of 2026-09-26: input checks, center of pressure, orientation, prints


def _area_length():
    return np.pi * 0.0635**2, 2 * 0.0635


def test_a_one_argument_function_is_read_by_its_name():
    """The Mach hint for unnamed one-input sources must not override a name
    that is a variable; a name that is not one still means Mach."""
    from rocketpy import AeroCoefficient

    assert AeroCoefficient(lambda alpha: alpha, single_var="mach").depends_on == (
        "alpha",
    )
    assert AeroCoefficient(lambda reynolds: 1.0, single_var="mach").depends_on == (
        "reynolds",
    )
    assert AeroCoefficient(lambda m: m, single_var="mach").depends_on == ("mach",)
    assert AeroCoefficient([[0, 0.4], [1, 0.5]], single_var="mach").depends_on == (
        "mach",
    )


def test_csv_header_may_have_spaces_after_the_commas(tmp_path):
    path = tmp_path / "cN.csv"
    path.write_text(
        '"alpha", "mach", "coefficient"\n0,0.3,0\n0.1,0.3,0.2\n0,0.9,0\n0.1,0.9,0.25\n'
    )
    surface = GenericSurface(*_area_length(), {"cN": str(path)})
    assert surface.cN(0.1, 0, 0.3, 0, 0, 0, 0) == pytest.approx(0.2)


@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        ({"coefficients": "cN.csv"}, TypeError, "dict"),
        ({"coefficients": {"cN": 0}, "center_of_pressure": 0.3}, TypeError, "tuple"),
        ({"coefficients": {"cN": lambda alpha, *, mach: alpha}}, ValueError, "cN"),
        ({"coefficients": {"cN": lambda alpha: None}}, ValueError, "cN"),
        (
            {"coefficients": {"cN": lambda mach, alpha, x, y, z, w, v: mach}},
            ValueError,
            "cN",
        ),
        (
            {"coefficients": {"cN": lambda alpha, mach=0.0: alpha * mach}},
            ValueError,
            "mach",
        ),
        (
            {
                "coefficients": {"cN": ([[0, 0], [1, 0.4]], ["mach"])},
                "interpolation": "splin",
            },
            ValueError,
            "splin",
        ),
        ({"coefficients": {"cN": (0.4, ["mach"])}}, ValueError, "constant"),
        (
            {"coefficients": {"cN": ([[0, 0.4], [0, 0.5], [1, 0.6]], ["mach"])}},
            ValueError,
            "same input",
        ),
        (
            {
                "coefficients": {
                    "cN": ({"alpha": [0, 0.1, 0.2], "mach": [0, 1]}, np.zeros((2, 3)))
                }
            },
            ValueError,
            "shape",
        ),
    ],
)
def test_bad_inputs_are_rejected_at_construction(kwargs, error, match):
    """Inputs that used to be misread or to fail only in flight are refused
    when the surface is built, with the coefficient named."""
    with pytest.raises(error, match=match):
        GenericSurface(*_area_length(), **kwargs)


def test_duplicate_csv_variable_is_rejected(tmp_path):
    path = tmp_path / "dup.csv"
    path.write_text("alpha,alpha,cN\n0,0,0\n0.1,0.1,0.2\n0.2,0.2,0.4\n")
    with pytest.raises(ValueError, match="more than once"):
        GenericSurface(*_area_length(), {"cN": str(path)})


def test_a_function_labelled_in_degrees_is_read_in_degrees():
    table = Function([[0, 0], [10, 0.5], [20, 1.0]], inputs="Alpha (deg)", outputs="cN")
    surface = GenericSurface(*_area_length(), {"cN": table})
    assert surface.cN(np.radians(10), 0, 0.3, 0, 0, 0, 0) == pytest.approx(0.5)


def test_wind_and_total_angle_tables_round_trip_without_pickling():
    area, length = _area_length()
    wind = GenericSurface(
        area,
        length,
        {
            "cL": (
                [[0, 0, 0], [0.1, 0, 0.2], [0, 1, 0], [0.1, 1, 0.25]],
                ["alpha", "mach"],
            ),
            "cD": 0.4,
        },
    )
    total = GenericSurface(
        area,
        length,
        {
            "cN": (
                [[0, 0, 0], [10, 0, 0.5], [0, 1, 0], [10, 1, 0.6]],
                ["alpha_total_deg", "mach"],
            )
        },
    )
    for surface in (wind, total):
        data = json.dumps(surface, cls=RocketPyEncoder, allow_pickle=False)
        loaded = json.loads(data, cls=RocketPyDecoder)
        args = (0.1, 0.05, 0.5, 0, 0, 0, 0)
        assert loaded.cN(*args) == pytest.approx(surface.cN(*args))
        assert loaded.cY(*args) == pytest.approx(surface.cY(*args))


def test_setting_the_center_of_pressure_updates_the_rocket():
    from rocketpy import Rocket

    surface = GenericSurface(*_area_length(), {"cN": lambda alpha: 2 * alpha})
    rocket = Rocket(
        radius=0.0635,
        mass=14.426,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag=0.5,
        power_on_drag=0.5,
        center_of_mass_without_motor=0,
    )
    rocket.add_surfaces(surface, 0.0)
    before = rocket.aerodynamic_center(0.3)
    surface.center_of_pressure = (0, 0, 0.3)
    assert rocket.aerodynamic_center(0.3) == pytest.approx(before + 0.3)


def test_center_of_pressure_given_against_mach():
    """A center of pressure that moves with Mach is folded into the moment,
    so it equals a hand-written ``cm``, and ``aerodynamic_center`` reports
    it."""
    area, length = _area_length()
    xcp = [[0, -0.3], [1, -0.2], [2, -0.1]]
    with_xcp = GenericSurface(
        area,
        length,
        {"cN": lambda alpha, mach: (2 + mach) * alpha},
        center_of_pressure=(0, 0, xcp),
    )
    by_hand = GenericSurface(
        area,
        length,
        {
            "cN": lambda alpha, mach: (2 + mach) * alpha,
            "cm": lambda alpha, mach: (
                (2 + mach)
                * alpha
                * np.interp(mach, [0, 1, 2], [-0.3, -0.2, -0.1])
                / length
            ),
        },
    )
    state = (
        Vector([0, -10, -100]),
        100.5,
        0.5,
        1.2,
        Vector([0, 0, 0]),
        (0, 0, 0),
        None,
        None,
        0,
    )
    assert with_xcp.compute_forces_and_moments(*state) == pytest.approx(
        by_hand.compute_forces_and_moments(*state)
    )
    assert with_xcp.aerodynamic_center(0.5) == pytest.approx(-0.25)


def test_linear_surface_hints_and_defaults():
    """A derivative tabulated in one column is against Mach; a value name
    instead of a derivative name is pointed to the right class."""
    area, length = _area_length()
    surface = LinearGenericSurface(
        area, length, {"cN_alpha": [[0, 2], [1, 2.5], [2, 3]]}
    )
    assert surface.cN_alpha(0, 0, 1, 0, 0, 0, 0) == pytest.approx(2.5)
    with pytest.raises(ValueError, match="cN_alpha"):
        LinearGenericSurface(area, length, {"cN": 2.0})


def test_surface_prints_show_the_body_coefficients(capsys):
    surface = GenericSurface(
        *_area_length(), {"cA": 0.4, "cl": lambda roll_rate: -0.1 * roll_rate}
    )
    surface.prints.all()
    out = capsys.readouterr().out
    assert "cA = 0.4000" in out and "cl = " in out and "cN = 0 (zero)" in out


def _converted_surfaces():
    area, length = _area_length()
    return {
        "wind": GenericSurface(
            area,
            length,
            {"cL": lambda alpha, mach: 2 * alpha, "cD": 0.4, "cQ": lambda beta: -beta},
            force_convention="wind",
        ),
        "alpha_total": GenericSurface(
            area,
            length,
            {"cN": lambda alpha_total: 2 * np.sin(alpha_total)},
        ),
    }


@pytest.mark.parametrize("kind", ["wind", "alpha_total"])
def test_converted_coefficients_reuse_nothing_stale(kind):
    """The coefficients of one state are computed together and kept for the
    next coefficient; switching between states must give each its own values."""
    surface = _converted_surfaces()[kind]
    first = (0.1, 0.05, 0.3, 0, 0.02, -0.01, 0)
    second = (-0.2, 0.1, 0.6, 0, -0.01, 0.03, 0)
    expected = {
        state: [getattr(surface, name)(*state) for name in ("cN", "cY", "cm", "cn")]
        for state in (first, second)
    }
    for state in (first, second, first, second):
        for name, value in zip(("cN", "cY", "cm", "cn"), expected[state]):
            assert getattr(surface, name)(*state) == value


@pytest.mark.parametrize("kind", ["wind", "alpha_total"])
def test_converted_coefficients_accept_arrays(kind):
    surface = _converted_surfaces()[kind]
    alphas, betas = np.array([0.1, -0.2, 0.3]), np.array([0.0, 0.05, -0.1])
    for name in ("cN", "cY"):
        coefficient = getattr(surface, name)
        values = coefficient(alphas, betas, 0.3, 0, 0, 0, 0)
        assert values == pytest.approx(
            [coefficient(a, b, 0.3, 0, 0, 0, 0) for a, b in zip(alphas, betas)]
        )


def test_roll_angle_of_the_wind_is_zero_flying_exactly_tail_first():
    from rocketpy.rocket.aero_surface._helpers import total_angle_and_roll

    assert total_angle_and_roll(np.pi, np.pi) == pytest.approx((np.pi, 0.0))


def test_rate_derivatives_add_damping_to_a_table():
    """A rate derivative given next to a coefficient is multiplied by its
    reduced rate and added to it: the same result as a linear surface with the
    same slopes, also after saving and loading."""
    table = [
        [alpha, mach, -3 * alpha * (1 + 0.1 * mach)]
        for alpha in np.linspace(-0.3, 0.3, 7)
        for mach in (0.0, 1.0)
    ]
    surface = GenericSurface(
        1.0,
        1.0,
        {
            "cm": (table, ["alpha", "mach"]),
            "cm_q": -800,
            "cN": lambda alpha: 2 * alpha,
            "cN_q": [[0, 40], [1, 50]],  # an unnamed curve is against Mach
            "cl_p": -9,
        },
    )
    linear = LinearGenericSurface(
        1.0,
        1.0,
        {"cm_alpha": -3.15, "cm_q": -800, "cN_alpha": 2, "cN_q": 45, "cl_p": -9},
    )
    state = (0.1, 0.0, 0.5, 0, 0.01, 0.0, 0.002)
    for name in ("cN", "cm", "cl"):
        assert getattr(surface, name)(*state) == pytest.approx(
            getattr(linear, name)(*state), abs=1e-12
        )
    # Without rotation the table is read as it is, and so is its slope
    assert surface.cm(0.1, 0, 0.5, 0, 0, 0, 0) == pytest.approx(-0.315)
    assert surface.cm_alpha(0, 0, 0.5, 0, 0, 0, 0) == pytest.approx(-3.15)
    assert surface.cm.depends_on == ("alpha", "mach", "pitch_rate")

    loaded = json.loads(json.dumps(surface, cls=RocketPyEncoder), cls=RocketPyDecoder)
    assert "cm_q" in loaded.to_dict()["coefficients"]
    assert loaded.cm(*state) == surface.cm(*state)


def test_rate_derivatives_work_with_the_other_input_forms():
    """Damping can be added to wind-frame coefficients and to data against the
    total angle of attack, where each plane takes its own derivative."""
    wind = GenericSurface(
        1.0, 1.0, {"cL": lambda alpha: 2 * alpha, "cD": 0.4, "cm_q": -800}
    )
    assert wind.force_convention == "wind"
    assert wind.cm(0, 0, 0.5, 0, 0.01, 0, 0) == pytest.approx(-8.0)

    total = GenericSurface(
        1.0,
        1.0,
        {"cm": lambda alpha_total: -20 * alpha_total, "cm_q": -800, "cn_r": -800},
    )
    assert total.cm(0.1, 0, 0, 0, 0.01, 0, 0) == pytest.approx(-2.0 - 8.0, rel=1e-3)
    assert total.cn(0, 0.1, 0, 0, 0, 0.01, 0) == pytest.approx(2.0 - 8.0, rel=1e-3)


def test_rate_derivative_names_are_checked():
    """Only the rate derivatives of the body-frame coefficients are taken, and a
    coefficient gets its rate dependence from one place."""
    with pytest.raises(ValueError, match="cm already depends on pitch_rate"):
        GenericSurface(
            1.0,
            1.0,
            {"cm": lambda alpha, pitch_rate: -3 * alpha - 5 * pitch_rate, "cm_q": -8},
        )
    for name in ("cm_alpha", "cN_0", "cL_q"):
        with pytest.raises(ValueError, match="Invalid coefficient name"):
            GenericSurface(1.0, 1.0, {name: 1.0})
