import numpy as np
import pytest

from rocketpy import Function


def _set_mass_fraction_model(rocket, source):
    """Use a 6 kg dry rocket with propellant centered 3 m behind it."""
    rocket.dry_mass = 6.0
    rocket.center_of_dry_mass_position = 0.0
    rocket.center_of_propellant_position = Function(-3.0)
    rocket.motor.propellant_mass = Function(
        source, interpolation="linear", extrapolation="constant"
    )
    rocket.total_mass = Function(
        lambda t: rocket.dry_mass + rocket.motor.propellant_mass(t)
    )
    rocket.evaluate_com_to_cdm_function()


@pytest.mark.parametrize("source_kind", ["array", "callable"])
@pytest.mark.parametrize("dx", [1e-3, 1e-4])
@pytest.mark.parametrize(
    "time, expected",
    [
        (-0.1, (1.2, 0.0, 0.0)),
        (0.0, (1.2, -0.36, -0.144)),
        (np.nextafter(0.0, 1.0), (1.2, -0.36, -0.144)),
        (0.5, (1.0, -4 / 9, -16 / 81)),
        (np.nextafter(1.0, 0.0), (0.75, -0.5625, -0.28125)),
        (1.0, (0.75, 0.0, 0.0)),
        (1.1, (0.75, 0.0, 0.0)),
    ],
)
def test_com_to_cdm_rates(calisto_motorless, source_kind, dx, time, expected):
    """Get the same physical rates from sampled and callable mass inputs."""
    rocket = calisto_motorless

    def mass_source(t):
        if t.real < 0:
            return 4.0
        if t.real > 1:
            return 2.0
        return 4.0 - 2.0 * t

    source = (
        np.array([[0.0, 4.0], [1.0, 2.0]]) if source_kind == "array" else mass_source
    )
    _set_mass_fraction_model(rocket, source)

    actual = rocket._get_com_to_cdm_rates(time, dx=dx)

    assert actual == pytest.approx(expected, rel=1e-6, abs=1e-8)
    assert actual[0] == pytest.approx(
        rocket.com_to_cdm_function(time), rel=1e-12, abs=1e-12
    )


@pytest.mark.parametrize(
    "time, dx",
    [(np.nan, 1e-4), (0.0, 0.0), (0.0, -1.0), (0.0, np.inf)],
)
def test_com_to_cdm_rates_invalid_inputs(calisto_motorless, time, dx):
    """Reject nonfinite times and invalid sample spacings."""
    with pytest.raises(ValueError):
        calisto_motorless._get_com_to_cdm_rates(time, dx=dx)


@pytest.mark.parametrize("dx", [1e-6, 1e-4, 1.0])
def test_com_to_cdm_rates_internal_knot(calisto_motorless, dx):
    """At an internal knot, use the mass slope from the segment after it."""
    rocket = calisto_motorless
    _set_mass_fraction_model(rocket, np.array([[0.0, 4.0], [0.5, 3.0], [1.0, 2.5]]))

    actual = rocket._get_com_to_cdm_rates(0.5, dx=dx)

    assert actual == pytest.approx((1.0, -2 / 9, -4 / 81), rel=1e-6)


def test_com_to_cdm_rates_delayed_ignition(calisto_motorless):
    """Find a nonzero ignition time without a manually supplied interval."""
    rocket = calisto_motorless
    rocket.motor.burn_time = (2.0, 3.0)
    rocket.motor.burn_start_time = 2.0
    rocket.motor.burn_out_time = 3.0
    _set_mass_fraction_model(rocket, np.array([[2.0, 4.0], [3.0, 2.0]]))

    assert rocket._get_com_to_cdm_rates(1.9) == pytest.approx((1.2, 0.0, 0.0))
    assert rocket._get_com_to_cdm_rates(2.0) == pytest.approx(
        (1.2, -0.36, -0.144), rel=1e-6
    )


@pytest.mark.parametrize("dx", [1e-6, 1e-4, 1e-3])
def test_com_to_cdm_rates_moving_propellant_center(calisto_motorless, dx):
    """Include the center's velocity, acceleration and product-rule term."""
    rocket = calisto_motorless
    _set_mass_fraction_model(rocket, np.array([[0.0, 4.0], [1.0, 2.0]]))
    rocket.center_of_propellant_position = Function(
        lambda t: -(3.0 + 0.1 * t + 0.25 * t**2)
    )
    rocket.evaluate_com_to_cdm_function()

    assert rocket._get_com_to_cdm_rates(0.0, dx=dx) == pytest.approx(
        (1.2, -0.32, 0.032), rel=1e-6, abs=1e-8
    )


@pytest.mark.parametrize("dx", [1e-6, 1e-4, 1e-3])
@pytest.mark.parametrize("source_kind", ["array", "callable"])
def test_com_to_cdm_rates_curved_mass(calisto_motorless, dx, source_kind):
    """Include the mass's second time derivative in the quotient rule."""
    rocket = calisto_motorless
    _set_mass_fraction_model(rocket, lambda t: 4.0 - 2.0 * t + t**2)
    if source_kind == "array":
        rocket.motor.propellant_mass = Function(
            [[0.0, 4.0], [0.5, 3.25], [1.0, 3.0]], interpolation="polynomial"
        )
        rocket.evaluate_com_to_cdm_function()

    assert rocket._get_com_to_cdm_rates(0.0, dx=dx) == pytest.approx(
        (1.2, -0.36, 0.216), rel=1e-6, abs=1e-8
    )


def test_com_to_cdm_rates_empty_propellant_center(calisto_motorless):
    """An undefined empty-tank center must not produce a NaN rocket offset."""
    rocket = calisto_motorless
    _set_mass_fraction_model(rocket, 0.0)
    rocket.center_of_propellant_position = Function(lambda t: np.nan + 0 * t)
    rocket.evaluate_com_to_cdm_function()

    assert rocket._get_com_to_cdm_rates(0.0) == (0.0, 0.0, 0.0)
    assert rocket.com_to_cdm_function(0.0) == 0.0
    np.testing.assert_array_equal(rocket.com_to_cdm_function([0.0, 1.0]), [0.0, 0.0])


@pytest.mark.parametrize("time", [0.0, 0.5, 3.9, 4.1])
def test_com_to_cdm_rates_orientation(calisto, calisto_nose_to_tail, time):
    """Mirroring user coordinates must preserve the body-frame offset rates."""
    assert calisto._get_com_to_cdm_rates(time) == pytest.approx(
        calisto_nose_to_tail._get_com_to_cdm_rates(time), rel=1e-10, abs=1e-12
    )


@pytest.mark.parametrize("boundary", ["grain_depletion", "tank_data_end"])
def test_hybrid_com_to_cdm_rates_constant_boundaries(calisto_hybrid_modded, boundary):
    """Stop CM motion when the stored mass and center become constant."""
    rocket = calisto_hybrid_modded
    if boundary == "grain_depletion":
        time = float(rocket.motor.solid.propellant_mass.x_array[-1])
    else:
        time = float(rocket.motor.liquid.propellant_mass.x_array[-1])

    _, first, second = rocket._get_com_to_cdm_rates(time)

    assert (first, second) == pytest.approx((0.0, 0.0), abs=1e-12)


def test_hybrid_com_to_cdm_rates_not_clamped_at_thrust_burnout(calisto_hybrid_modded):
    """Follow tank data that still changes after the nominal thrust cutoff."""
    rocket = calisto_hybrid_modded
    time = rocket.motor.burn_out_time
    offset = rocket.com_to_cdm_function
    h = 1e-3
    expected_first = (offset(time + h) - offset(time - h)) / (2 * h)
    expected_second = (offset(time + h) - 2 * offset(time) + offset(time - h)) / h**2

    _, first, second = rocket._get_com_to_cdm_rates(time)

    assert abs(first) > 1e-6
    assert (first, second) == pytest.approx(
        (expected_first, expected_second), rel=1e-5, abs=1e-9
    )
