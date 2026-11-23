import json

import numpy as np
import pytest

from rocketpy import TubeFins
from rocketpy._encoders import RocketPyDecoder, RocketPyEncoder


@pytest.fixture
def tube_fins():
    return TubeFins(
        n=6,
        length=0.1,
        inner_radius=0.045,
        outer_radius=0.05,
        rocket_radius=0.05,
    )


def _single_ring(aspect_ratio, length=0.1):
    """Tube-fin set whose tubes have the given inner diameter to length ratio.
    The tube count only sets the outer radius the geometry check asks for."""
    n = 6
    rocket_radius = 1.0
    outer_radius = rocket_radius  # touching tubes for n = 6
    return TubeFins(
        n=n,
        length=length,
        inner_radius=aspect_ratio * length / 2,
        outer_radius=outer_radius,
        rocket_radius=rocket_radius,
    )


def test_tube_fins_geometry(tube_fins):
    assert tube_fins.aspect_ratio == pytest.approx(0.9)
    assert tube_fins.reference_area == pytest.approx(np.pi * tube_fins.rocket_radius**2)
    assert tube_fins.reference_length == pytest.approx(2 * tube_fins.rocket_radius)
    assert tube_fins.tube_separation == pytest.approx(0, abs=1e-12)
    assert tube_fins.tube_center_radius == pytest.approx(0.1)


# Lift-curve slope per degree of an isolated ring wing, on the area
# inner diameter x length, read from figure 11 of NACA TN 4117
@pytest.mark.parametrize(
    ("aspect_ratio", "measured_slope_per_degree"),
    [(1 / 3, 0.018), (2 / 3, 0.034), (1.0, 0.049), (1.5, 0.069), (3.0, 0.102)],
)
def test_tube_fins_normal_force_slope_matches_ring_wing_measurements(
    aspect_ratio, measured_slope_per_degree
):
    tube_fins = _single_ring(aspect_ratio)
    planform_area = 2 * tube_fins.inner_radius * tube_fins.length

    slope_per_tube = tube_fins.clalpha(0) * tube_fins.reference_area / tube_fins.n
    slope_per_degree = np.radians(slope_per_tube / planform_area)

    assert slope_per_degree == pytest.approx(measured_slope_per_degree, rel=0.05)


def test_tube_fins_normal_force_slope_limits():
    """A long tube tends to the slender body value and a wide ring to the
    lifting line value, both on the area inner diameter x length."""
    long_tube, wide_ring = _single_ring(0.01), _single_ring(100.0, length=0.01)

    def slope(tube_fins):
        planform_area = 2 * tube_fins.inner_radius * tube_fins.length
        return tube_fins.clalpha(0) * tube_fins.reference_area / (6 * planform_area)

    assert slope(long_tube) == pytest.approx(np.pi * 0.01, rel=0.01)
    assert slope(wide_ring) == pytest.approx(np.pi**2, rel=0.03)


def test_tube_fins_normal_force_does_not_depend_on_mach(tube_fins):
    assert tube_fins.clalpha(0.8) == pytest.approx(tube_fins.clalpha(0))
    assert tube_fins.clalpha(2.0) == pytest.approx(tube_fins.clalpha(0))


def test_tube_fins_lift_is_linear_in_the_angle_of_attack(tube_fins):
    slope = tube_fins.clalpha(0)

    assert tube_fins.cl(np.radians(10), 0) == pytest.approx(slope * np.radians(10))
    assert tube_fins.cl(np.radians(30), 0) == pytest.approx(slope * np.radians(30))
    assert tube_fins.cl(np.radians(-30), 0) == pytest.approx(-slope * np.radians(30))


# Center of pressure from the leading edge, in tube lengths, read from
# figure 8 of NACA TN 4117
@pytest.mark.parametrize(
    ("aspect_ratio", "measured_position"),
    [
        (1 / 3, -0.11),
        (2 / 3, 0.14),
        (1.0, 0.20),
        (1.5, 0.25),
        (3.0, 0.35),
        (0.1, -0.11),  # below the measured range: nearest measured value
        (5.0, 0.35),  # above the measured range: nearest measured value
        (0.5, 0.015),  # halfway between the first two measured points
    ],
)
def test_tube_fins_center_of_pressure_follows_measurements(
    aspect_ratio, measured_position
):
    tube_fins = _single_ring(aspect_ratio, length=0.2)

    assert tube_fins.cp == pytest.approx((0, 0, measured_position * 0.2))


def test_tube_fins_damp_a_roll_rate(tube_fins):
    """The roll moment opposes the roll rate, grows in proportion to it and
    equals the normal force the roll produces on the tubes times their
    distance from the rocket axis."""
    rho, speed, roll_rate = 1.2, 100.0, 10.0
    stream_velocity = (0.0, 0.0, -speed)

    def roll_moment(rate):
        return tube_fins.compute_forces_and_moments(
            stream_velocity, speed, 0.3, rho, (0, 0, 0), (0, 0, rate)
        )[5]

    dynamic_pressure = 0.5 * rho * speed**2
    local_angle_of_attack = roll_rate * tube_fins.tube_center_radius / speed
    normal_force = (
        dynamic_pressure
        * tube_fins.reference_area
        * tube_fins.clalpha(0)
        * local_angle_of_attack
    )

    assert roll_moment(0.0) == 0
    assert roll_moment(roll_rate) == pytest.approx(
        -normal_force * tube_fins.tube_center_radius
    )
    assert roll_moment(-roll_rate) == pytest.approx(-roll_moment(roll_rate))
    assert roll_moment(2 * roll_rate) == pytest.approx(2 * roll_moment(roll_rate))


def test_tube_fins_roll_rate_does_not_change_the_normal_force(tube_fins):
    rho, speed = 1.2, 100.0
    stream_velocity = (speed * np.sin(0.05), 0.0, -speed * np.cos(0.05))

    still = tube_fins.compute_forces_and_moments(
        stream_velocity, speed, 0.3, rho, (0, 0, 0), (0, 0, 0)
    )
    rolling = tube_fins.compute_forces_and_moments(
        stream_velocity, speed, 0.3, rho, (0, 0, 0), (0, 0, 10.0)
    )

    assert still[0] > 0
    assert rolling[:5] == pytest.approx(still[:5])


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"n": 2}, "greater than or equal to 3"),
        ({"n": 6.0}, "must be an integer"),
        ({"length": 0}, "length.*greater than zero"),
        ({"inner_radius": 0}, "inner_radius.*greater than zero"),
        ({"inner_radius": 0.05}, "smaller than.*outer_radius"),
        ({"outer_radius": 0}, "outer_radius.*greater than zero"),
        ({"rocket_radius": 0}, "rocket_radius.*greater than zero"),
        ({"length": np.nan}, "length.*finite"),
        ({"outer_radius": 0.049}, "separated tube fins"),
        ({"outer_radius": 0.06}, "overlapping tube fins"),
    ],
)
def test_tube_fins_reject_unsupported_geometry(overrides, message):
    parameters = {
        "n": 6,
        "length": 0.1,
        "inner_radius": 0.045,
        "outer_radius": 0.05,
        "rocket_radius": 0.05,
    }
    parameters.update(overrides)

    with pytest.raises(ValueError, match=message):
        TubeFins(**parameters)


def test_tube_fins_setters_update_dependent_values(tube_fins):
    initial_clalpha = tube_fins.clalpha(0)

    tube_fins.length = 0.2
    assert tube_fins.aspect_ratio == pytest.approx(0.45)
    assert tube_fins.cpz == pytest.approx(
        0.2 * np.interp(0.45, [1 / 3, 2 / 3], [-0.11, 0.14])
    )
    assert tube_fins.clalpha(0) != pytest.approx(initial_clalpha)

    previous_outer_radius = tube_fins.outer_radius
    with pytest.raises(ValueError, match="separated tube fins"):
        tube_fins.outer_radius = 0.049
    assert tube_fins.outer_radius == previous_outer_radius


def test_tube_fins_add_to_rocket(calisto):
    initial_clalpha = calisto.total_lift_coeff_der(0)
    tube_fins = TubeFins(
        n=6,
        length=0.12,
        inner_radius=0.055,
        outer_radius=calisto.radius,
        rocket_radius=calisto.radius,
    )

    calisto.add_surfaces(tube_fins, -1.1)

    assert not hasattr(calisto, "add_tube_fins")
    assert calisto.aerodynamic_surfaces[-1].component is tube_fins
    assert calisto.aerodynamic_surfaces[-1].position.z == pytest.approx(-1.1)
    assert calisto.total_lift_coeff_der(0) == pytest.approx(
        initial_clalpha + tube_fins.clalpha(0)
    )


@pytest.mark.parametrize(
    ("include_outputs", "discretize"),
    [(False, False), (True, False), (True, True)],
)
def test_tube_fins_json_round_trip(tube_fins, include_outputs, discretize):
    encoded = json.dumps(
        tube_fins,
        cls=RocketPyEncoder,
        include_outputs=include_outputs,
        discretize=discretize,
    )
    decoded = json.loads(encoded, cls=RocketPyDecoder)

    assert isinstance(decoded, TubeFins)
    assert decoded.n == tube_fins.n
    assert decoded.length == pytest.approx(tube_fins.length)
    assert decoded.inner_radius == pytest.approx(tube_fins.inner_radius)
    assert decoded.outer_radius == pytest.approx(tube_fins.outer_radius)
    assert decoded.rocket_radius == pytest.approx(tube_fins.rocket_radius)
    assert decoded.cp == pytest.approx(tube_fins.cp)
    assert decoded.clalpha(0) == pytest.approx(tube_fins.clalpha(0))


def test_tube_fins_info_and_draw(tube_fins, capsys, monkeypatch):
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)

    assert tube_fins.info() is None
    assert "Number of tubes: 6" in capsys.readouterr().out
    assert tube_fins.draw() is None
