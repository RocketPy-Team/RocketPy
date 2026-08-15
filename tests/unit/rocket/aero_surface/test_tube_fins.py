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


def test_tube_fins_geometry_and_normal_force_slope(tube_fins):
    aspect_ratio = 2 * tube_fins.inner_radius / tube_fins.length
    modified_aspect_ratio = 2 * aspect_ratio / np.pi
    expected_clalpha = (
        tube_fins.n
        * 2
        * modified_aspect_ratio
        / (1 + modified_aspect_ratio)
        * np.pi**2
        * tube_fins.inner_radius
        * tube_fins.length
        / (np.pi * tube_fins.rocket_radius**2)
    )

    assert tube_fins.aspect_ratio == pytest.approx(aspect_ratio)
    assert tube_fins.cp == pytest.approx((0, 0, tube_fins.length / 4))
    assert tube_fins.reference_area == pytest.approx(np.pi * tube_fins.rocket_radius**2)
    assert tube_fins.reference_length == pytest.approx(2 * tube_fins.rocket_radius)
    assert tube_fins.tube_separation == pytest.approx(0, abs=1e-12)
    assert tube_fins.clalpha(0) == pytest.approx(expected_clalpha)
    assert tube_fins.clalpha(0.5) == pytest.approx(expected_clalpha)


def test_tube_fins_lift_is_capped_at_twenty_degrees(tube_fins):
    capped_lift = tube_fins.clalpha(0) * np.radians(20)

    assert tube_fins.cl(np.radians(10), 0) == pytest.approx(capped_lift / 2)
    assert tube_fins.cl(np.radians(30), 0) == pytest.approx(capped_lift)
    assert tube_fins.cl(np.radians(-30), 0) == pytest.approx(-capped_lift)


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
    assert tube_fins.cpz == pytest.approx(0.05)
    assert tube_fins.aspect_ratio == pytest.approx(0.45)
    assert tube_fins.clalpha(0) != pytest.approx(initial_clalpha)

    previous_outer_radius = tube_fins.outer_radius
    with pytest.raises(ValueError, match="separated tube fins"):
        tube_fins.outer_radius = 0.049
    assert tube_fins.outer_radius == previous_outer_radius


def test_tube_fins_add_to_rocket(calisto):
    initial_clalpha = calisto.total_lift_coeff_der(0)
    tube_fins = calisto.add_tube_fins(
        n=6,
        length=0.12,
        inner_radius=0.055,
        outer_radius=calisto.radius,
        position=-1.1,
    )

    assert tube_fins in calisto.tube_fins
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
