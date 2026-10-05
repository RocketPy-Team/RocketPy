"""Saving and loading aerodynamic surfaces through the RocketPy encoder."""

import json
import math

import numpy as np
import pytest

from rocketpy import (
    EllipticalFins,
    GenericSurface,
    NoseCone,
    TrapezoidalFin,
    TrapezoidalFins,
)
from rocketpy._encoders import RocketPyDecoder, RocketPyEncoder

RADIUS = 0.0635
AIRFOIL = ("data/airfoils/NACA0012-radians.txt", "radians")
STATE = (0.05, 0.02, 0.7, 1e6, 0.0, 0.0, 0.01)


def _round_trip(obj, **options):
    text = json.dumps(obj, cls=RocketPyEncoder, **options)
    return json.loads(text, cls=RocketPyDecoder)


@pytest.mark.parametrize("options", [{}, {"include_outputs": True, "discretize": True}])
def test_fin_set_with_airfoil_table_keeps_its_lift(options):
    """A tabulated airfoil is saved as it is: resampling it on save changed the
    lift slope of the loaded fins (by 29% for this table)."""
    fins = TrapezoidalFins(
        n=3,
        root_chord=0.12,
        tip_chord=0.05,
        span=0.1,
        rocket_radius=RADIUS,
        cant_angle=2,
        airfoil=AIRFOIL,
    )
    loaded = _round_trip(fins, **options)
    assert loaded.clalpha_single_fin(0.3) == pytest.approx(
        fins.clalpha_single_fin(0.3), rel=1e-10
    )
    for name in ("cN", "cl"):
        assert getattr(loaded, name)(*STATE) == pytest.approx(
            getattr(fins, name)(*STATE), rel=1e-10
        )


def test_single_fin_saves_its_airfoil_data_not_the_file_path():
    fin = TrapezoidalFin(
        angular_position=30,
        root_chord=0.12,
        tip_chord=0.05,
        span=0.1,
        rocket_radius=RADIUS,
        airfoil=AIRFOIL,
    )
    saved = json.loads(json.dumps(fin, cls=RocketPyEncoder))
    assert isinstance(saved["airfoil"][0], dict)
    loaded = _round_trip(fin)
    assert loaded.cY(*STATE) == pytest.approx(fin.cY(*STATE), rel=1e-10)


def test_saving_with_outputs_does_not_change_the_surface():
    fins = EllipticalFins(n=4, root_chord=0.12, span=0.1, rocket_radius=RADIUS)
    before = fins.clalpha(0.7)
    fins.to_dict(include_outputs=True, discretize=True)
    assert fins.clalpha(0.7) == before


def test_bluff_nose_cone_keeps_its_length():
    """The saved length is the one given, so the bluff tip does not shorten the
    nose cone again on every load or change of shape."""
    nose = NoseCone(
        length=0.5,
        kind="ogive",
        bluffness=0.3,
        base_radius=RADIUS,
        rocket_radius=RADIUS,
    )
    loaded = _round_trip(nose)
    assert loaded.length == pytest.approx(nose.length)
    assert loaded.cpz == pytest.approx(nose.cpz)
    nose.bluffness = 0.3
    assert nose.length == pytest.approx(loaded.length)


def test_sweep_angle_is_restored():
    fins = TrapezoidalFins(
        n=4,
        root_chord=0.12,
        tip_chord=0.05,
        span=0.1,
        rocket_radius=RADIUS,
        sweep_angle=20,
    )
    loaded = _round_trip(fins)
    assert loaded.sweep_angle == 20
    assert loaded.sweep_length == pytest.approx(fins.sweep_length)


def _swept_fin(fin_class, **sweep):
    placement = {"n": 4} if fin_class is TrapezoidalFins else {"angular_position": 0}
    return fin_class(
        root_chord=0.12,
        tip_chord=0.06,
        span=0.08,
        rocket_radius=RADIUS,
        **placement,
        **sweep,
    )


@pytest.mark.parametrize("fin_class", [TrapezoidalFins, TrapezoidalFin])
def test_a_sweep_length_set_after_an_angle_replaces_it(fin_class):
    """Setting the length drops the angle the fin was built with, so the fin
    does not reload with the old sweep."""
    fin = _swept_fin(fin_class, sweep_angle=30)
    fin.sweep_length = 0.01
    assert fin.sweep_angle is None
    loaded = _round_trip(fin)
    assert loaded.sweep_length == pytest.approx(0.01)
    assert loaded.cN(*STATE) == pytest.approx(fin.cN(*STATE))


@pytest.mark.parametrize("fin_class", [TrapezoidalFins, TrapezoidalFin])
def test_a_sweep_given_as_an_angle_follows_the_span(fin_class):
    fin = _swept_fin(fin_class, sweep_angle=30)
    fin.span = 0.12
    assert fin.sweep_length == pytest.approx(np.tan(np.radians(30)) * 0.12)
    loaded = _round_trip(fin)
    assert loaded.sweep_length == pytest.approx(fin.sweep_length)
    assert loaded.cpz == pytest.approx(fin.cpz)


def test_function_coefficient_saved_without_pickle_gives_a_clear_error():
    surface = GenericSurface(
        math.pi * RADIUS**2, 2 * RADIUS, {"cN": lambda alpha: 2 * alpha}
    )
    text = json.dumps(surface, cls=RocketPyEncoder, allow_pickle=False)
    with pytest.raises(ValueError, match="allow_pickle=True"):
        json.loads(text, cls=RocketPyDecoder)
