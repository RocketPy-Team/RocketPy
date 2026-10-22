"""DATCOM-only ogive tails preserve the original ordinary Tail behavior."""

import json

import numpy as np
import pytest

from rocketpy import Tail
from rocketpy._encoders import RocketPyDecoder, RocketPyEncoder
from rocketpy.rocket.aero_surface.datcom_tail import DatcomTail


def test_ogive_tail_requires_datcom(calisto):
    """An ordinary rocket rejects ogive tails before modifying its assembly."""
    # Arrange
    count = len(calisto.aerodynamic_surfaces)
    # Act / Assert
    with pytest.raises(ValueError, match="DATCOM fin set"):
        calisto.add_tail(0.06, 0.04, 0.2, position=-1, kind="ogive")
    assert len(calisto.aerodynamic_surfaces) == count
    with pytest.raises(ValueError, match="DATCOM fin set"):
        calisto.add_surfaces(DatcomTail(0.06, 0.04, 0.2, 0.06), -1)
    with pytest.raises(ValueError, match="Tail kind"):
        calisto.add_tail(0.06, 0.04, 0.2, position=-1, kind="unknown")


def test_datcom_ogive_geometry_and_serialization(datcom_rocket):
    """Ogive shape survives serialization and has no component aero model."""
    # Arrange / Act
    tail = datcom_rocket.add_tail(0.0635, 0.045, 0.1, position=-1.2, kind="ogive")
    restored = json.loads(json.dumps(tail, cls=RocketPyEncoder), cls=RocketPyDecoder)
    # Assert
    assert isinstance(tail, DatcomTail)
    assert restored.kind == tail.kind == "ogive"
    assert tail.radius_at([0, 0.1]) == pytest.approx([0.0635, 0.045])
    assert tail.radius_at(0.05) > (0.0635 + 0.045) / 2
    np.testing.assert_allclose(restored.shape_vec, tail.shape_vec)
    assert tail.clalpha(0) == 0
    with pytest.raises(ValueError, match="DATCOM"):
        tail.compute_forces_and_moments()
    with pytest.raises(AttributeError):
        tail.kind = "conical"


def test_conical_tail_preserves_original_model(calisto):
    """The ordinary helper constructs exactly the original Tail class and equations."""
    # Arrange
    top, bottom, length = 0.06, 0.04, 0.2
    r = top / bottom
    # Act
    tail = calisto.add_tail(top, bottom, length, -1, kind="conical")
    # Assert
    assert type(tail) is Tail
    assert tail.cpz == pytest.approx((length / 3) * (1 + (1 - r) / (1 - r**2)))
    assert "kind" not in tail.to_dict()


def test_datcom_ogive_rejects_invalid_dimensions():
    """A tangent ogive cannot overhang its upstream cylinder joint."""
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="radius change"):
        DatcomTail(0.09, 0.01, 0.02, 0.09)


def test_removing_datcom_fins_rejects_ogive_before_flight(datcom_rocket):
    """An assembled ogive cannot silently fall back to ordinary aerodynamics."""
    # Arrange
    datcom_rocket.add_tail(0.0635, 0.045, 0.1, -1.2, kind="ogive")
    for fin in list(datcom_rocket.fins):
        datcom_rocket.aerodynamic_surfaces.remove(fin)
    # Act / Assert
    with pytest.raises(ValueError, match="DATCOM fin set"):
        datcom_rocket.prepare_datcom()
