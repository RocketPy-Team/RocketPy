"""Explicit native DATCOM calibration and prepared-model lifecycle."""

import json
import re
from unittest.mock import patch

import numpy as np
import pytest

from rocketpy._encoders import RocketPyDecoder, RocketPyEncoder
from rocketpy.rocket.datcom import datcom_decks


def test_datcom_calibration_native_units_and_geometry(datcom_rocket):
    """Sand roughness uses dimensionless RHR while LER remains in metres."""
    # Arrange
    rocket = datcom_rocket
    original = rocket.fins[0].leading_edge_radius
    # Act
    rocket.add_datcom_calibration(roughness_height=30.48e-6, leading_edge_radius=0.0005)
    decks, _, _, _ = datcom_decks(rocket)
    # Assert
    for deck in decks.values():
        assert float(re.search(r"RHR\s*=\s*([-+\d.eE]+)", deck)[1]) == pytest.approx(
            400
        )
        assert "LER = 0.0005,0.0005," in deck
        assert "ROUGH =" not in deck
    assert rocket.fins[0].leading_edge_radius == original


@pytest.mark.parametrize("field", ["roughness_height", "leading_edge_radius"])
@pytest.mark.parametrize("value", [-1, np.nan, np.inf, True, "invalid", [0.001]])
def test_datcom_calibration_rejects_invalid_values_atomically(
    datcom_rocket, field, value
):
    """Invalid replacement preserves the previous calibration and native deck."""
    # Arrange
    datcom_rocket.add_datcom_calibration(roughness_height=10e-6)
    before = datcom_decks(datcom_rocket)[0]
    # Act / Assert
    with pytest.raises(ValueError, match=field):
        datcom_rocket.add_datcom_calibration(**{field: value})
    assert datcom_decks(datcom_rocket)[0] == before


def test_datcom_calibration_requires_datcom_fins(calisto_motorless):
    """Calibration cannot silently affect an ordinary RocketPy rocket."""
    # Arrange
    rocket = calisto_motorless
    # Act / Assert
    with pytest.raises(ValueError, match="DATCOM fin set"):
        rocket.add_datcom_calibration(roughness_height=10e-6)


def test_datcom_calibration_cache_and_reset(datcom_rocket, datcom_table_factory):
    """Changed inputs rebuild both cases; identical inputs reuse the model."""
    # Arrange
    rocket = datcom_rocket
    original = datcom_decks(rocket)[0]
    # Act
    with patch(
        "rocketpy.rocket.datcom.run_datcom", return_value=datcom_table_factory()
    ) as run:
        first = rocket.prepare_datcom()
        rocket.add_datcom_calibration(roughness_height=20e-6)
        assert rocket._datcom_model is None
        calibrated = rocket.prepare_datcom()
        rocket.add_datcom_calibration(roughness_height=20e-6)
        unchanged = rocket.prepare_datcom()
        rocket.add_datcom_calibration()
        reset = rocket.prepare_datcom()
    # Assert
    assert first is not calibrated
    assert unchanged is calibrated
    assert reset is not calibrated
    assert run.call_count == 6
    assert datcom_decks(rocket)[0] == original


def test_datcom_calibration_serialization_and_copy(datcom_rocket):
    """Saved rockets retain calibration; editing a returned dict cannot change it."""
    # Arrange
    datcom_rocket.add_datcom_calibration(
        roughness_height=20e-6, leading_edge_radius=0.0005
    )
    # Act
    returned = datcom_rocket.datcom_calibration
    returned["roughness_height"] = 1
    restored = json.loads(
        json.dumps(datcom_rocket, cls=RocketPyEncoder), cls=RocketPyDecoder
    )
    # Assert
    assert restored.datcom_calibration == datcom_rocket.datcom_calibration
    assert restored.datcom_calibration["roughness_height"] == pytest.approx(20e-6)
    assert restored._datcom_model is None
    assert datcom_decks(restored)[0] == datcom_decks(datcom_rocket)[0]


def test_datcom_calibration_naca_radius_is_not_ignored(datcom_rocket):
    """A NACA airfoil defines its own leading edge and rejects an override."""
    # Arrange
    datcom_rocket.fins[0].section_type = "NACA"
    # Act / Assert
    with pytest.raises(ValueError, match="NACA"):
        datcom_rocket.add_datcom_calibration(leading_edge_radius=0.001)
