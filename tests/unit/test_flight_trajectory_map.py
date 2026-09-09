"""Tests for optional Folium flight trajectory maps."""

from unittest.mock import MagicMock, patch

import pytest

from rocketpy.plots.flight_plots import _FlightPlots


def test_trajectory_on_map_requires_folium(flight_calisto_robust):
    """Missing folium should raise a clear ImportError via optional import."""
    with patch(
        "rocketpy.plots.flight_plots.import_optional_dependency",
        side_effect=ImportError(
            "folium is an optional dependency and is not installed.\n"
            "\t\tUse 'pip install folium' to install it or "
            "'pip install rocketpy[all]' to install all optional dependencies."
        ),
    ):
        with pytest.raises(ImportError, match="folium"):
            flight_calisto_robust.plots.trajectory_on_map()


def test_trajectory_on_map_builds_map_with_mocked_folium(flight_calisto_robust):
    """Map construction should add a path and launch/landing markers."""
    mock_folium = MagicMock()
    mock_map = MagicMock()
    mock_folium.Map.return_value = mock_map
    mock_polyline = MagicMock()
    mock_folium.PolyLine.return_value = mock_polyline
    mock_marker = MagicMock()
    mock_folium.Marker.return_value = mock_marker

    with patch(
        "rocketpy.plots.flight_plots.import_optional_dependency",
        return_value=mock_folium,
    ):
        result = flight_calisto_robust.plots.trajectory_on_map()

    assert result is mock_map
    mock_folium.Map.assert_called_once()
    mock_folium.PolyLine.assert_called_once()
    assert mock_folium.Marker.call_count == 2
    mock_polyline.add_to.assert_called_once_with(mock_map)
    assert mock_marker.add_to.call_count == 2
    mock_map.save.assert_not_called()


def test_trajectory_on_map_saves_html_with_mocked_folium(
    flight_calisto_robust, tmp_path
):
    """filename= should call Map.save with the requested path."""
    mock_folium = MagicMock()
    mock_map = MagicMock()
    mock_folium.Map.return_value = mock_map
    mock_folium.PolyLine.return_value = MagicMock()
    mock_folium.Marker.return_value = MagicMock()
    out = tmp_path / "trajectory.html"

    with patch(
        "rocketpy.plots.flight_plots.import_optional_dependency",
        return_value=mock_folium,
    ):
        result = flight_calisto_robust.plots.trajectory_on_map(filename=str(out))

    assert result is mock_map
    mock_map.save.assert_called_once_with(str(out))


def test_trajectory_on_map_creates_html_file(flight_calisto_robust, tmp_path):
    """With folium installed, save an HTML map and return a Map instance."""
    folium = pytest.importorskip("folium")

    out = tmp_path / "trajectory.html"
    result = flight_calisto_robust.plots.trajectory_on_map(filename=str(out))

    assert isinstance(result, folium.Map)
    assert isinstance(flight_calisto_robust.plots, _FlightPlots)
    assert out.is_file()
    assert out.stat().st_size > 0
    html = out.read_text(encoding="utf-8")
    assert "leaflet" in html.lower()
    assert "Launch" in html
    assert "Landing" in html
