"""Tests for optional Folium flight trajectory maps."""

from unittest.mock import MagicMock, patch

import pytest

from rocketpy.plots.flight_plots import _FlightPlots


def mocked_folium():
    """Build a MagicMock standing in for the folium module."""
    folium = MagicMock()
    folium.Map.return_value = MagicMock()
    folium.PolyLine.return_value = MagicMock()
    folium.Marker.return_value = MagicMock()
    folium.TileLayer.return_value = MagicMock()
    folium.Circle.return_value = MagicMock()
    folium.FeatureGroup.return_value = MagicMock()
    folium.LayerControl.return_value = MagicMock()
    return folium


def patch_folium(folium):
    """Patch the optional import so that ``folium`` is the injected mock."""
    return patch(
        "rocketpy.plots.flight_plots.import_optional_dependency",
        return_value=folium,
    )


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
    """Map construction should add a path, tiles and the flight markers."""
    folium = mocked_folium()
    mock_map = folium.Map.return_value

    with patch_folium(folium):
        result = flight_calisto_robust.plots.trajectory_on_map()

    assert result is mock_map
    folium.Map.assert_called_once()
    folium.PolyLine.assert_called_once()
    folium.PolyLine.return_value.add_to.assert_called_once_with(mock_map)
    # Launch, apogee and landing markers.
    assert folium.Marker.call_count == 3
    assert folium.Marker.return_value.add_to.call_count == 3
    # OpenStreetMap and Esri satellite backgrounds, plus a control to swap them.
    assert folium.TileLayer.call_count == 2
    folium.LayerControl.assert_called_once()
    # No safety circles and no title unless explicitly requested.
    folium.Circle.assert_not_called()
    mock_map.get_root.assert_not_called()
    mock_map.save.assert_not_called()


def test_trajectory_on_map_marks_apogee_between_launch_and_landing(
    flight_calisto_robust,
):
    """The apogee marker should sit between the launch and landing markers."""
    folium = mocked_folium()

    with patch_folium(folium):
        flight_calisto_robust.plots.trajectory_on_map()

    labels = [call.kwargs["tooltip"] for call in folium.Marker.call_args_list]
    assert labels[0] == "Launch"
    assert labels[1].startswith("Apogee")
    assert labels[-1] == "Landing"


def test_trajectory_on_map_skips_apogee_when_not_detected(flight_calisto_robust):
    """A flight without a detected apogee should only get two markers."""
    folium = mocked_folium()

    with patch_folium(folium):
        with patch.object(flight_calisto_robust, "apogee_time", 0):
            flight_calisto_robust.plots.trajectory_on_map()

    labels = [call.kwargs["tooltip"] for call in folium.Marker.call_args_list]
    assert labels == ["Launch", "Landing"]


def test_trajectory_on_map_draws_safety_radii(flight_calisto_robust):
    """safety_radii= should draw one circle per radius on the launch site."""
    folium = mocked_folium()

    with patch_folium(folium):
        flight_calisto_robust.plots.trajectory_on_map(safety_radii=[2500, 5000])

    assert folium.Circle.call_count == 2
    radii = [call.kwargs["radius"] for call in folium.Circle.call_args_list]
    assert radii == [2500.0, 5000.0]
    folium.FeatureGroup.assert_called_once_with(name="Safety radii")


def test_trajectory_on_map_frames_safety_radii(flight_calisto_robust):
    """The initial viewport should widen to contain the requested circles."""
    folium = mocked_folium()
    mock_map = folium.Map.return_value

    with patch_folium(folium):
        flight_calisto_robust.plots.trajectory_on_map()
    (track_bounds,) = mock_map.fit_bounds.call_args.args

    folium = mocked_folium()
    mock_map = folium.Map.return_value
    with patch_folium(folium):
        flight_calisto_robust.plots.trajectory_on_map(safety_radii=[5000])
    (widened,) = mock_map.fit_bounds.call_args.args

    (track_sw, track_ne), (wide_sw, wide_ne) = track_bounds, widened
    assert wide_sw[0] < track_sw[0] and wide_sw[1] < track_sw[1]
    assert wide_ne[0] > track_ne[0] and wide_ne[1] > track_ne[1]
    # 5 km is roughly 0.045 degrees of latitude.
    assert 0.08 < (wide_ne[0] - wide_sw[0]) < 0.12


def test_trajectory_on_map_escapes_title(flight_calisto_robust):
    """A title should be injected as HTML with its markup escaped."""
    folium = mocked_folium()

    with patch_folium(folium):
        flight_calisto_robust.plots.trajectory_on_map(title="<script>alert(1)</script>")

    folium.Element.assert_called_once()
    title_html = folium.Element.call_args.args[0]
    assert "<script>" not in title_html
    assert "&lt;script&gt;" in title_html


def test_trajectory_on_map_time_step_resamples_path(flight_calisto_robust):
    """time_step= should replace the raw integration steps by a uniform grid."""
    folium = mocked_folium()

    with patch_folium(folium):
        flight_calisto_robust.plots.trajectory_on_map(time_step=1.0)

    path = folium.PolyLine.call_args.kwargs["locations"]
    expected = int((flight_calisto_robust.t_final - flight_calisto_robust.t_initial))
    # One sample per second, give or take the inclusive right edge.
    assert expected <= len(path) <= expected + 2
    assert len(path) < len(flight_calisto_robust.latitude[:, 1])


def test_trajectory_on_map_saves_html_with_mocked_folium(
    flight_calisto_robust, tmp_path
):
    """filename= should call Map.save with the requested path."""
    folium = mocked_folium()
    mock_map = folium.Map.return_value
    out = tmp_path / "trajectory.html"

    with patch_folium(folium):
        result = flight_calisto_robust.plots.trajectory_on_map(filename=str(out))

    assert result is mock_map
    mock_map.save.assert_called_once_with(str(out))


def test_trajectory_on_map_creates_html_file(flight_calisto_robust, tmp_path):
    """With folium installed, save an HTML map and return a Map instance."""
    folium = pytest.importorskip("folium")

    out = tmp_path / "trajectory.html"
    result = flight_calisto_robust.plots.trajectory_on_map(
        filename=str(out), safety_radii=[2500], title="Calisto"
    )

    assert isinstance(result, folium.Map)
    assert isinstance(flight_calisto_robust.plots, _FlightPlots)
    assert out.is_file()
    assert out.stat().st_size > 0
    html = out.read_text(encoding="utf-8")
    assert "leaflet" in html.lower()
    assert "Launch" in html
    assert "Apogee" in html
    assert "Landing" in html
    assert "Calisto" in html
    assert "arcgisonline" in html
