from unittest.mock import patch

import pytest

# TODO: these tests should be deleted after the deprecated methods are removed


def test_export_data_deprecated_emits_warning_and_delegates(flight_calisto, tmp_path):
    """Expect: calling Flight.export_data emits DeprecationWarning and delegates to FlightDataExporter.export_data."""
    out = tmp_path / "out.csv"
    with patch(
        "rocketpy.simulation.flight_data_exporter.FlightDataExporter.export_data"
    ) as spy:
        with pytest.warns(DeprecationWarning):
            flight_calisto.export_data(str(out))
    spy.assert_called_once()


def test_export_pressures_deprecated_emits_warning_and_delegates(
    flight_calisto_robust, tmp_path
):
    """Expect: calling Flight.export_pressures emits DeprecationWarning and delegates to FlightDataExporter.export_pressures."""
    out = tmp_path / "p.csv"
    with patch(
        "rocketpy.simulation.flight_data_exporter.FlightDataExporter.export_pressures"
    ) as spy:
        with pytest.warns(DeprecationWarning):
            flight_calisto_robust.export_pressures(str(out), time_step=0.1)
    spy.assert_called_once()


def test_export_kml_deprecated_emits_warning_and_delegates(flight_calisto, tmp_path):
    """Expect: calling Flight.export_kml emits DeprecationWarning and delegates to FlightDataExporter.export_kml."""
    out = tmp_path / "traj.kml"
    with patch(
        "rocketpy.simulation.flight_data_exporter.FlightDataExporter.export_kml"
    ) as spy:
        with pytest.warns(DeprecationWarning):
            flight_calisto.export_kml(str(out), time_step=0.5)
    spy.assert_called_once()
