import json
from types import SimpleNamespace
from rocketpy.simulation import FlightDataExporter


def test_export_data_writes_csv_header(flight_calisto, tmp_path):
    """Expect: direct exporter writes a CSV with a header containing 'Time (s)'."""
    out = tmp_path / "out.csv"
    FlightDataExporter(flight_calisto).export_data(str(out))
    text = out.read_text(encoding="utf-8")
    assert "Time (s)" in text


def test_export_pressures_writes_rows(flight_calisto_robust, tmp_path):
    """Expect: direct exporter writes a pressure file with time-first CSV rows."""
    out = tmp_path / "p.csv"
    FlightDataExporter(flight_calisto_robust).export_pressures(str(out), time_step=0.2)
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) > 5
    # basic CSV shape “t, value”
    parts = lines[0].split(",")
    assert len(parts) in (2, 3)


def test_export_sensor_data_writes_json_when_sensor_data_present(
    flight_calisto, tmp_path, monkeypatch
):
    """Expect: exporter writes JSON mapping sensor.name -> data when sensor_data is present."""

    class DummySensor:
        def __init__(self, name):
            self.name = name

    s1 = DummySensor("DummySensor")
    monkeypatch.setattr(
        flight_calisto, "sensor_data", {s1: [1.0, 2.0, 3.0]}, raising=False
    )
    out = tmp_path / "sensors.json"

    FlightDataExporter(flight_calisto).export_sensor_data(str(out))

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["DummySensor"] == [1.0, 2.0, 3.0]
