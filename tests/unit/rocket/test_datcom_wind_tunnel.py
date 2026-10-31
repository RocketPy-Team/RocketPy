"""Wind-tunnel coefficient fits, units, frames and failure atomicity."""

import json
import re
from unittest.mock import patch

import numpy as np
import pytest

from rocketpy.mathutils.vector_matrix import Vector
from rocketpy.rocket.datcom import DatcomSurface, datcom_decks
from rocketpy.rocket.datcom_calibration import _measurements, _predictions


def _linear_native(factory):
    """A deterministic native surrogate with independently identifiable inputs."""

    def run(path, deck, timeout):
        rough = re.search(r"RHR\s*=\s*([-+\d.eE]+)", deck)
        height = float(rough[1]) * 3e-6 * 0.0254 if rough else 0
        radius = float(re.search(r"LER\s*=\s*([-+\d.eE]+)", deck)[1])
        return factory(
            {"CA": 0.2 + 10000 * height, "CN": 0.3 + 500 * radius, "CMQ": -0.2}
        )

    return run


@pytest.mark.parametrize("source", ["mapping", "csv"])
def test_wind_tunnel_drag_only_fit(
    datcom_rocket, datcom_table_factory, tmp_path, source
):
    """Mach-alpha-CD alone recovers a known input and applies it explicitly."""
    # Arrange
    data = {"mach": [0.2, 0.6], "alpha": [0, 0], "CD": [0.7, 0.7]}
    if source == "csv":
        path = tmp_path / "tunnel.csv"
        path.write_text("mach,alpha,CD\n0.2,0,0.7\n0.6,0,0.7\n")
        data = path
    # Act
    with patch(
        "rocketpy.rocket.datcom.run_datcom",
        side_effect=_linear_native(datcom_table_factory),
    ):
        report = datcom_rocket.fit_datcom_calibration(data)
    # Assert
    assert report["calibration"]["roughness_height"] == pytest.approx(50e-6, abs=1e-8)
    assert report["weighted_rmse_fitted"] < 1e-5
    assert report["weighted_rmse_initial"] == pytest.approx(0.5)
    assert report["reference_area"] == datcom_rocket.area
    assert datcom_rocket.datcom_calibration == report["calibration"]
    assert datcom_rocket._datcom_model is None
    json.dumps(report, allow_nan=False)


def test_wind_tunnel_two_parameter_fit(datcom_rocket, datcom_table_factory):
    """Independent axial/normal measurements constrain roughness and radius."""
    # Arrange
    data = {"mach": [0.2, 0.6], "CA": [0.7, 0.7], "CN": [0.55, 0.55]}
    # Act
    with patch(
        "rocketpy.rocket.datcom.run_datcom",
        side_effect=_linear_native(datcom_table_factory),
    ):
        report = datcom_rocket.fit_datcom_calibration(
            data,
            parameters=("roughness_height", "leading_edge_radius"),
            bounds={"roughness_height": (0, 100e-6), "leading_edge_radius": (0, 0.001)},
            max_evaluations=150,
        )
    # Assert
    assert report["calibration"]["roughness_height"] == pytest.approx(50e-6, abs=1e-8)
    assert report["calibration"]["leading_edge_radius"] == pytest.approx(
        0.0005, abs=1e-7
    )
    assert report["weighted_rmse_fitted"] < 1e-5


@pytest.mark.parametrize("orientation", [1, -1])
def test_wind_tunnel_force_moment_references(
    datcom_rocket, datcom_table_factory, orientation
):
    """Moment transfer matches an independent force/cross-product calculation."""
    # Arrange
    rocket = datcom_rocket
    rocket._csys = orientation
    table = datcom_table_factory(
        {"CA": 0.4, "CN": 0.2, "CY": 0.1, "CM": 0.03, "CLN": 0.04, "CLL": 0.05}
    )
    model = DatcomSurface(rocket, {False: table}, {False: 0.4})
    origin = -0.2
    diameter = 2 * rocket.radius
    area, length, lateral = 2 * rocket.area, 3 * diameter, 4 * diameter
    columns, measured = _measurements(
        {
            "mach": [0.2],
            "alpha": [1],
            "CD": [0],
            "CL": [0],
            "CA": [0],
            "CN": [0],
            "CY": [0],
            "CM": [0],
            "CLN": [0],
            "CLL": [0],
        },
        False,
    )
    force = Vector([0.1, -0.2, -0.4]) * rocket.area
    moment = Vector([0.03, 0.04, 0.05]) * (rocket.area * diameter)
    moment += Vector([0, 0, (0.4 - origin) * orientation]) ^ force
    flow = Vector([0, np.sin(np.radians(1)), np.cos(np.radians(1))])
    # Act
    values = _predictions(
        model, rocket, columns, measured, area, length, lateral, origin, "wind"
    )
    # Assert
    assert values["CD"][0] == pytest.approx(
        -sum(force[i] * flow[i] for i in range(3)) / area
    )
    assert values["CA"][0] == pytest.approx(0.2)
    assert values["CM"][0] == pytest.approx(moment[0] / (area * length))
    assert values["CLN"][0] == pytest.approx(moment[1] / (area * lateral))
    assert values["CLL"][0] == pytest.approx(moment[2] / (area * lateral))


def test_wind_tunnel_reference_area_and_missing_channels(
    datcom_rocket, datcom_table_factory
):
    """Area scaling and blank partial channels are honored in weighted residuals."""
    # Arrange
    data = {"mach": [0.2, 0.6], "CA": [0.35, np.nan], "CN": [np.nan, 0.15]}
    # Act
    with patch(
        "rocketpy.rocket.datcom.run_datcom",
        side_effect=_linear_native(datcom_table_factory),
    ):
        report = datcom_rocket.fit_datcom_calibration(
            data, reference_area=2 * datcom_rocket.area, weights={"CA": 2}
        )
    # Assert
    assert report["calibration"]["roughness_height"] == pytest.approx(50e-6, abs=1e-8)
    assert len(report["residuals"]) == 2
    assert report["rmse"]["CN"]["fitted"] == pytest.approx(0)


@pytest.mark.parametrize(
    "data,extra,message",
    [
        ({"mach": [0.2]}, {}, "at least one coefficient"),
        ({"mach": [0.2], "CD": [np.inf]}, {}, "infinite"),
        ({"mach": [0.2], "CD": [np.nan]}, {}, "contain a measurement"),
        ({"mach": [0.2], "alpha": [0, 1], "CD": [0.4]}, {}, "equal lengths"),
        ({"mach": [0.2], "CM": [0]}, {}, "reference_length"),
        ({"mach": [0.2], "CM": [0]}, {"reference_length": 1}, "moment_reference"),
        (
            {"mach": [0.2], "CD": [0.4]},
            {"parameters": ["leading_edge_radius"]},
            "bounds",
        ),
        ({"mach": [0.2], "CD": [0.4]}, {"weights": {"CD": 0}}, "weight"),
        ({"mach": [0.2], "CD": [0.4]}, {"drag_axis": "body"}, "drag_axis"),
        ({"mach": [0.2], "CD": [0.4], "power_on": [2]}, {}, "power_on"),
        ({"mach": [0.2], "CD": [0.4], "reynolds_per_meter": [-1]}, {}, "Reynolds"),
    ],
)
def test_wind_tunnel_invalid_data_is_atomic(datcom_rocket, data, extra, message):
    """Bad measurements cannot change calibration or start native execution."""
    # Arrange
    datcom_rocket.add_datcom_calibration(roughness_height=10e-6)
    before = datcom_rocket.datcom_calibration
    # Act / Assert
    with patch("rocketpy.rocket.datcom.run_datcom") as run:
        with pytest.raises(ValueError, match=message):
            datcom_rocket.fit_datcom_calibration(data, **extra)
    run.assert_not_called()
    assert datcom_rocket.datcom_calibration == before


@pytest.mark.parametrize("failure", ["native", "envelope", "budget", "beta"])
def test_wind_tunnel_failed_fit_preserves_prepared_model(
    datcom_rocket, datcom_table_factory, failure
):
    """Execution failures and unsupported measurements preserve the live model."""
    # Arrange
    with patch(
        "rocketpy.rocket.datcom.run_datcom", return_value=datcom_table_factory()
    ):
        original = datcom_rocket.prepare_datcom()
    settings = datcom_rocket.datcom_calibration
    data = {"mach": [0.9 if failure == "envelope" else 0.2], "CD": [0.7]}
    if failure == "beta":
        data["beta"] = [1]
    extra = {"max_evaluations": 3} if failure == "budget" else {}
    native = (
        RuntimeError("native failed")
        if failure == "native"
        else _linear_native(datcom_table_factory)
    )
    # Act / Assert
    with patch("rocketpy.rocket.datcom.run_datcom", side_effect=native):
        with pytest.raises((RuntimeError, ValueError)):
            datcom_rocket.fit_datcom_calibration(data, **extra)
    assert datcom_rocket._datcom_model is original
    assert datcom_rocket.datcom_calibration == settings


def test_wind_tunnel_reynolds_is_probe_only(datcom_rocket, datcom_table_factory):
    """Tunnel REN replaces ALT only during probes and never changes Flight options."""
    # Arrange
    native = _linear_native(datcom_table_factory)
    original = dict(datcom_rocket.fins[0].datcom_options)
    decks = []

    def run(path, deck, timeout):
        decks.append(deck)
        return native(path, deck, timeout)

    # Act
    with patch("rocketpy.rocket.datcom.run_datcom", side_effect=run):
        report = datcom_rocket.fit_datcom_calibration(
            {"mach": [0.2, 0.6], "CD": [0.7, 0.7], "reynolds_per_meter": [1e6, 2e6]}
        )
    # Assert
    assert all(
        "REN = 1000000.,2000000.," in deck and "ALT =" not in deck for deck in decks
    )
    assert datcom_rocket.fins[0].datcom_options == original
    assert "ALT =" in datcom_decks(datcom_rocket)[0][False]
    assert report["reynolds_source"] == "measurements"


def test_wind_tunnel_scale_model_references(datcom_rocket, datcom_table_factory):
    """A half-size tunnel model preserves coefficients and equivalent geometry."""
    # Arrange
    scale = 0.5
    data = {"mach": [0.2, 0.6], "CD": [0.7, 0.7], "reynolds_per_meter": [2e6, 4e6]}
    native = _linear_native(datcom_table_factory)
    decks = []

    def run(path, deck, timeout):
        decks.append(deck)
        return native(path, deck, timeout)

    # Act
    with patch("rocketpy.rocket.datcom.run_datcom", side_effect=run):
        report = datcom_rocket.fit_datcom_calibration(
            data, reference_area=datcom_rocket.area * scale**2, model_scale=scale
        )
    # Assert
    assert report["calibration"]["roughness_height"] == pytest.approx(50e-6, abs=1e-8)
    assert report["reference_area"] == datcom_rocket.area
    assert report["model_scale"] == scale
    assert all("REN = 1000000.,2000000.," in deck for deck in decks)


@pytest.mark.parametrize("axis", ["wind", "stability"])
def test_wind_tunnel_drag_axes_with_sideslip(datcom_rocket, datcom_table_factory, axis):
    """Wind-axis CD includes lateral force; stability drag is the pitch projection."""
    # Arrange
    table = datcom_table_factory({"CA": 0.4, "CN": 0.2, "CY": 0.1})
    table.beta = np.array([-5.0, 0.0, 5.0])
    table.sideslip = {angle: table for angle in table.beta}
    model = DatcomSurface(datcom_rocket, {False: table}, {False: 0})
    columns, measured = _measurements(
        {"mach": [0.2], "alpha": [1], "beta": [3], "CD": [0]}, False
    )
    alpha, beta = np.radians([1, 3])
    force = Vector([0.1, -0.2, -0.4])
    flow = Vector([np.tan(beta), np.tan(alpha), 1])
    flow = flow / abs(flow)
    expected = (
        -sum(force[i] * flow[i] for i in range(3))
        if axis == "wind"
        else 0.4 * np.cos(alpha) + 0.2 * np.sin(alpha)
    )
    # Act
    result = _predictions(
        model,
        datcom_rocket,
        columns,
        measured,
        datcom_rocket.area,
        None,
        None,
        None,
        axis,
    )
    # Assert
    assert result["CD"][0] == pytest.approx(expected)


@pytest.mark.parametrize("ren", [0, -1, np.nan, [1], [1, 2, 3], {"power_on": 1}])
def test_datcom_reynolds_schedule_validation(datcom_rocket, ren):
    """Invalid native Reynolds conditions cannot silently select an atmosphere."""
    # Arrange
    datcom_rocket.fins[0].datcom_options["reynolds"] = ren
    # Act / Assert
    with pytest.raises(ValueError, match="reynolds"):
        datcom_decks(datcom_rocket)
