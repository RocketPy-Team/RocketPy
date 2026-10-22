"""Prepare only the requested measured/model comparison values and curves.

Run reproduce_rockets.py first. Apogee-fitted constant CD is an explicitly
inferred one-parameter calibration, not measured CD or a DATCOM correction.
"""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

import model_sources as c
from rocketpy import Flight, Function
from rocketpy.rocket.datcom import run_datcom
from asteria_model import build

OUT = Path(__file__).resolve().parent
ROCKETS = OUT / "rockets"
EXE = Path(
    os.environ.get(
        "DATCOM_EXECUTABLE", c.ROOT.parent / "Asteria-Flight-Dynamics/works/dat0/datcom"
    )
)
VELOCITY = {
    "andromeda": ("t(s)", "vz(m/s)"),
    "bella_lui": ("time aprox (s)", "v (m/s)"),
    "cavour": ("ts", "velocity[m/s]"),
    "juno3": ("time", "velocity"),
    "prometheus_2022": ("time", "speed"),
}


def error(predicted, measured):
    return (
        100 * (predicted - measured) / abs(measured)
        if pd.notna(predicted) and pd.notna(measured) and measured != 0
        else np.nan
    )


def fit_cd(rocket, env, launch, target):
    if rocket.air_brakes:
        return np.nan, np.nan, "not fitted: active air-brake/controller model"
    evaluations = {}

    def residual(cd):
        rocket.power_on_drag = rocket.power_off_drag = Function(float(cd))
        rocket.power_on_drag_7d = rocket.power_off_drag_7d = Function(
            lambda alpha, beta, mach, reynolds, q, r, p: float(cd),
            inputs=["alpha", "beta", "mach", "reynolds", "q", "r", "p"],
        )
        with contextlib.redirect_stdout(io.StringIO()):
            flight = Flight(rocket, env, **launch)
        height = float(flight.apogee - env.elevation)
        evaluations[float(cd)] = height
        return height - target

    low, high = 0.05, 2.0
    if residual(low) * residual(high) > 0:
        return np.nan, np.nan, "target not bracketed by constant CD in [0.05, 2]"
    value = brentq(residual, low, high, xtol=1e-5)
    mismatch = residual(value)
    if abs(mismatch) > 0.5:
        raise AssertionError("Apogee CD fit did not reproduce target within 0.5 m")
    return (
        value,
        mismatch,
        "inferred: single constant CD, same powered/coast, ordinary lateral model",
    )


def rockets():
    results = pd.read_csv(ROCKETS / "results.csv")
    metrics = []
    for _, row in results.iterrows():
        name = row.rocket
        print("VALUES", name, flush=True)
        ref = c.flight_reference(c.ROOT / f"docs/examples/{name}_flight_sim.ipynb")
        measured_v = np.nan
        telemetry = ref["telemetry"]
        if telemetry and name in VELOCITY:
            tcol, vcol = VELOCITY[name]
            table = pd.read_csv(c.ROOT / telemetry["path"])
            if name == "bella_lui":
                table = table.iloc[:573]
            t, v = table[tcol].to_numpy(float), table[vcol].to_numpy(float)
            accepted = (
                np.interp(t, telemetry["time"], telemetry["qa_valid"].astype(float))
                > 0.99
            )
            end = telemetry["time"][
                np.argmax(
                    np.where(telemetry["qa_valid"], telemetry["altitude"], -np.inf)
                )
            ]
            accepted &= np.isfinite(t) & np.isfinite(v) & (t >= 0) & (t <= end)
            measured_v = float(np.max(v[accepted]))
            pd.DataFrame(dict(time_s=t, measured_vz_m_s=v, qa_valid=accepted)).to_csv(
                ROCKETS / f"{name}_velocity.csv", index=False
            )
        for metric, unit, measured, suffix, condition in [
            ("Apogee AGL", "m", row.actual_agl_m, "agl_m", ref["reference_kind"]),
            (
                "Peak upward vertical velocity",
                "m/s",
                measured_v,
                "peak_vertical_velocity_m_s",
                "recorded ascent velocity channel; not total airspeed",
            ),
            (
                "Initial static margin",
                "cal",
                2.875 if name == "ndrt_2020" else np.nan,
                "initial_static_margin_cal",
                "Mach 0, loaded ignition; recorded NDRT conditions unspecified; DATCOM uses lowest positive Mach",
            ),
            (
                "Coast CD at Mach 0.6",
                "1",
                np.nan,
                "coast_cd_mach06",
                "zero alpha/beta; measured CD unavailable",
            ),
        ]:
            ordinary = row.get("ordinary_" + suffix, np.nan)
            native = row.get("datcom_" + suffix, np.nan)
            metrics.append(
                dict(
                    rocket=name,
                    metric=metric,
                    unit=unit,
                    measured=measured,
                    rocketpy=ordinary,
                    datcom=native,
                    rocketpy_error_pct=error(ordinary, measured),
                    datcom_error_pct=error(native, measured),
                    condition=condition,
                    availability=row.reason if pd.notna(row.reason) else "",
                )
            )
        # Saved archived values remain identified; do not invent fresh models.
        if name in {"hedy", "valkyrie"}:
            for record in metrics:
                if record["rocket"] == name and record["metric"] == "Apogee AGL":
                    record["rocketpy"] = row.saved_first_agl_m
                    record["rocketpy_error_pct"] = error(
                        row.saved_first_agl_m, row.actual_agl_m
                    )
                    record["condition"] += (
                        "; archived RocketPy output, not fresh simulation"
                    )
        if row.status in {"paired untuned comparison", "ordinary only"}:
            with contextlib.redirect_stdout(io.StringIO()):
                rocket, env, launch, _ = c.build_model(c.ROOT / ref["source_notebook"])
            grid = np.linspace(0.01, max(1.2, row.ordinary_max_mach * 1.15), 151)
            curves = pd.DataFrame(dict(mach=grid))
            for state, label in [(True, "powered"), (False, "coast")]:
                function = (
                    rocket.power_on_drag_7d if state else rocket.power_off_drag_7d
                )
                curves[f"rocketpy_{label}_cd"] = [
                    function(0, 0, m, 1e6, 0, 0, 0) for m in grid
                ]
                path = ROCKETS / f"{name}_{label}_for005.dat"
                if path.exists():
                    parsed = run_datcom(EXE, path.read_text())
                    zero = np.flatnonzero(parsed[5] == 0)[0]
                    curves[f"datcom_{label}_cd"] = np.interp(
                        grid, parsed[4], parsed[1][:, zero]
                    )
            curves.to_csv(ROCKETS / f"{name}_cd.csv", index=False)
            cd, mismatch, status = fit_cd(rocket, env, launch, row.actual_agl_m)
            metrics.append(
                dict(
                    rocket=name,
                    metric="Apogee-fitted constant CD",
                    unit="1",
                    measured=np.nan,
                    rocketpy=np.nan,
                    datcom=np.nan,
                    inferred_cd=cd,
                    fit_apogee_residual_m=mismatch,
                    rocketpy_error_pct=np.nan,
                    datcom_error_pct=np.nan,
                    condition=status,
                    availability="",
                )
            )
        pd.DataFrame(metrics).to_csv(OUT / "rocket_metrics.csv", index=False)

    frame = pd.DataFrame(metrics)
    fits = (
        frame[frame.metric == "Apogee-fitted constant CD"]
        .set_index("rocket")
        .inferred_cd
    )
    cd_rows = frame.metric == "Coast CD at Mach 0.6"
    frame.loc[cd_rows, "inferred_cd"] = frame.loc[cd_rows, "rocket"].map(fits)
    for label in ["rocketpy", "datcom"]:
        frame.loc[cd_rows, f"{label}_vs_fitted_cd_pct"] = [
            error(predicted, inferred)
            for predicted, inferred in zip(
                frame.loc[cd_rows, label], frame.loc[cd_rows, "inferred_cd"]
            )
        ]
    frame.to_csv(OUT / "rocket_metrics.csv", index=False)


def asteria():
    with contextlib.redirect_stdout(io.StringIO()):
        ordinary, env, launch = build(False, EXE)
        native, native_env, native_launch = build(True, EXE)
        baseline = Flight(ordinary, env, **launch)
        atmosphere = native.set_datcom_atmosphere(baseline)
        flight = Flight(native, native_env, **native_launch)
        for _ in range(2):
            atmosphere = native.set_datcom_atmosphere(flight)
            flight = Flight(native, native_env, **native_launch)
    times = np.linspace(0, max(baseline.apogee_time, flight.apogee_time), 601)
    histories = pd.DataFrame(dict(time_s=times))
    for label, trajectory in [("rocketpy", baseline), ("datcom", flight)]:
        active = times <= trajectory.apogee_time
        histories[f"{label}_altitude_agl_m"] = np.where(
            active, trajectory.z(times) - env.elevation, np.nan
        )
        histories[f"{label}_vz_m_s"] = np.where(active, trajectory.vz(times), np.nan)
        histories[f"{label}_margin_cal"] = np.where(
            active, trajectory.stability_margin(times), np.nan
        )
    histories.to_csv(OUT / "asteria_histories.csv", index=False)
    grid = np.linspace(0.01, atmosphere["mach"][-1], 151)
    curves = pd.DataFrame(dict(mach=grid))
    for powered, label in [(True, "powered"), (False, "coast")]:
        function = ordinary.power_on_drag_7d if powered else ordinary.power_off_drag_7d
        curves[f"rocketpy_{label}_cd"] = [function(0, 0, m, 1e6, 0, 0, 0) for m in grid]
        curves[f"datcom_{label}_cd"] = [
            native._datcom_model.drag_coefficient(powered, 0, 0, m, 0, 0, 0, 0)
            for m in grid
        ]
    curves.to_csv(OUT / "asteria_cd.csv", index=False)
    rows = []
    for metric, unit, ordinary_value, native_value in [
        (
            "Apogee AGL",
            "m",
            baseline.apogee - env.elevation,
            flight.apogee - env.elevation,
        ),
        (
            "Peak upward vertical velocity",
            "m/s",
            np.max(baseline.vz(times[times <= baseline.apogee_time])),
            np.max(flight.vz(times[times <= flight.apogee_time])),
        ),
        (
            "Initial static margin",
            "cal",
            ordinary.static_margin(0),
            native.static_margin(0),
        ),
        (
            "Powered CD at Mach 0.6",
            "1",
            ordinary.power_on_drag_7d(0, 0, 0.6, 1e6, 0, 0, 0),
            native._datcom_model.drag_coefficient(True, 0, 0, 0.6, 0, 0, 0, 0),
        ),
        (
            "Coast CD at Mach 0.6",
            "1",
            ordinary.power_off_drag_7d(0, 0, 0.6, 1e6, 0, 0, 0),
            native._datcom_model.drag_coefficient(False, 0, 0, 0.6, 0, 0, 0, 0),
        ),
    ]:
        rows.append(
            dict(
                metric=metric,
                unit=unit,
                rocketpy=float(ordinary_value),
                datcom=float(native_value),
                datcom_vs_rocketpy_pct=error(native_value, ordinary_value),
                measured=np.nan,
            )
        )
    pd.DataFrame(rows).to_csv(OUT / "asteria_metrics.csv", index=False)
    samples = np.linspace(0, ordinary.motor.burn_out_time, 101)
    np.testing.assert_allclose(
        ordinary.total_mass(samples), native.total_mass(samples), atol=1e-10
    )
    np.testing.assert_allclose(
        ordinary.center_of_mass(samples), native.center_of_mass(samples), atol=1e-10
    )
    (OUT / "asteria_provenance.json").write_text(
        json.dumps(
            dict(
                flight_status="AST-HER has not flown",
                atmosphere=atmosphere,
                ordinary_drag_source="Asteria-Flight-Dynamics/input/drag_curve.CSV",
                ordinary_drag_sha256=hashlib.sha256(
                    (OUT / "asteria_drag_curve.csv").read_bytes()
                ).hexdigest(),
                tail_note="ordinary RocketPy conical approximation; DATCOM configured ogive",
                mass_cg_match=True,
                source_configuration="data/rockets/ast_her/configuration.json",
                scope="rail to apogee; standard RocketPy rail; no Asteria hold-down clamp or PROTUB",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--asteria-only", action="store_true")
    args = parser.parse_args()
    if not args.asteria_only:
        rockets()
    asteria()
