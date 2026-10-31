"""Untuned ordinary/DATCOM ascent comparisons and full notebook evidence audit."""

import argparse
import contextlib
import io
import json
import faulthandler
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
import model_sources as c

from rocketpy import Flight  # noqa: E402
from rocketpy.rocket.datcom import datcom_decks  # noqa: E402

OUT = Path(__file__).resolve().parent / "rockets"
OUT.mkdir(exist_ok=True)
EXE = Path(
    os.environ.get(
        "DATCOM_EXECUTABLE", c.ROOT.parent / "Asteria-Flight-Dynamics/works/dat0/datcom"
    )
)


def finalize_result(result, record):
    """Use one shared measured-ascent overlap for both resampled histories."""
    result["notes"] = " | ".join(record["notes"])
    telemetry = record["telemetry"]
    if telemetry:
        result["telemetry_qa_max_agl_m"] = telemetry["qa_max_agl_m"]
        result["telemetry_qa_rejected_count"] = telemetry["qa_rejected_count"]
        for label in ("ordinary", "datcom"):
            if f"{label}_agl_m" in result:
                result[f"{label}_vs_telemetry_pct"] = 100 * (
                    result[f"{label}_agl_m"] / telemetry["qa_max_agl_m"] - 1
                )
        path = OUT / f"{record['rocket']}_altitude.csv"
        if path.exists():
            history = pd.read_csv(path)
            measured = pd.read_csv(OUT / f"{record['rocket']}_telemetry.csv")
            measured = measured[measured.qa_valid]
            labels = [
                label for label in ("ordinary", "datcom") if f"{label}_agl_m" in history
            ]
            upper = min(
                [float(measured.loc[measured.altitude_agl_m.idxmax(), "time_s"])]
                + [
                    float(
                        history.loc[history[f"{label}_agl_m"].notna(), "time_s"].max()
                    )
                    for label in labels
                ]
            )
            lower = max(0, float(measured.time_s.min()))
            if labels and upper > lower:
                time = np.linspace(lower, upper, 301)
                observed = np.interp(time, measured.time_s, measured.altitude_agl_m)
                for label in labels:
                    mask = history[f"{label}_agl_m"].notna()
                    predicted = np.interp(
                        time, history.time_s[mask], history[f"{label}_agl_m"][mask]
                    )
                    result[f"{label}_ascent_altitude_rmse_m"] = float(
                        np.sqrt(np.mean((predicted - observed) ** 2))
                    )
                result["rmse_start_s"], result["rmse_end_s"] = lower, upper
    if result.get("status") == "paired untuned comparison":
        result["within_sampled_envelope"] = all(
            result[key] == 0
            for key in ["outside_mach", "outside_alpha", "outside_beta"]
        )
    return result


def summarize(flight, reference, prefix):
    result = {
        f"{prefix}_agl_m": float(flight.apogee - flight.env.elevation),
        f"{prefix}_apogee_time_s": float(flight.apogee_time),
    }
    result[f"{prefix}_peak_vertical_velocity_m_s"] = float(
        np.max(flight.vz(np.linspace(0, flight.apogee_time, 2001)))
    )
    result[f"{prefix}_initial_static_margin_cal"] = float(
        flight.rocket.static_margin(0)
    )
    if prefix == "datcom":
        result[f"{prefix}_coast_cd_mach06"] = float(
            flight.rocket._datcom_model.drag_coefficient(False, 0, 0, 0.6, 0, 0, 0, 0)
        )
    else:
        result[f"{prefix}_coast_cd_mach06"] = float(
            flight.rocket.power_off_drag_7d(0, 0, 0.6, 1e6, 0, 0, 0)
        )
    time = np.linspace(0, flight.apogee_time, 501)
    result[f"{prefix}_max_mach"] = float(np.max(flight.mach_number(time)))
    telemetry = reference["telemetry"]
    if telemetry:
        mask = telemetry["qa_valid"]
        t = telemetry["time"][mask]
        height = telemetry["altitude"][mask]
        upper = min(flight.apogee_time, float(t[np.argmax(height)]))
        lower = max(0, float(t.min()))
        if upper > lower:
            sample = np.linspace(lower, upper, 301)
            result[f"{prefix}_ascent_altitude_rmse_m"] = float(
                np.sqrt(
                    np.mean(
                        (
                            flight.z(sample)
                            - flight.env.elevation
                            - np.interp(sample, t, height)
                        )
                        ** 2
                    )
                )
            )
    return result


def one(name, refinements=1):
    faulthandler.dump_traceback_later(30)
    source = c.ROOT / f"docs/examples/{name}_flight_sim.ipynb"
    ref = c.flight_reference(source)
    row = {
        key: ref[key]
        for key in [
            "rocket",
            "actual_agl_m",
            "reported_agl_m",
            "saved_first_agl_m",
            "reference_kind",
        ]
    }
    row["telemetry_max_agl_m"] = (
        ref["telemetry"]["raw_max_agl_m"] if ref["telemetry"] else None
    )
    row["notes"] = " | ".join(ref["notes"])
    row["status"] = "not run"
    row["reason"] = ""
    if name == "hedy" and not (c.ROOT / "data/rockets/tust").exists():
        row.update(
            status="archived comparison only",
            reason="Source data/rockets/tust is absent; CoolProp is not installed, and the negative-span fin root cutout requires geometry support.",
        )
    else:
        try:
            np.random.seed(0)
            print("BUILD SOURCE", name, flush=True)
            with contextlib.redirect_stdout(io.StringIO()):
                rocket, env, launch, cells = c.build_model(source)
            print("ORDINARY FLIGHT", name, flush=True)
            with contextlib.redirect_stdout(io.StringIO()):
                baseline = Flight(rocket, env, **launch)
            row.update(summarize(baseline, ref, "ordinary"))
            row["status"] = "ordinary only"
            row["source_adaptations"] = " | ".join(rocket._example_adaptations)
            times = np.linspace(0, baseline.apogee_time, 501)
            history = pd.DataFrame(
                dict(
                    time_s=times,
                    ordinary_agl_m=baseline.z(times) - env.elevation,
                    ordinary_vz_m_s=baseline.vz(times),
                    ordinary_margin_cal=baseline.stability_margin(times),
                )
            )
            history.to_csv(OUT / f"{name}_altitude.csv", index=False)
            if rocket.air_brakes:
                row["reason"] = (
                    "Source active air brakes override complete rocket drag. Combining their whole-vehicle CD with DATCOM would double-count base drag; no controlled DATCOM result claimed."
                )
            else:
                np.random.seed(0)
                print("BUILD DATCOM", name, flush=True)
                with contextlib.redirect_stdout(io.StringIO()):
                    other, other_env, other_launch, _ = c.build_model(source)
                c.h.attach_datcom(other, EXE, 0)
                atmosphere = other.set_datcom_atmosphere(baseline)
                grid = atmosphere["mach"]
                samples = np.linspace(0, rocket.motor.burn_out_time, 101)
                np.testing.assert_allclose(
                    rocket.total_mass(samples),
                    other.total_mass(samples),
                    atol=1e-10,
                    rtol=0,
                )
                np.testing.assert_allclose(
                    rocket.center_of_mass(samples),
                    other.center_of_mass(samples),
                    atol=1e-10,
                    rtol=0,
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    print("NATIVE FLIGHT", name, file=sys.stderr, flush=True)
                    native = Flight(other, other_env, **other_launch)
                iterations = [
                    {
                        "pass": 0,
                        "apogee_agl_m": float(native.apogee - env.elevation),
                        "reference": "ordinary trajectory",
                    }
                ]
                for iteration in range(refinements):
                    atmosphere = other.set_datcom_atmosphere(native)
                    grid = atmosphere["mach"]
                    with contextlib.redirect_stdout(io.StringIO()):
                        native = Flight(other, other_env, **other_launch)
                    iterations.append(
                        {
                            "pass": iteration + 1,
                            "apogee_agl_m": float(native.apogee - env.elevation),
                            "reference": "previous DATCOM trajectory",
                        }
                    )
                schedules = atmosphere["altitude_m_asl"]
                decks, _, _, _ = datcom_decks(other)
                for powered, deck in decks.items():
                    (
                        OUT / f"{name}_{'powered' if powered else 'coast'}_for005.dat"
                    ).write_text(deck)
                row["atmosphere_refinements"] = refinements
                row["atmosphere_last_apogee_change_m"] = (
                    iterations[-1]["apogee_agl_m"] - iterations[-2]["apogee_agl_m"]
                    if refinements
                    else None
                )
                row.update(summarize(native, ref, "datcom"))
                row.update(
                    c.h.envelope_diagnostics(
                        native,
                        other._datcom_model,
                        mach_grid=grid,
                        alpha_grid=atmosphere["alpha"],
                    )
                )
                row["status"] = "paired untuned comparison"
                row["reason"] = ""
                row["mass_cg_match"] = True
                time = np.linspace(
                    0, max(native.apogee_time, baseline.apogee_time), 601
                )
                ordinary = np.where(
                    time <= baseline.apogee_time,
                    baseline.z(time) - env.elevation,
                    np.nan,
                )
                datcom = np.where(
                    time <= native.apogee_time, native.z(time) - env.elevation, np.nan
                )
                pd.DataFrame(
                    dict(
                        time_s=time,
                        ordinary_agl_m=ordinary,
                        datcom_agl_m=datcom,
                        ordinary_vz_m_s=np.where(
                            time <= baseline.apogee_time, baseline.vz(time), np.nan
                        ),
                        datcom_vz_m_s=np.where(
                            time <= native.apogee_time, native.vz(time), np.nan
                        ),
                        ordinary_margin_cal=np.where(
                            time <= baseline.apogee_time,
                            baseline.stability_margin(time),
                            np.nan,
                        ),
                        datcom_margin_cal=np.where(
                            time <= native.apogee_time,
                            native.stability_margin(time),
                            np.nan,
                        ),
                    )
                ).to_csv(OUT / f"{name}_altitude.csv", index=False)
                metadata = dict(
                    construction_cells=sorted(set(cells)),
                    calibration=other.datcom_calibration,
                    source_adaptations=rocket._example_adaptations,
                    mach=grid,
                    alpha=atmosphere["alpha"],
                    beta=c.h.BETA,
                    altitude_schedules_m_asl=schedules,
                    atmosphere=atmosphere,
                    atmosphere_iterations=iterations,
                    atmosphere_method="RocketPy reference ascent rho*a*M/mu; native REN",
                    source_notebook=ref["source_notebook"],
                    source_sha256=ref["source_sha256"],
                    construction_assumption="carbon fibre and vinyl when unspecified; no numerical roughness inferred",
                    fin_assumption="2% HEX, 5% bevels, sharp edges, auto body attachment",
                    powered_base_drag="nozzle",
                    damp=True,
                )
                (OUT / f"{name}_provenance.json").write_text(
                    json.dumps(metadata, indent=2)
                )
        except Exception as error:
            row["reason"] = f"{type(error).__name__}: {error}"
            if row["status"] == "not run":
                row["status"] = "unavailable source model"
    for label in ["ordinary", "datcom"]:
        if f"{label}_agl_m" in row:
            row[f"{label}_error_m"] = row[f"{label}_agl_m"] - row["actual_agl_m"]
            row[f"{label}_error_pct"] = (
                100 * row[f"{label}_error_m"] / row["actual_agl_m"]
            )
    (OUT / f"{name}_result.json").write_text(json.dumps(row, indent=2, allow_nan=False))
    print(json.dumps(row), flush=True)
    faulthandler.cancel_dump_traceback_later()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case")
    parser.add_argument("--only", nargs="*")
    parser.add_argument(
        "--refinements",
        type=int,
        default=1,
        choices=range(6),
        help="DATCOM-trajectory atmosphere updates after the ordinary seed (default 1)",
    )
    args = parser.parse_args()
    if args.case:
        one(args.case, args.refinements)
        return
    inventory = []
    for path in sorted((c.ROOT / "docs/examples").glob("*flight_sim.ipynb")):
        record = c.flight_reference(path)
        telemetry = record.pop("telemetry")
        record["telemetry"] = (
            {
                key: value
                for key, value in telemetry.items()
                if key not in ["time", "altitude", "qa_valid"]
            }
            if telemetry
            else None
        )
        if telemetry:
            pd.DataFrame(
                dict(
                    time_s=telemetry["time"],
                    altitude_agl_m=telemetry["altitude"],
                    qa_valid=telemetry["qa_valid"],
                )
            ).to_csv(OUT / f"{record['rocket']}_telemetry.csv", index=False)
        inventory.append(record)
    (OUT / "inventory.json").write_text(json.dumps(inventory, indent=2))
    results = []
    for record in inventory:
        name = record["rocket"]
        if args.only is None or name in args.only:
            print("RUN", name, flush=True)
            try:
                with (OUT / f"{name}_execution.log").open("w") as log:
                    subprocess.run(
                        [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "--case",
                            name,
                            "--refinements",
                            str(args.refinements),
                        ],
                        cwd=c.ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=90,
                        check=True,
                    )
            except subprocess.TimeoutExpired:
                (OUT / f"{name}_result.json").write_text(
                    json.dumps(
                        dict(
                            rocket=name,
                            status="source simulation timed out",
                            reason="Source model did not complete within 90 s; inputs/equations not silently changed.",
                            actual_agl_m=record["actual_agl_m"],
                            reported_agl_m=record["reported_agl_m"],
                            saved_first_agl_m=record["saved_first_agl_m"],
                        )
                    )
                )
            except subprocess.CalledProcessError:
                print("CHECK LOG", name, flush=True)
        path = OUT / f"{name}_result.json"
        if path.exists():
            result = finalize_result(json.loads(path.read_text()), record)
            path.write_text(json.dumps(result, indent=2, allow_nan=False))
            results.append(result)
            pd.DataFrame(results).to_csv(OUT / "results.csv", index=False)
            print(
                name,
                result["status"],
                result.get("ordinary_agl_m"),
                result.get("datcom_agl_m"),
                result["reason"],
                flush=True,
            )


if __name__ == "__main__":
    main()
