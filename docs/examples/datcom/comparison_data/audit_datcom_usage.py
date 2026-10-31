"""Recheck manual contracts without adding another example notebook or fitting.

The nozzle alternatives below are sensitivity bounds/hypotheses, not measured
hardware corrections. No source input, source result or measured target changes.
"""

import contextlib
import hashlib
import io
import json
import os
import re
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import model_sources as c
from rocketpy import Flight
from rocketpy.rocket.datcom import DAMPING_KEYS, _namelist, run_datcom

OUT = Path(__file__).resolve().parent
EXE = Path(
    os.environ.get(
        "DATCOM_EXECUTABLE", c.ROOT.parent / "Asteria-Flight-Dynamics/works/dat0/datcom"
    )
)


def conditions(deck, mach):
    """Controlled conditions for sweep/grid/unit probes, not a flight fit."""
    return re.sub(
        r"\$FLTCON.*?\$",
        lambda _: _namelist(
            "FLTCON",
            [
                ("NMACH", len(mach)),
                ("MACH", mach),
                ("REN", np.asarray(mach) * 2e7),
                ("NALPHA", 3),
                ("ALPHA", [-2, 0, 2]),
                ("BETA", 0),
            ],
        ),
        deck,
        flags=re.S,
    )


def variants(deck):
    """Independent equivalent deck encodings recommended/defined by the guide."""
    dimensional = {
        "XCG",
        "LREF",
        "LNOSE",
        "DNOSE",
        "LCENTR",
        "DCENTR",
        "LAFT",
        "DAFT",
        "DEXIT",
        "XLE",
        "SSPAN",
        "CHORD",
        "LER",
    }

    def rewrite(match, *, inches):
        fields = []
        for field in re.finditer(r"(\w+)\s*=\s*(.*?)(?=\b\w+\s*=|$)", match[2], re.S):
            key, raw = field[1], field[2].strip(" ,\n")
            try:
                value = np.array([float(v) for v in raw.split(",")])
            except ValueError:
                value = raw
            fields.append((key, value))
        values = dict(fields)
        if inches:
            for key, value in fields:
                if key in dimensional:
                    values[key] = value / 0.0254
                elif key == "SREF":
                    values[key] = value / 0.0254**2
                elif key == "REN":
                    # FLTCON uses /ft in all English geometry units (Table 3).
                    values[key] = value * 0.3048
        elif match[1].startswith("FINSET"):
            te = values["XLE"] + values["CHORD"]
            values["SWEEP"] = np.degrees(
                np.arctan2(np.diff(te), np.diff(values["SSPAN"]))
            )
            values["STA"] = 1
            values["XLE"] = values["XLE"][:1]
        return _namelist(match[1], list(values.items()))

    return {
        "explicit_trailing_edge_sweep": re.sub(
            r"\$(\w+)\s*(.*?)\$",
            lambda m: rewrite(m, inches=False),
            deck,
            flags=re.S,
        ),
        "inch_geometry_feet_reynolds": re.sub(
            r"\$(\w+)\s*(.*?)\$",
            lambda m: rewrite(m, inches=True),
            deck,
            flags=re.S,
        ).replace("DIM M", "DIM IN"),
    }


def main():
    report = {
        "manual": "Missile DATCOM Users Guide, revision 5/97",
        "native": "local revision 3/99 executable",
        "executable_sha256": hashlib.sha256(EXE.read_bytes()).hexdigest(),
        "controlled_conditions": {
            "mach": [0.3, 0.6, 0.9, 1.2, 3.0],
            "alpha_deg": [-2, 0, 2],
            "beta_deg": 0,
            "reynolds_per_meter": "Mach * 2e7",
            "note": "encoding probes use controlled conditions, not flight fits",
        },
        "manual_reference": {},
        "part_checks": [],
        "encoding_checks": [],
        "nozzle_inputs": [],
        "nozzle_sensitivity": [],
        "unit_sensitivity": [],
    }
    sample = c.ROOT / "tests/fixtures/surfaces/data/datcom/manual_example_for005.dat"
    worked = run_datcom(EXE, sample.read_text())
    report["manual_reference"] = dict(
        source="Figures 19-21, printed pp. 65-67",
        mach=float(worked[4][0]),
        published_ca_alpha0=0.466,
        native_ca_alpha0=float(worked.body["CA"][0, 0]),
        published_cn_alpha4=1.179,
        native_cn_alpha4=float(worked.body["CN"][0, 1]),
    )
    results = pd.read_csv(OUT / "rockets/results.csv")
    for _, row in results[results.status == "paired untuned comparison"].iterrows():
        name = row.rocket
        print("AUDIT", name, flush=True)
        # Exact final zero-beta case, same geometry/conditions. PART only changes
        # diagnostic output; all force/moment and dynamic tables must agree.
        deck = (OUT / "rockets" / f"{name}_coast_for005.dat").read_text().split(
            "NEXT CASE"
        )[2] + "NEXT CASE\n"
        plain = run_datcom(EXE, deck)
        partial = run_datcom(EXE, deck.replace("NEXT CASE", "PART\nNEXT CASE"))
        max_body = max(
            float(np.max(np.abs(plain.body[key] - partial.body[key])))
            for key in plain.body
        )
        max_damp = max(
            float(np.max(np.abs(plain.damping[key] - partial.damping[key])))
            for key in DAMPING_KEYS
        )
        assert max_body == 0 and max_damp == 0
        report["part_checks"].append(
            dict(
                rocket=name,
                max_body_difference=max_body,
                max_damping_difference=max_damp,
                diagnostics=partial.diagnostics,
            )
        )
        controlled = conditions(deck, [0.3, 0.6, 0.9, 1.2, 3.0])
        baseline = run_datcom(EXE, controlled)
        for encoding, equivalent in variants(controlled).items():
            other = run_datcom(EXE, equivalent)
            max_ca = float(np.max(np.abs(other.body["CA"] - baseline.body["CA"])))
            max_cn = float(np.max(np.abs(other.body["CN"] - baseline.body["CN"])))
            report["encoding_checks"].append(
                dict(
                    rocket=name,
                    encoding=encoding,
                    max_ca_difference=max_ca,
                    max_cn_difference=max_cn,
                    zero_alpha_ca_reference=baseline.body["CA"][:, 1].tolist(),
                    zero_alpha_ca_alternative=other.body["CA"][:, 1].tolist(),
                )
            )
        alone = run_datcom(EXE, conditions(deck, [0.6]))
        report["encoding_checks"].append(
            dict(
                rocket=name,
                encoding="Mach 0.6 alone versus larger Mach grid",
                max_ca_difference=float(
                    np.max(np.abs(alone.body["CA"][0] - baseline.body["CA"][1]))
                ),
                max_cn_difference=float(
                    np.max(np.abs(alone.body["CN"][0] - baseline.body["CN"][1]))
                ),
            )
        )
        with contextlib.redirect_stdout(io.StringIO()):
            rocket, env, launch, _ = c.build_model(
                c.ROOT / f"docs/examples/{name}_flight_sim.ipynb"
            )
        description = [
            part for part in (rocket.motor.description_eng_file or []) if part
        ]
        case_diameter = float(description[1]) / 1000 if len(description) >= 7 else None
        report["nozzle_inputs"].append(
            dict(
                rocket=name,
                motor_description=description,
                source_nozzle_radius_m=rocket.motor.nozzle_radius,
                assumed_datcom_exit_diameter_m=2 * rocket.motor.nozzle_radius,
                listed_motor_case_diameter_m=case_diameter,
                exceeds_listed_case=bool(
                    case_diameter and 2 * rocket.motor.nozzle_radius > case_diameter
                ),
                note="case diameter is not a measured nozzle exit diameter",
            )
        )
        # Broad bound: full powered base drag. It isolates base-area uncertainty;
        # it is not asserted to describe the actual powered jet or flown rocket.
        provenance = json.loads(
            (OUT / "rockets" / f"{name}_provenance.json").read_text()
        )
        c.h.attach_datcom(rocket, EXE, 0)
        options = dict(
            mach=provenance["mach"],
            alpha=provenance["alpha"],
            beta=provenance["beta"],
            reynolds=provenance["atmosphere"]["reynolds"],
            powered_base_drag="full",
        )
        alternatives = [("full powered base (bound)", None)]
        if name == "genesis":
            alternatives += [
                ("exit equal to listed case diameter (upper bound)", 0.075),
                ("source radius treated as diameter (unverified hypothesis)", 0.042354),
            ]
        for label, diameter in alternatives:
            trial = dict(options)
            if diameter is not None:
                trial.update(powered_base_drag="nozzle", nozzle_exit_diameter=diameter)
            for fin in rocket.fins:
                if hasattr(fin, "datcom_options"):
                    fin.datcom_options = dict(trial)
            with contextlib.redirect_stdout(io.StringIO()):
                flight = Flight(rocket, env, **launch)
            altitude = float(flight.apogee - env.elevation)
            report["nozzle_sensitivity"].append(
                dict(
                    rocket=name,
                    scenario=label,
                    assumed_exit_diameter_m=diameter,
                    source_native_apogee_agl_m=row.datcom_agl_m,
                    alternative_apogee_agl_m=altitude,
                    recorded_apogee_agl_m=row.actual_agl_m,
                    alternative_error_pct=100 * (altitude / row.actual_agl_m - 1),
                    fitted=False,
                    motor_nozzle_radius_unchanged_m=rocket.motor.nozzle_radius,
                )
            )
        if name in ("andromeda", "astra", "genesis", "valetudo"):
            # Equivalent units isolate the native transonic sensitivity from
            # nozzle assumptions. Keep the saved flight's original nozzle mode.
            for fin in rocket.fins:
                if hasattr(fin, "datcom_options"):
                    fin.datcom_options = dict(options, powered_base_drag="nozzle")
            rocket._invalidate_datcom()

            def in_inches(path, input_deck, timeout):
                equivalent = variants(input_deck)["inch_geometry_feet_reynolds"]
                return run_datcom(path, equivalent, timeout)

            with (
                patch("rocketpy.rocket.datcom.run_datcom", side_effect=in_inches),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                flight = Flight(rocket, env, **launch)
            altitude = float(flight.apogee - env.elevation)
            report["unit_sensitivity"].append(
                dict(
                    rocket=name,
                    metric_apogee_agl_m=row.datcom_agl_m,
                    inch_apogee_agl_m=altitude,
                    change_m=altitude - row.datcom_agl_m,
                    recorded_apogee_agl_m=row.actual_agl_m,
                    fitted=False,
                )
            )
        (OUT / "datcom_usage_audit.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
