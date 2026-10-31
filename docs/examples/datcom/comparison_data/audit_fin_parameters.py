"""Probe unspecified fin sections without changing the baseline comparisons.

All alternative dimensions below are sensitivity assumptions, not measurements
or fitted geometry. Native tables are regenerated; no CD multiplier is applied.
"""

import contextlib
import hashlib
import io
import json
import re
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import audit_datcom_usage as audit
from rocketpy import Flight
from rocketpy.rocket.datcom import _namelist, run_datcom

OUT = Path(__file__).resolve().parent


def fin_fields(deck, label):
    def change(match):
        values = {}
        for field in re.finditer(r"(\w+)\s*=\s*(.*?)(?=\b\w+\s*=|$)", match[2], re.S):
            key, raw = field[1], field[2].strip(" ,\n")
            try:
                val = np.array([float(v) for v in raw.split(",")])
            except ValueError:
                val = raw
            values[key] = val
        if not match[1].startswith("FINSET"):
            return match[0]
        chord = values["CHORD"]
        n = len(chord)
        if label.startswith("ratio_"):
            total = float(label[6:])
            values.update(ZUPPER=np.full(n, total / 2), ZLOWER=np.full(n, total / 2))
        elif label.startswith("bevel_"):
            bevel = float(label[6:])
            values.update(
                LMAXU=np.full(n, bevel),
                LMAXL=np.full(n, bevel),
                LFLATU=np.full(n, 1 - 2 * bevel),
                LFLATL=np.full(n, 1 - 2 * bevel),
            )
        elif label.startswith("radius_"):
            values["LER"] = np.full(n, float(label[7:]))
        elif label == "constant_4mm_5mm_bevel":
            values.update(
                ZUPPER=0.002 / chord,
                ZLOWER=0.002 / chord,
                LMAXU=0.005 / chord,
                LMAXL=0.005 / chord,
                LFLATU=1 - 0.01 / chord,
                LFLATL=1 - 0.01 / chord,
            )
        elif label.startswith("user_"):
            for k in ["ZUPPER", "ZLOWER", "LMAXU", "LMAXL", "LFLATU", "LFLATL"]:
                values.pop(k, None)
            x = np.array([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1])
            upper = np.full(7, 0.01)
            if label == "user_sharp_both":
                upper[[0, -1]] = 0
            elif label == "user_blunt_TE":
                upper[0] = 0
            else:
                raise ValueError(f"Unknown USER section variant: {label}")
            values.update(SECTYP="USER", XCORD=x, YUPPER=upper, YLOWER=-upper)
        return _namelist(match[1], list(values.items()))

    return re.sub(r"\$(\w+)\s*(.*?)\$", change, deck, flags=re.S)


LABELS = [
    "baseline",
    "ratio_0.01",
    "ratio_0.04",
    "ratio_0.08",
    "bevel_0.25",
    "bevel_0.45",
    "radius_0.0005",
    "radius_0.001",
    "constant_4mm_5mm_bevel",
    "user_sharp_both",
    "user_blunt_TE",
]
FLIGHT_LABELS = [
    "baseline",
    "constant_4mm_5mm_bevel",
    "ratio_0.04",
    "user_sharp_both",
    "user_blunt_TE",
]


def main():
    report = {
        "fitted": False,
        "hardware_geometry_verified": False,
        "executable_sha256": hashlib.sha256(audit.EXE.read_bytes()).hexdigest(),
        "conditions": {
            "mach": [0.3, 0.6, 0.9, 1.05, 1.2],
            "alpha_deg": [-2, 0, 2],
            "beta_deg": 0,
            "reynolds_per_meter": "Mach * 2e7",
        },
        "interpretation": {
            "baseline": "HEX total thickness/chord 2%, 5% leading/trailing bevels, LER=0",
            "ratio_*": "HEX total thickness/chord; all other inputs unchanged",
            "bevel_*": "HEX bevel length/chord at both edges; total thickness/chord remains 2%",
            "radius_*": "leading-edge radius in metres; all other inputs unchanged",
            "constant_4mm_5mm_bevel": "4 mm total plate thickness, 5 mm bevel length at both edges; ratios vary with spanwise chord",
            "user_sharp_both": "USER profile, total thickness/chord 2%, 5% leading/trailing bevels; control for the section-method change",
            "user_blunt_TE": "same USER profile, 5% leading bevel, full 2% thickness/chord at trailing edge",
            "scope": "Only fin section inputs change. Flight grids, mass, motor, nozzle, atmosphere and planform retain saved comparison inputs.",
            "limitation": "USER uses one normalized section across the span, so the blunt USER plate tapers in physical thickness. Native airfoil-section properties for blunt trailing edges are approximate (guide 4.2.2).",
        },
        "coefficients": [],
        "flights": [],
    }
    results = pd.read_csv(OUT / "rockets/results.csv").set_index("rocket")
    for name in ["genesis", "andromeda", "astra"]:
        deck = (OUT / "rockets" / f"{name}_coast_for005.dat").read_text().split(
            "NEXT CASE"
        )[2] + "NEXT CASE\n"
        controlled = audit.conditions(deck, report["conditions"]["mach"])
        controlled = controlled.replace("NEXT CASE", "PART\nNEXT CASE")
        for label in LABELS:
            try:
                table = run_datcom(audit.EXE, fin_fields(controlled, label))
                record = dict(
                    rocket=name,
                    variant=label,
                    ca_zero=table.body["CA"][:, 1].tolist(),
                    cn_alpha2=table.body["CN"][:, 2].tolist(),
                    diagnostics=table.diagnostics,
                )
            except RuntimeError as exc:
                record = dict(rocket=name, variant=label, error=str(exc))
            report["coefficients"].append(record)
        with contextlib.redirect_stdout(io.StringIO()):
            rocket, env, launch, _ = audit.c.build_model(
                audit.c.ROOT / f"docs/examples/{name}_flight_sim.ipynb"
            )
            audit.c.h.attach_datcom(rocket, audit.EXE, 0)
        provenance = json.loads(
            (OUT / "rockets" / f"{name}_provenance.json").read_text()
        )
        options = dict(
            mach=provenance["mach"],
            alpha=provenance["alpha"],
            beta=provenance["beta"],
            reynolds=provenance["atmosphere"]["reynolds"],
            powered_base_drag="nozzle",
        )
        for fin in rocket.fins:
            if hasattr(fin, "datcom_options"):
                fin.datcom_options = dict(options)
        for label in FLIGHT_LABELS:
            rocket._invalidate_datcom()

            def native(path, input_deck, timeout):
                return run_datcom(path, fin_fields(input_deck, label), timeout)

            with (
                patch("rocketpy.rocket.datcom.run_datcom", side_effect=native),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                flight = Flight(rocket, env, **launch)
            model = rocket.prepare_datcom()
            apogee = float(flight.apogee - env.elevation)
            measured = float(results.loc[name, "actual_agl_m"])
            record = dict(
                rocket=name,
                variant=label,
                apogee_agl_m=apogee,
                recorded_agl_m=measured,
                error_pct=100 * (apogee / measured - 1),
                coast_cd_mach06=float(
                    model.drag_coefficient(False, 0, 0, 0.6, 0, 0, 0, 0)
                ),
                envelope=audit.c.h.envelope_diagnostics(
                    flight, model, provenance["mach"], provenance["alpha"]
                ),
            )
            report["flights"].append(record)
            print(json.dumps(record), flush=True)
        (OUT / "fin_parameter_audit.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    return report


if __name__ == "__main__":
    main()
