"""Isolate Andromeda drag discrepancies without fitting or changing baselines.

Run from the repository root. Requires a native DATCOM executable at
DATCOM_EXECUTABLE or the neighbouring Asteria repository's usual location.
All alternative sections/roughnesses are hypotheses, not measured hardware.
"""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import re
import sys
from types import MethodType
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = next(p for p in OUT.parents if (p / "rocketpy").is_dir())
sys.path.insert(0, str(ROOT))

from audit_fin_parameters import fin_fields  # noqa: E402
from audit_reference_examples import EXE, native_text  # noqa: E402
import model_sources as sources  # noqa: E402
from rocketpy import Flight, Function  # noqa: E402
from rocketpy.rocket.datcom import _namelist, parse_datcom_output, run_datcom  # noqa: E402

MACH = np.array([0.1, 0.3, 0.6, 0.8, 0.9, 1.0, 1.05, 1.2])
VARIANTS = (
    "baseline",
    "constant_4mm_5mm_bevel",
    "ratio_0.04",
    "user_sharp_both",
    "user_blunt_TE",
    "user_square_edges",
    "rough_6.35um",
    "rough_25.4um",
    "rough_100um",
)


def alter(deck, variant):
    """Change only the named input; ROUGH is centimetres even with DIM M."""
    if variant.startswith("rough_"):
        height_m = float(variant.removeprefix("rough_").removesuffix("um")) * 1e-6
        return deck.replace("$REFQ", f"$REFQ\n  ROUGH = {100 * height_m:.9g},")
    if variant == "user_square_edges":
        deck = fin_fields(deck, "user_blunt_TE")
        return re.sub(r"(YUPPER\s*=\s*)0\.,", r"\g<1>0.01,", deck).replace(
            "YLOWER = -0.,", "YLOWER = -0.01,"
        )
    return fin_fields(deck, variant)


def components(raw):
    """Read printed PART breakdowns independently of the production parser."""
    body, fin = {}, {}
    for block in re.split(r"\n1\s+\*", raw):
        match = re.search(r"MACH NO\s*=\s*([\d.]+)", block)
        if not match:
            continue
        mach = float(match[1])
        if "BODY ALONE PARTIAL OUTPUT" in block:
            zero = re.search(r"^\s+0\.00\s+(.*)$", block, re.M)
            values = [float(x) for x in zero[1].split()]
            # At alpha=0 the final entry is CA-ALP. Blank PROT/SEP are zero.
            assert abs(values[-1]) < 1e-9 and len(values) == 4
            body[mach] = values[:3]
        if "FIN SET 1 CA PARTIAL OUTPUT" in block:
            panel = float(re.search(r"TOTAL CAO\s+([\d.]+)", block)[1])
            total = float(re.search(r"^\s+0\.00\s+[\d.-]+\s+([\d.]+)", block, re.M)[1])
            fin[mach] = panel, total
    return [
        dict(
            mach=m,
            body_friction_cd=values[0],
            body_pressure_cd=values[1],
            body_base_cd=values[2],
            one_fin_cd=fin[m][0],
            three_fins_cd=fin[m][1],
            printed_component_sum_cd=sum(values) + fin[m][1],
        )
        for m, values in body.items()
    ]


def main():
    source_path = ROOT / "data/rockets/andromeda/drag_coefficient.csv"
    supplied = np.loadtxt(source_path, delimiter=",")
    supplied_cd = Function(
        source_path, interpolation="linear", extrapolation="constant"
    )
    peak = int(np.argmax(supplied[:, 0]))
    branch_mach = np.linspace(0.2, 0.8, 200)
    branch_gap = np.interp(branch_mach, *supplied[: peak + 1].T) - np.interp(
        branch_mach, *supplied[peak:][::-1].T
    )
    provenance = json.loads((OUT / "rockets/andromeda_provenance.json").read_text())
    report = dict(
        fitted=False,
        hardware_section_verified=False,
        executable_sha256=hashlib.sha256(EXE.read_bytes()).hexdigest(),
        drag_source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        source_curve=dict(
            minimum_mach=float(supplied[:, 0].min()),
            maximum_mach=float(supplied[:, 0].max()),
            rows=len(supplied),
            descending_mach_steps=int(np.sum(np.diff(supplied[:, 0]) < 0)),
            exact_duplicate_mach_rows=len(supplied) - len(np.unique(supplied[:, 0])),
            same_file_for_powered_and_coast=True,
            extrapolation="constant; no measured-CD provenance is supplied",
            maximum_branch_cd_difference_mach02_to08=float(np.max(abs(branch_gap))),
        ),
        manual="Missile DATCOM Users Guide, rev 5/97: Table 4 (ROUGH/BLAYER), Table 5 (DEXIT), section 3.1.6 (SSPAN), section 4.2.2 (airfoil approximations)",
        hypotheses="No fin thickness, edge shape or roughness measurement is supplied. 6.35 um is the manual's carefully applied matte-paint example, not a carbon/vinyl material assignment. Square USER edges use approximate native section methods.",
        external_hardware_evidence=dict(
            checked_date="2026-11-06",
            launch_report="https://asat.gr/asat-launches-twice-and-takes-7th-place-in-the-european-rocketry-challenge-2022-euroc/",
            automatic_control="https://asat.gr/automatic-control/",
            aerodynamics="https://asat.gr/aerodynamics/",
            finding="The team reports airbrakes used on both 2022 EuRoC flights; neither comparison model reconstructs their geometry or deployment. The boattail was not used. CSV provenance and deployment history remain unknown.",
            team_reported_cots_apogee_m=3446,
            notebook_recorded_apogee_m=3443,
        ),
    )
    rows, breakdowns = [], []
    report["native_diagnostics"] = {}
    for phase, label in [(True, "powered"), (False, "coast")]:
        saved = (OUT / "rockets" / f"andromeda_{label}_for005.dat").read_text()
        deck = saved.split("NEXT CASE")[2] + "NEXT CASE\n"
        flight_conditions = _namelist(
            "FLTCON",
            [
                ("NMACH", len(MACH)),
                ("MACH", MACH),
                (
                    "REN",
                    np.interp(
                        MACH,
                        provenance["mach"],
                        provenance["atmosphere"]["reynolds"][
                            "power_on" if phase else "power_off"
                        ],
                    ),
                ),
                ("NALPHA", 3),
                ("ALPHA", [-2, 0, 2]),
                ("BETA", 0),
            ],
        )
        deck = re.sub(r"\$FLTCON.*?\$", lambda _: flight_conditions, deck, flags=re.S)
        for variant in VARIANTS:
            probe = alter(deck, variant)
            if variant == "baseline":
                probe = probe.replace("NEXT CASE", "PART\nNEXT CASE")
            raw = native_text(probe)
            table = parse_datcom_output(".", _output=raw)
            report["native_diagnostics"][f"{label}/{variant}"] = table.diagnostics
            np.testing.assert_allclose(
                table.body["CA"][:, 1], table[1][:, 1], atol=1e-12
            )
            if variant == "baseline":
                stem = OUT / "rockets" / f"andromeda_audit_{label}"
                stem.with_name(stem.name + "_for005.dat").write_text(probe)
                stem.with_name(stem.name + "_for006.dat").write_text(
                    "\n".join(line.rstrip() for line in raw.splitlines()) + "\n"
                )
                part = components(raw)
                for item, cd in zip(part, table[1][:, 1]):
                    assert abs(item["printed_component_sum_cd"] - cd) < 0.001
                    breakdowns.append(dict(phase=label, **item, native_cd=float(cd)))
            for m, cd in zip(MACH, table[1][:, 1]):
                rows.append(
                    dict(
                        phase=label,
                        variant=variant,
                        mach=m,
                        datcom_cd=float(cd),
                        supplied_cd=float(supplied_cd(m)),
                    )
                )
    curves = pd.DataFrame(rows)
    curves.to_csv(OUT / "andromeda_drag_audit_curves.csv", index=False)
    pd.DataFrame(breakdowns).to_csv(OUT / "andromeda_drag_breakdown.csv", index=False)

    with contextlib.redirect_stdout(io.StringIO()):
        rocket, env, launch, _ = sources.build_model(
            ROOT / "docs/examples/andromeda_flight_sim.ipynb"
        )
    original = {True: rocket.power_on_drag_7d, False: rocket.power_off_drag_7d}
    np.testing.assert_allclose(rocket.area, 0.017836777135856116, atol=1e-12)
    report["geometry"] = dict(
        reference_area_m2=float(rocket.area),
        diameter_m=2 * rocket.radius,
        length_m=2.4996,
        nose_length_m=0.3,
        nose_kind="von Karman",
        fins=dict(
            n=3,
            span_m=0.185,
            root_chord_m=0.23,
            tip_chord_m=0.11,
            sweep_m=0.12,
            root_le_from_nose_m=2.145,
            tip_le_from_nose_m=2.265,
        ),
        assumed_section="2% HEX with sharp ends and 5%-chord bevels: root 4.6 mm thick, tip 2.2 mm",
    )
    flights = []
    for variant in [
        "ordinary_supplied_cd",
        "baseline",
        "supplied_zero_cd_in_datcom",
        "supplied_powered_cd_in_datcom",
        "supplied_coast_cd_in_datcom",
        "user_blunt_TE",
        "user_square_edges",
        "rough_6.35um",
        "rough_25.4um",
    ]:
        with contextlib.redirect_stdout(io.StringIO()):
            rocket, env, launch, _ = sources.build_model(
                ROOT / "docs/examples/andromeda_flight_sim.ipynb"
            )
            if variant != "ordinary_supplied_cd":
                sources.h.attach_datcom(rocket, EXE, 0)
                for fin in rocket.fins:
                    fin.datcom_options = dict(
                        mach=provenance["mach"],
                        alpha=provenance["alpha"],
                        beta=provenance["beta"],
                        reynolds=provenance["atmosphere"]["reynolds"],
                        powered_base_drag="nozzle",
                    )

                def native(path, input_deck, timeout):
                    return run_datcom(
                        path,
                        alter(
                            input_deck,
                            "baseline" if variant.startswith("supplied_") else variant,
                        ),
                        timeout,
                    )

                with patch("rocketpy.rocket.datcom.run_datcom", side_effect=native):
                    model = rocket.prepare_datcom()
                if variant.startswith("supplied_"):
                    native_coefficients = model.coefficients

                    def replacement(self, powered, mach, alpha, beta=0):
                        values = native_coefficients(powered, mach, alpha, beta).copy()
                        if (
                            variant == "supplied_powered_cd_in_datcom" and not powered
                        ) or (variant == "supplied_coast_cd_in_datcom" and powered):
                            return values
                        zero = native_coefficients(powered, mach, 0, 0)
                        delta = original[powered](0, 0, mach, 1e6, 0, 0, 0) - zero[2]
                        values[2] += delta
                        values[-1] += delta * np.cos(alpha)
                        return values

                    model.coefficients = MethodType(replacement, model)
            flight = Flight(rocket, env, **launch, datcom_uncertainty=False)
        apogee = float(flight.apogee - env.elevation)
        record = dict(
            variant=variant,
            apogee_agl_m=apogee,
            recorded_agl_m=3443,
            error_pct=100 * (apogee / 3443 - 1),
            peak_mach=float(flight.max_mach_number),
        )
        flights.append(record)
        print(record, flush=True)
    report["flights"] = flights
    report["mach06"] = curves[curves.mach == 0.6].to_dict("records")
    report["substitution_scope"] = (
        "Only the native zero-incidence axial-force curve is replaced. Native incidence increments, lateral coefficients, moments, damping, CG, motor and atmosphere are retained. This is a causal diagnostic, not a native DATCOM prediction or calibration."
    )
    baseline = next(r for r in flights if r["variant"] == "baseline")
    np.testing.assert_allclose(
        baseline["apogee_agl_m"], 4170.733199067169, atol=0.1, rtol=0
    )
    (OUT / "andromeda_drag_audit.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    pd.DataFrame(flights).to_csv(OUT / "andromeda_drag_audit_flights.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for axis, phase in zip(axes, ["powered", "coast"]):
        for variant, color, title in [
            ("baseline", "#d17a19", "DATCOM baseline: 2% HEX, bevelled edges"),
            ("user_blunt_TE", "#638c42", "Hypothesis: blunt trailing edge"),
            (
                "rough_25.4um",
                "#9467bd",
                "Hypothesis: roughness 25.4 µm",
            ),
            ("rough_6.35um", "#955b3a", "Hypothesis: roughness 6.35 µm"),
        ]:
            values = curves[(curves.phase == phase) & (curves.variant == variant)]
            axis.plot(values.mach, values.datcom_cd, color=color, label=title)
        source_mach = np.linspace(0.02, 1.2, 501)
        axis.plot(
            source_mach,
            supplied_cd(source_mach),
            color="#3979b7",
            linewidth=2,
            label="RocketPy supplied curve (not measured CD)",
        )
        axis.axvspan(
            supplied[:, 0].max(),
            1.2,
            color="#eeeeee",
            zorder=0,
            label="Beyond supplied Mach data",
        )
        axis.set(
            title=phase.capitalize(),
            xlabel="Mach number",
            ylabel="Zero-incidence CD",
            xlim=(0.02, 1.2),
            ylim=(0, 1.1),
        )
        axis.grid(alpha=0.2)
    fig.suptitle("Andromeda: drag assumptions and supplied-curve coverage", fontsize=15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9)
    fig.tight_layout(rect=(0, 0.20, 1, 0.95))
    fig.savefig(OUT / "andromeda_drag_audit.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
