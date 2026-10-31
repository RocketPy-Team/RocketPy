"""Reproduce 20%-minimum CD / 2%-CP ascents with saved comparison conditions.

No apogee fit or nominal comparison data is overwritten. These transferred
engineering assumptions are not statistical confidence bounds.
"""

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path

import pandas as pd

import model_sources as sources
from rocketpy import Flight

OUT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rockets",
        nargs="+",
        help="Defaults to all rockets with saved DATCOM results.",
    )
    args = parser.parse_args()
    if args.rockets is None:
        metrics = pd.read_csv(OUT / "rocket_metrics.csv")
        args.rockets = metrics.loc[
            (metrics.metric == "Apogee AGL") & metrics.datcom.notna(), "rocket"
        ].tolist()
    executable = Path(
        os.environ.get(
            "DATCOM_EXECUTABLE",
            sources.ROOT.parent / "Asteria-Flight-Dynamics/works/dat0/datcom",
        )
    )
    report = {
        "source": "https://doi.org/10.1088/1757-899X/612/3/032096",
        "interpretation": "CD excursions are at least 20%; larger native-relative Wu CA discrepancies are used inside Mach 0.8–3.02 and alpha ±6 degrees. Outside that envelope use 20%. Nose-to-CP distance varies by 2%, with static moments transferred consistently. Native damping is unchanged. These are engineering scenarios, not confidence intervals.",
        "design_reference": "https://doi.org/10.2514/6.1995-1893",
        "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "rockets": {},
    }
    rows = []
    for name in args.rockets:
        print(f"SENSITIVITY {name}", flush=True)
        with contextlib.redirect_stdout(io.StringIO()):
            rocket, env, launch, _ = sources.build_model(
                sources.ROOT / f"docs/examples/{name}_flight_sim.ipynb"
            )
            sources.h.attach_datcom(rocket, executable, 0)
        provenance = json.loads(
            (OUT / "rockets" / f"{name}_provenance.json").read_text()
        )
        for fin in rocket.fins:
            if hasattr(fin, "datcom_options"):
                fin.datcom_options = dict(
                    mach=provenance["mach"],
                    alpha=provenance["alpha"],
                    beta=provenance["beta"],
                    reynolds=provenance["atmosphere"]["reynolds"],
                    powered_base_drag="nozzle",
                )
        flight = Flight(rocket, env, **launch)
        result = flight.datcom_uncertainty
        if result is None:
            raise RuntimeError(f"{name}: {flight.datcom_uncertainty_status}")
        record = {
            "policy": result.policy,
            "cases": result.cases,
            "errors": result.errors,
            "apogee_uncertainty_m": flight.datcom_apogee_uncertainty,
        }
        for case, values in result.cases.items():
            rows.append(dict(rocket=name, case=case, **values))
            replay = flight if case == "nominal" else result.flights[case]
            record.setdefault("native_envelope", {})[case] = (
                sources.h.envelope_diagnostics(
                    replay,
                    replay._datcom_load_model,
                    provenance["mach"],
                    provenance["alpha"],
                )
            )
        report["rockets"][name] = record
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            flight.prints.apogee_conditions()
            flight.prints.stability_margin()
        (OUT / f"{name}_uncertainty.txt").write_text(output.getvalue())
        (OUT / "uncertainty_sensitivity.json").write_text(
            json.dumps(report, indent=2, allow_nan=False)
        )
        pd.DataFrame(rows).to_csv(OUT / "uncertainty_sensitivity.csv", index=False)
        print(
            f"{name}: {flight.apogee - env.elevation:.3f} ± {flight.datcom_apogee_uncertainty:.3f} m (AGL)",
            flush=True,
        )


if __name__ == "__main__":
    main()
