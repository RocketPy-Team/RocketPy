"""Compare published Missile DATCOM examples with RocketPy-generated geometry.

The manual example is recreated through the public Rocket/fin APIs. SOSE is
added to the raw deck only for the matching-method check; no production default
or measured-rocket comparison is changed. The Ansys wrapper demonstration is
reconstructed from its documented numerical inputs to inspect native defaults.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np

import model_sources as sources
from rocketpy import Rocket
from rocketpy.rocket.datcom import (
    DAMPING_KEYS,
    _namelist,
    datcom_decks,
    parse_datcom_output,
)

OUT = Path(__file__).resolve().parent
EXE = Path(
    os.environ.get(
        "DATCOM_EXECUTABLE",
        sources.ROOT.parent / "Asteria-Flight-Dynamics/works/dat0/datcom",
    )
)
ANSYS_URL = "https://ansyshelp.ansys.com/public/Views/Secured/ModelCenter/v252/en/ModelCenter_Rem_Exec/ModelCenter_Rem_Exec/mcre_functions/wrappers_components/FileWrapper/examples/ref_fileWrapper_example_missiledatcom.html"


def native_text(deck):
    """Read diagnostic/trim output independently, retaining no executable copy."""
    with tempfile.TemporaryDirectory(prefix="rocketpy-reference-review-") as directory:
        binary = Path(directory, "datcom")
        shutil.copy2(EXE, binary)
        Path(directory, "for005.dat").write_text(deck, encoding="ascii")
        subprocess.run(
            [str(binary)], cwd=directory, capture_output=True, timeout=60, check=True
        )
        raw = Path(directory, "for006.dat").read_text()
    if "** ERROR **" in raw:
        raise RuntimeError("Native input error in reference probe")
    return raw


def tables(deck):
    return parse_datcom_output(".", _output=native_text(deck))


def manual_model():
    inch = 0.0254
    rocket = Rocket(
        radius=3.75 * inch / 2,
        mass=1,
        inertia=(1, 1, 1),
        power_off_drag=0,
        power_on_drag=0,
        center_of_mass_without_motor=18.75 * inch,
    )
    rocket.add_nose(length=11.25 * inch, kind="ogive", position=37.5 * inch)
    options = dict(
        mach=[2.35, 2.36],
        alpha=[-4, 0, 4, 8, 12, 16, 20, 24, 28],
        beta=[0],
        reynolds=3e6 / 0.3048,
        body_length=26.25 * inch,
    )
    for n, xle, root, tip, span, maximum, flat, angles in [
        (2, 15.42, 6.96, 0, 3.48, 0.238, 0.524, [90, 270]),
        (4, 31.915, 5.585, 2.792, 4.385, 0.288, 0.428, [0, 90, 180, 270]),
    ]:
        rocket.add_datcom_hex_fins(
            n=n,
            root_chord=root * inch,
            tip_chord=tip * inch,
            span=span * inch,
            position=(37.5 - xle) * inch,
            sweep_length=(root - tip) * inch,
            path=EXE,
            thickness_to_chord=0.04476,
            max_thickness_position=maximum,
            flat_length=flat,
            leading_edge_radius=0.015 * inch,
            panel_angles=angles,
            datcom_options=options,
        )
    return rocket


def ansys_deck(upper):
    return "\n".join(
        [
            "DIM IN",
            "HYPER",
            _namelist(
                "FLTCON",
                [
                    ("NMACH", 1),
                    ("MACH", 1),
                    ("ALT", 0),
                    ("NALPHA", 2),
                    ("ALPHA", [0, 10]),
                    ("BETA", 0),
                ],
            ),
            _namelist(
                "REFQ",
                [
                    ("XCG", 81),
                    ("LREF", 21),
                    ("SREF", 346.4),
                    ("SCALE", 1),
                    ("ROUGH", 2.5e-4),
                    ("BLAYER", "TURB"),
                ],
            ),
            _namelist(
                "AXIBOD",
                [
                    ("TNOSE", "CONE"),
                    ("LNOSE", 20),
                    ("DNOSE", 20),
                    ("LCENTR", 140),
                    ("DCENTR", 20),
                    ("TAFT", "CONE"),
                    ("LAFT", 20),
                    ("DAFT", 20),
                    ("DEXIT", 16),
                ],
            ),
            _namelist(
                "FINSET1",
                [
                    ("SECTYP", "ARC"),
                    ("ZUPPER", upper),
                    ("LMAXU", [0.4, 0.4]),
                    ("LFLATU", [0, 0]),
                    ("SSPAN", [10, 15]),
                    ("CHORD", [20, 10.8]),
                    ("SWEEP", 30),
                    ("XLE", 0),
                    ("STA", 0),
                    ("NPANEL", 4),
                    ("PHIF", [45, 135, 225, 315]),
                ],
            ),
            _namelist(
                "TRIM", [("SET", 1), *[(f"PANL{i}", ".TRUE.") for i in range(1, 5)]]
            ),
            "PART",
            "DAMP",
            "SAVE",
            "NEXT CASE",
            "",
        ]
    )


def main():
    reference = (
        sources.ROOT / "tests/fixtures/surfaces/data/datcom/manual_example_for005.dat"
    ).read_text()
    published = tables(reference)
    deck = datcom_decks(manual_model())[0][False]
    (OUT / "manual_generated_for005.dat").write_text(deck)
    generated = tables(deck)
    sose = tables(deck.replace("NEXT CASE", "SOSE\nPART\nNEXT CASE"))
    aligned_conditions = _namelist(
        "FLTCON",
        [
            ("NMACH", 2),
            ("MACH", [2.35, 2.36]),
            ("REN", [3e6, 3e6]),
            ("NALPHA", 9),
            ("ALPHA", [-4, 0, 4, 8, 12, 16, 20, 24, 28]),
            ("BETA", 0),
        ],
    )
    aligned = tables(
        re.sub(r"\$FLTCON.*?\$", lambda _: aligned_conditions, reference, flags=re.S)
    )
    result = {
        "executable_sha256": hashlib.sha256(EXE.read_bytes()).hexdigest(),
        "scope": "Reference-example audit; original flight comparisons and production defaults unchanged",
        "manual_source": "Missile DATCOM Users Guide, Figures 19-21, printed pp. 65-67",
        "manual_source_units": "DIM IN; REN per foot",
        "generated_units": "DIM M; REN per metre; identical physical reference area/length/CG",
        "manual_comparison": {},
        "ansys_source": ANSYS_URL,
        "ansys_sections": [],
        "low_mach_method_checks": [],
    }
    for key in ["CA", "CN", "CM"]:
        expected = published.body[key][0]
        matched = sose.body[key][-1, 1:]
        result["manual_comparison"][key] = dict(
            alpha_deg=published[5].tolist(),
            published=expected.tolist(),
            generated_default=generated.body[key][-1, 1:].tolist(),
            generated_with_sose=matched.tolist(),
            max_difference_same_method=float(np.max(np.abs(expected - matched))),
        )
        np.testing.assert_allclose(matched, expected, atol=0.002, rtol=0)
    result["manual_damping_grid_effect"] = {
        "change": "Add alpha=-4 degrees ahead of the published 0..28 schedule; extra Mach 2.35 alone was checked to have no effect",
        "cna_at_zero_original_per_degree": float(published.derivatives["CNA"][0, 0]),
        "cna_at_zero_signed_per_degree": float(aligned.derivatives["CNA"][-1, 1]),
        "max_dynamic_difference_by_channel": {
            k: float(
                np.max(np.abs(published.damping[k][0] - aligned.damping[k][-1, 1:]))
            )
            for k in DAMPING_KEYS
        },
    }
    result["manual_damping_max_difference_same_method_and_grid"] = max(
        float(np.max(np.abs(aligned.damping[k][-1] - sose.damping[k][-1])))
        for k in DAMPING_KEYS
    )
    assert result["manual_damping_max_difference_same_method_and_grid"] <= 0.002
    for label, upper in [
        ("root_only_as_published", [0.04]),
        ("root_and_tip_explicit", [0.04, 0.04]),
    ]:
        raw = native_text(ansys_deck(upper))
        thickness = float(
            re.search(r"MAXIMUM AIRFOIL THICKNESS\s*=\s*([\d.]+)", raw)[1]
        )
        result["ansys_sections"].append(
            dict(
                variant=label,
                zupper=upper,
                effective_total_thickness_to_chord=thickness,
                output_kind="trimmed coefficients; unsupported by the passive whole-vehicle table parser",
            )
        )
    untrimmed = re.sub(r"\$TRIM.*?\$", "", ansys_deck([0.04]), flags=re.S)
    native = tables(untrimmed)
    normalized = tables(
        untrimmed.replace("SREF = 346.4", f"SREF = {np.pi * 10**2:.10f}").replace(
            "LREF = 21.", "LREF = 20."
        )
    )
    result["ansys_reference_normalization"] = dict(
        area_ratio=346.4 / (np.pi * 10**2),
        example_ca_alpha0=float(native.body["CA"][0, 0]),
        frontal_area_ca_alpha0=float(normalized.body["CA"][0, 0]),
        untrimmed_cd_alpha10=float(native[1][0, 1]),
        published_trimmed_cd_alpha10=0.736,
        note="Three-decimal coefficient printing limits normalization comparisons; trimmed and untrimmed CD are different states",
    )
    for name in ["genesis", "andromeda", "astra"]:
        source = (OUT / "rockets" / f"{name}_coast_for005.dat").read_text().split(
            "NEXT CASE"
        )[2] + "NEXT CASE\n"
        # Minimal same-atmosphere subsonic/transonic probe: only method changes.
        flight = _namelist(
            "FLTCON",
            [
                ("NMACH", 5),
                ("MACH", [0.3, 0.6, 0.9, 1.05, 1.2]),
                ("REN", np.array([0.3, 0.6, 0.9, 1.05, 1.2]) * 2e7),
                ("NALPHA", 3),
                ("ALPHA", [-2, 0, 2]),
                ("BETA", 0),
            ],
        )
        source = re.sub(r"\$FLTCON.*?\$", lambda _: flight, source, flags=re.S)
        a = tables(source)
        b = tables(source.replace("NEXT CASE", "SOSE\nNEXT CASE"))
        result["low_mach_method_checks"].append(
            dict(
                rocket=name,
                mach=a[4].tolist(),
                max_ca_difference=float(np.abs(a.body["CA"] - b.body["CA"]).max()),
                max_cn_difference=float(np.abs(a.body["CN"] - b.body["CN"]).max()),
                max_cm_difference=float(np.abs(a.body["CM"] - b.body["CM"]).max()),
            )
        )
    (OUT / "reference_example_audit.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
