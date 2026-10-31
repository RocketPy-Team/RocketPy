"""Missile DATCOM decks, isolated execution and complete aerodynamic tables.

The for006 parser is adapted from Asteria-Flight-Dynamics/classes/datcom/datcom.py.
Geometry and execution are independent of Asteria's project configuration.
"""

import json
import math
import re
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from rocketpy.mathutils.vector_matrix import Vector
from rocketpy.rocket.aero_surface.datcom_fins import (
    DatcomFins,
    DatcomFreeFormFins,
    _station_values,
)
from rocketpy.rocket.aero_surface.fins.trapezoidal_fins import TrapezoidalFins
from rocketpy.rocket.aero_surface.generic_surface import GenericSurface
from rocketpy.rocket.aero_surface.nose_cone import NoseCone
from rocketpy.rocket.aero_surface.tail import Tail
from rocketpy.rocket.datcom_conditions import adaptive_mach_grid

BODY_KEYS = ("CN", "CM", "CA", "CY", "CLN", "CLL")
LATERAL_KEYS = ("CYB", "CLNB", "CLLB")
DAMPING_KEYS = (
    "CNQ",
    "CMQ",
    "CAQ",
    "CNAD",
    "CMAD",
    "CYR",
    "CLNR",
    "CLLR",
    "CYP",
    "CLNP",
    "CLLP",
)


class DatcomResult(tuple):
    """Backward-compatible six-array result with body-axis and lateral tables."""

    def __new__(
        cls,
        values,
        body=None,
        derivatives=None,
        damping=None,
        beta=None,
        sideslip=None,
        diagnostics=None,
    ):
        result = super().__new__(cls, values)
        result.body = {} if body is None else body
        result.derivatives = {} if derivatives is None else derivatives
        result.damping = {} if damping is None else damping
        result.beta = np.array([0.0]) if beta is None else np.asarray(beta)
        result.sideslip = sideslip
        result.diagnostics = [] if diagnostics is None else list(diagnostics)
        return result


def zero_alpha_index(alpha):
    indices = np.flatnonzero(np.isclose(alpha, 0.0, atol=1e-12, rtol=0))
    if len(indices) != 1:
        raise ValueError("DATCOM tables must contain exactly one zero-angle row")
    return int(indices[0])


def parse_datcom_output(base_dir, n_alpha_expected=None, _output=None):
    """Read every angle row, including signed alpha and lateral derivatives.

    The public tuple retains CL, CD, CM, XCP, Mach, alpha (degrees here).
    Body coefficients and derivatives (per degree) are retained as attributes.
    Never truncate a table to a hardcoded row count.
    """
    output = Path(base_dir, "for006.dat").read_text() if _output is None else _output
    lines = output.splitlines()
    diagnostic_terms = (
        "CALCULATIONS IN SUBROUTINE",
        "ANALYSIS FOR THIS AIRFOIL",
        "CREST CRITICAL MACH NUMBER EXCEEDED",
        "HYBRID THEORY INVALID",
        "SECOND ORDER SHOCK EXPANSION TO BE USED",
        "* WARNING *",
    )
    diagnostics = list(
        dict.fromkeys(
            line.strip()
            for line in lines
            if any(term in line for term in diagnostic_terms)
        )
    )
    if n_alpha_expected is None:
        declared = re.search(r"\bNALPHA\s*=\s*([0-9.]+)", output)
        if declared is not None:
            n_alpha_expected = int(float(declared[1]))
    starts = [i for i, line in enumerate(lines) if "----- LONGITUDINAL -----" in line]
    if not starts:
        raise RuntimeError("No LONGITUDINAL blocks found in for006.dat")

    # Multiple fixed-BETA cases can share one executable invocation. Preserve
    # complete body/rate channels at each beta instead of imposing symmetry on
    # arbitrary panel layouts. The six-array view remains the beta=0 slice.
    beta_blocks = {}
    for index, start in enumerate(starts):
        mach_start = next(
            (i for i in range(start - 1, -1, -1) if "MACH NO" in lines[i]), None
        )
        beta_line = next(
            (line for line in reversed(lines[:start]) if "SIDESLIP =" in line),
            "SIDESLIP = 0",
        )
        beta = float(beta_line.split("=")[1].split()[0])
        stop = starts[index + 1] if index + 1 < len(starts) else len(lines)
        if mach_start is not None:
            beta_blocks.setdefault(beta, []).extend(lines[mach_start:stop])
    if len(beta_blocks) > 1:
        tables = {
            beta: parse_datcom_output(base_dir, n_alpha_expected, "\n".join(block))
            for beta, block in sorted(beta_blocks.items())
        }
        if 0.0 not in tables:
            raise RuntimeError("DATCOM sideslip database must include beta=0")
        zero = tables[0.0]
        if any(
            not np.array_equal(t[4], zero[4]) or not np.array_equal(t[5], zero[5])
            for t in tables.values()
        ):
            raise RuntimeError("DATCOM sideslip cases have different Mach/alpha grids")
        return DatcomResult(
            tuple(zero),
            zero.body,
            zero.derivatives,
            zero.damping,
            beta=list(tables),
            sideslip=tables,
            diagnostics=diagnostics,
        )

    def read_table(block, columns, *, last=False):
        headers = [i for i, line in enumerate(block) if line.split() == columns]
        header = (headers[-1] if last else headers[0]) if headers else None
        if header is None:
            raise RuntimeError(f"DATCOM missing {' '.join(columns)} table")
        rows = []
        for line in block[header + 1 :]:
            fields = line.split()
            if not fields and not rows:
                continue
            if len(fields) != len(columns):
                # Fixed-width output can join a negative value to its preceding
                # column, e.g. CL/CD=-0.000 followed by X-C.P.=-28096.705.
                number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
                fields = re.findall(number, line)
                if len(fields) != len(columns) or re.sub(number, "", line).strip():
                    break
            try:
                rows.append([float(value.replace("D", "E")) for value in fields])
            except ValueError as exc:
                raise RuntimeError(
                    "DATCOM returned a nonnumeric coefficient row"
                ) from exc
        if not rows:
            raise RuntimeError("DATCOM returned an empty coefficient table")
        return np.asarray(rows)

    mach, alpha_rows, lift_rows = [], [], []
    body_rows, derivative_rows, damping_rows = [], [], []
    for index, start in enumerate(starts):
        stop = starts[index + 1] if index + 1 < len(starts) else len(lines)
        block = lines[start:stop]
        mach_line = next(
            (line for line in reversed(lines[:start]) if "MACH NO" in line), None
        )
        if mach_line is None:
            raise RuntimeError("DATCOM missing Mach reference")
        mach.append(float(mach_line.split("=")[1].split()[0]))
        body = read_table(block, ["ALPHA", "CN", "CM", "CA", "CY", "CLN", "CLL"])
        lift = read_table(block, ["ALPHA", "CL", "CD", "CL/CD", "X-C.P."])
        derivatives = read_table(block, ["ALPHA", "CNA", "CMA", "CYB", "CLNB", "CLLB"])
        if not np.array_equal(body[:, 0], lift[:, 0]) or not np.array_equal(
            body[:, 0], derivatives[:, 0]
        ):
            raise RuntimeError("DATCOM coefficient tables have different angle grids")
        if n_alpha_expected is not None and len(body) != n_alpha_expected:
            raise RuntimeError("DATCOM angle count does not match the requested count")
        alpha_rows.append(body[:, 0])
        body_rows.append(body[:, 1:])
        lift_rows.append(lift[:, 1:])
        derivative_rows.append(derivatives[:, 1:])
        if "DYNAMIC DERIVATIVES" in "\n".join(block):
            # PART prints body-alone damping before the assembled vehicle.
            # The last pair belongs to the total configuration, like CN/CM/CA.
            longitudinal = read_table(
                block, ["ALPHA", "CNQ", "CMQ", "CAQ", "CNAD", "CMAD"], last=True
            )
            lateral = read_table(
                block,
                ["ALPHA", "CYR", "CLNR", "CLLR", "CYP", "CLNP", "CLLP"],
                last=True,
            )
            if not (
                np.array_equal(body[:, 0], longitudinal[:, 0])
                and np.array_equal(body[:, 0], lateral[:, 0])
            ):
                raise RuntimeError("DATCOM damping angle grid mismatch")
            if "DYNAMIC DERIVATIVES (PER DEGREE)" not in "\n".join(block):
                raise RuntimeError("Expected DATCOM dynamic derivatives per degree")
            damping_rows.append(np.column_stack([longitudinal[:, 1:], lateral[:, 1:]]))
    alpha = alpha_rows[0]
    if any(not np.array_equal(row, alpha) for row in alpha_rows[1:]):
        raise RuntimeError("DATCOM Mach blocks have different angle grids")
    mach = np.asarray(mach)
    if np.any(np.diff(mach) <= 0) or np.any(np.diff(alpha) <= 0):
        raise RuntimeError("DATCOM Mach and angle grids must be strictly increasing")
    body = np.asarray(body_rows)
    lift = np.asarray(lift_rows)
    derivatives = np.asarray(derivative_rows)
    if not all(np.isfinite(values).all() for values in (body, lift, derivatives)):
        raise RuntimeError("DATCOM returned nonfinite coefficient tables")
    if damping_rows and len(damping_rows) != len(mach):
        raise RuntimeError("Incomplete DATCOM damping Mach grid")
    damping = (
        dict(
            zip(
                (
                    "CNQ",
                    "CMQ",
                    "CAQ",
                    "CNAD",
                    "CMAD",
                    "CYR",
                    "CLNR",
                    "CLLR",
                    "CYP",
                    "CLNP",
                    "CLLP",
                ),
                np.moveaxis(np.asarray(damping_rows), 2, 0),
            )
        )
        if damping_rows
        else {}
    )
    if any(not np.isfinite(v).all() for v in damping.values()):
        raise RuntimeError("Nonfinite DATCOM damping coefficients")
    if not damping:
        raise RuntimeError(
            "DATCOM output lacks DAMP derivatives; include DAMP in the deck"
        )
    return DatcomResult(
        (lift[:, :, 0], lift[:, :, 1], body[:, :, 1], lift[:, :, 3], mach, alpha),
        dict(zip(("CN", "CM", "CA", "CY", "CLN", "CLL"), np.moveaxis(body, 2, 0))),
        dict(
            zip(("CNA", "CMA", "CYB", "CLNB", "CLLB"), np.moveaxis(derivatives, 2, 0))
        ),
        damping,
        beta=list(beta_blocks),
        diagnostics=diagnostics,
    )


def _flight_conditions(options):
    """Validate the supported single-altitude-per-Mach DATCOM flight envelope."""
    unknown = set(options) - {
        "mach",
        "alpha",
        "beta",
        "altitude",
        "timeout",
        "body_length",
        "powered_base_drag",
        "nozzle_exit_diameter",
        "reynolds",
    }
    if unknown:
        raise ValueError(f"Unknown DATCOM options: {sorted(unknown)}")
    mach = np.asarray(options.get("mach", adaptive_mach_grid(2)), float)
    alpha = np.asarray(
        options.get(
            "alpha",
            [
                -20,
                -10,
                -5,
                -2,
                -1,
                -0.5,
                -0.25,
                -0.1,
                -0.05,
                0,
                0.05,
                0.1,
                0.25,
                0.5,
                1,
                2,
                5,
                10,
                20,
            ],
        ),
        float,
    )
    for grid, name in ((mach, "mach"), (alpha, "alpha")):
        if (
            grid.ndim != 1
            or not 2 <= len(grid) <= 20
            or not np.isfinite(grid).all()
            or np.any(np.diff(grid) <= 0)
        ):
            raise ValueError(f"DATCOM {name} needs 2..20 finite, increasing values")
    if mach[0] <= 0:
        raise ValueError("DATCOM Mach values must be positive")
    if not alpha[0] < 0 < alpha[-1] or np.count_nonzero(alpha == 0) != 1:
        raise ValueError(
            "DATCOM alpha must include negative angles, zero and positive angles"
        )
    raw_altitude = options.get("altitude", 0)
    if isinstance(raw_altitude, dict):
        if set(raw_altitude) != {"power_on", "power_off"}:
            raise ValueError("altitude schedules need power_on and power_off")
        schedules = {True: raw_altitude["power_on"], False: raw_altitude["power_off"]}
    else:
        schedules = {True: raw_altitude, False: raw_altitude}
    altitude = {}
    for powered, values in schedules.items():
        values = np.asarray(values, float)
        if values.ndim == 0:
            values = np.full(len(mach), float(values))
        if values.shape != mach.shape or not np.isfinite(values).all():
            raise ValueError(
                "DATCOM altitude must be a finite scalar or one value per Mach"
            )
        altitude[powered] = values
    timeout = float(options.get("timeout", 60))
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("DATCOM timeout must be positive and finite")
    return mach, alpha, altitude, timeout


def _namelist(name, items):
    """Wrap Fortran namelist lines below the legacy 80-column input limit."""
    lines = [f" ${name}"]
    for key, values in items:
        if isinstance(values, str):
            tokens = [values]
        else:
            tokens = [f"{float(v):.9G}" for v in np.atleast_1d(values)]
            tokens = [
                token
                if "." in token
                else (token.replace("E", ".E") if "E" in token else token + ".")
                for token in tokens
            ]
        line = f"  {key} = "
        for token in tokens:
            if len(line) + len(token) + 1 > 76:
                lines.append(line)
                line = "    "
            line += token + ","
        lines.append(line)
    lines[-1] += "$"
    return "\n".join(lines)


def datcom_decks(rocket, *, calibration=None, flight_conditions=None):
    """Build powered/coast decks from assembled geometry, without executing.

    DATCOM x is measured aft from the nose tip, in meters. The powered moment
    reference is the filled CG at ignition; the coast reference is the empty
    CG (including the dry motor). No plume or protuberance model is requested.
    """
    surfaces = list(rocket.aerodynamic_surfaces)
    datcom_fins = [
        s for s, _, _ in surfaces if isinstance(s, (DatcomFins, DatcomFreeFormFins))
    ]
    if not datcom_fins:
        raise ValueError("Add a DATCOM fin set before preparing DATCOM aerodynamics")
    first = datcom_fins[0]
    options = dict(first.datcom_options)
    options.update(flight_conditions or {})
    calibration = rocket.datcom_calibration if calibration is None else calibration
    edge_radius = calibration["leading_edge_radius"]
    powered_base_drag = options.get(
        "powered_base_drag", "nozzle" if "nozzle_exit_diameter" in options else "none"
    )
    if powered_base_drag not in ("none", "nozzle", "full"):
        raise ValueError("powered_base_drag must be 'none', 'nozzle' or 'full'")
    if "nozzle_exit_diameter" in options and powered_base_drag != "nozzle":
        raise ValueError("nozzle_exit_diameter requires powered_base_drag='nozzle'")
    mach, alpha, altitude, timeout = _flight_conditions(options)

    def canonical(value):
        return json.dumps(
            value, sort_keys=True, default=lambda v: np.asarray(v).tolist()
        )

    for fins in datcom_fins[1:]:
        if fins.datcom_path != first.datcom_path or canonical(
            fins.datcom_options
        ) != canonical(first.datcom_options):
            raise ValueError(
                "DATCOM fin sets must use the same path and datcom_options"
            )
    noses = [(s, p) for s, p, _ in surfaces if isinstance(s, NoseCone)]
    tails = [(s, p) for s, p, _ in surfaces if isinstance(s, Tail)]
    if len(noses) != 1 or len(tails) > 1:
        raise ValueError("DATCOM requires exactly one nose and at most one tail")
    if any(
        not isinstance(s, (NoseCone, Tail, TrapezoidalFins, DatcomFreeFormFins))
        for s, _, _ in surfaces
    ):
        raise ValueError(
            "DATCOM supports a nose, DATCOM or trapezoidal fins and a conical or ogive tail"
        )
    if any(abs(p.x) > 1e-12 or abs(p.y) > 1e-12 for _, p, _ in surfaces):
        raise ValueError("DATCOM geometry must be centered on the rocket axis")
    nose, tip = noses[0]
    nose_types = {
        "conical": "CONE",
        "ogive": "OGIVE",
        "tangent": "OGIVE",
        "tangentogive": "OGIVE",
        "vonkarman": "KARMAN",
        "lvhaack": "HAACK",
        "powerseries": "POWER",
    }
    kind = nose.kind.replace(" ", "").lower()
    if kind not in nose_types or nose.bluffness:
        raise ValueError(
            "DATCOM requires an unblunted conical, ogive, Haack, Karman or power nose"
        )
    if not math.isclose(nose.base_radius, rocket.radius):
        raise ValueError("DATCOM nose base radius must match the rocket radius")

    def aft(position):
        return (tip.z - position) * rocket._csys

    if tails:
        tail, position = tails[0]
        body_length = aft(position.z) - nose.length
        if not math.isclose(tail.top_radius, rocket.radius):
            raise ValueError("DATCOM tail top radius must match the rocket radius")
        if tail.bottom_radius <= 0 or tail.bottom_radius > tail.top_radius:
            raise ValueError("DATCOM tail must be a boattail with positive base radius")
        aft_length = tail.length
        aft_diameter = 2 * tail.bottom_radius
        if "body_length" in options and not math.isclose(
            float(options["body_length"]), body_length
        ):
            raise ValueError(
                "DATCOM body_length conflicts with the existing tail position"
            )
    else:
        body_length = float(
            options.get("body_length", aft(rocket.nozzle_position) - nose.length)
        )
        aft_length, aft_diameter = 0, 2 * rocket.radius
    if not math.isfinite(body_length) or body_length <= 0:
        raise ValueError(
            "DATCOM cylindrical body length must be positive; add a tail or set body_length"
        )
    if aft_length < 0 or not math.isfinite(aft_length):
        raise ValueError("DATCOM tail length must be finite and nonnegative")
    total_length = nose.length + body_length + aft_length

    def body_radius(x):
        if x < nose.length:
            return float(np.interp(x, nose.shape_vec[0], nose.shape_vec[1]))
        if aft_length > 0 and x > nose.length + body_length:
            if hasattr(tail, "radius_at"):
                return float(tail.radius_at(x - nose.length - body_length))
            fraction = (x - nose.length - body_length) / aft_length
            return rocket.radius + fraction * (aft_diameter / 2 - rocket.radius)
        return rocket.radius

    fin_items, naca_cards, deflections = [], [], []
    all_fins = [
        (s, p)
        for s, p, _ in surfaces
        if isinstance(s, (TrapezoidalFins, DatcomFreeFormFins))
    ]
    all_fins.sort(key=lambda pair: aft(pair[1].z))
    if len(all_fins) > 4:
        raise ValueError("This DATCOM integration supports at most four fin sets")
    previous_end, previous_span = None, None
    for index, (fins, position) in enumerate(all_fins, 1):
        is_datcom = isinstance(fins, (DatcomFins, DatcomFreeFormFins))
        spans = np.asarray(
            fins.datcom_span_stations if is_datcom else [0, fins.span], float
        )
        chords = np.asarray(
            fins.datcom_chords if is_datcom else [fins.root_chord, fins.tip_chord],
            float,
        )
        edges = np.asarray(
            fins.datcom_leading_edges if is_datcom else [0, fins.sweep_length], float
        )
        xle = aft(position.z) + edges
        if (
            len(spans) > 10
            or len(spans) < 2
            or np.any(np.diff(spans) <= 0)
            or not np.isfinite(np.concatenate((spans, chords, edges))).all()
            or np.any(chords[:-1] <= 0)
            or chords[-1] < 0
        ):
            raise ValueError("Invalid DATCOM fin span stations/chords/leading edges")
        # Automatic attachment samples the body at the exposed root chord
        # centre (Missile DATCOM guide, p. 13). A fin may trail past the base;
        # requiring every trailing edge inside the body rejects valid cases.
        root_center = float(xle[0] + chords[0] / 2)
        if np.any(xle < 0) or not 0 <= root_center <= total_length + 1e-9:
            raise ValueError(
                "DATCOM fin root chord centers must lie within the body's axial extent; leading edges must be aft of the nose tip"
            )
        if not isinstance(fins.n, (int, np.integer)) or not 1 <= fins.n <= 8:
            raise ValueError("DATCOM requires 1..8 panels per fin set")
        start, finish = float(xle.min()), float((xle + chords).max())
        if previous_end is not None:
            gap = start - previous_end
            if gap < -1e-9:
                raise ValueError("DATCOM fin-set planforms must not overlap")
            if gap < 4 * previous_span:
                warnings.warn(
                    "DATCOM fin sets are separated by less than four exposed semispans; "
                    "the upstream vortex-roll-up approximation may be inaccurate.",
                    UserWarning,
                    stacklevel=2,
                )
        previous_end, previous_span = finish, float(spans[-1])
        attachment = body_radius(float(xle[0] + chords[0] / 2))
        automatic = getattr(fins, "auto_body_radius", True)
        if not automatic and not math.isclose(fins.rocket_radius, attachment):
            raise ValueError(
                "DATCOM fin root radius must match the body at root mid-chord"
            )
        if is_datcom:
            fins._datcom_attachment_radius = attachment
            fins._datcom_drawing_body = {
                "has_tail": bool(tails),
                "end": tip.z - rocket._csys * total_length,
            }
        span_input = spans if automatic else spans + fins.rocket_radius
        count = len(spans)
        section = getattr(fins, "section_type", "HEX")
        if edge_radius is not None and section == "NACA":
            raise ValueError("NACA sections define their own leading-edge radius")
        radius = (
            getattr(fins, "leading_edge_radius", 0)
            if edge_radius is None
            else edge_radius
        )
        items = [
            ("XLE", xle),
            ("NPANEL", fins.n),
            (
                "PHIF",
                _station_values(
                    getattr(fins, "panel_angles", np.arange(fins.n) * 360 / fins.n),
                    "panel_angles",
                    fins.n,
                ),
            ),
            ("GAM", _station_values(getattr(fins, "dihedral", 0), "dihedral", fins.n)),
            ("SSPAN", span_input),
            ("CHORD", chords),
            ("SECTYP", section),
        ]
        if section in ("HEX", "ARC"):
            items.extend(
                [
                    (
                        "ZUPPER",
                        _station_values(
                            getattr(fins, "upper_thickness_to_chord", 0.01),
                            "upper thickness",
                            count,
                        ),
                    ),
                    (
                        "ZLOWER",
                        _station_values(
                            getattr(fins, "lower_thickness_to_chord", 0.01),
                            "lower thickness",
                            count,
                        ),
                    ),
                    (
                        "LER",
                        _station_values(
                            radius,
                            "leading_edge_radius",
                            count,
                        ),
                    ),
                ]
            )
            if section == "HEX":
                items.extend(
                    [
                        (
                            "LMAXU",
                            _station_values(
                                getattr(fins, "max_thickness_position", 0.25),
                                "max_thickness_position",
                                count,
                            ),
                        ),
                        (
                            "LFLATU",
                            _station_values(
                                getattr(fins, "flat_length", 0.5), "flat_length", count
                            ),
                        ),
                        (
                            "LMAXL",
                            _station_values(
                                getattr(fins, "lower_max_thickness_position", 0.25),
                                "lower_max_thickness_position",
                                count,
                            ),
                        ),
                        (
                            "LFLATL",
                            _station_values(
                                getattr(fins, "lower_flat_length", 0.5),
                                "lower_flat_length",
                                count,
                            ),
                        ),
                    ]
                )
        elif section == "NACA":
            naca_cards.append(f"NACA-{index}-{fins.naca}")
        elif section == "USER":
            coordinates = fins.airfoil_coordinates
            items.append(("XCORD", coordinates["x"]))
            for key, variable in (
                ("upper", "YUPPER"),
                ("lower", "YLOWER"),
                ("mean", "MEAN"),
                ("thickness", "THICK"),
            ):
                if key in coordinates:
                    items.append((variable, coordinates[key]))
            items.append(
                (
                    "LER",
                    _station_values(radius, "leading_edge_radius", count),
                )
            )
        else:
            raise ValueError("Unsupported DATCOM fin section")
        fin_items.append(_namelist(f"FINSET{index}", items))
        incidence = fins.datcom_incidence if is_datcom else (-fins.cant_angle,) * fins.n
        deflections.append(
            (f"DELTA{index}", _station_values(incidence, "incidence", fins.n))
        )

    filled_cg = float(rocket.center_of_mass(rocket.motor.burn_start_time))
    empty_cg = float(rocket.center_of_dry_mass_position)
    if not all(
        math.isfinite(cg) and 0 <= aft(cg) <= total_length
        for cg in (filled_cg, empty_cg)
    ):
        raise ValueError("DATCOM filled and empty CG must lie inside the body")
    needs_beta_grid = any(
        s.n < 3
        or not np.allclose(
            getattr(s, "panel_angles", np.arange(s.n) * 360 / s.n),
            np.arange(s.n) * 360 / s.n,
        )
        or np.any(getattr(s, "dihedral", 0))
        or np.any(getattr(s, "datcom_incidence", -s.cant_angle))
        for s, _ in all_fins
    )
    beta_values = np.asarray(
        options.get("beta", [-5, 0, 5] if needs_beta_grid else [0]), float
    )
    if (
        beta_values.ndim != 1
        or not 1 <= len(beta_values) <= 20
        or not np.isfinite(beta_values).all()
        or np.any(np.diff(beta_values) <= 0)
        or np.count_nonzero(beta_values == 0) != 1
        or (len(beta_values) > 1 and not beta_values[0] < 0 < beta_values[-1])
    ):
        raise ValueError(
            "DATCOM beta must be [0] or increasing signed angles including zero"
        )
    exit_diameter = None
    if powered_base_drag == "nozzle":
        try:
            raw_exit = options.get(
                "nozzle_exit_diameter", 2 * rocket.motor.nozzle_radius
            )
            if np.ndim(raw_exit) != 0:
                raise ValueError
            exit_diameter = float(raw_exit)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "Powered nozzle diameter must be a finite positive scalar"
            ) from exc
        if not math.isfinite(exit_diameter) or not 0 < exit_diameter <= aft_diameter:
            raise ValueError(
                "Powered nozzle diameter must be positive and fit inside the aft body"
            )
    elif powered_base_drag == "full":
        exit_diameter = 0
    decks = {}
    reynolds = options.get("reynolds")
    if reynolds is not None:
        if not isinstance(reynolds, dict):
            reynolds = {"power_on": reynolds, "power_off": reynolds}
        if set(reynolds) != {"power_on", "power_off"}:
            raise ValueError("reynolds schedules need power_on and power_off")
        ren = {}
        for powered, label in ((True, "power_on"), (False, "power_off")):
            values = np.asarray(reynolds[label], float)
            if values.ndim == 0:
                values = np.full(len(mach), float(values))
            if (
                values.shape != mach.shape
                or not np.isfinite(values).all()
                or np.any(values <= 0)
            ):
                raise ValueError(
                    "DATCOM reynolds needs positive Reynolds numbers per meter, one per Mach"
                )
            ren[powered] = values
    for powered, cg in ((True, filled_cg), (False, empty_cg)):
        body = [
            ("TNOSE", nose_types[kind]),
            ("LNOSE", nose.length),
            ("DNOSE", 2 * rocket.radius),
            ("LCENTR", body_length),
            ("DCENTR", 2 * rocket.radius),
            ("BASE", ".FALSE."),
        ]
        if kind == "powerseries":
            body.append(("POWER", nose.power))
        if aft_length > 0 and not math.isclose(aft_diameter, 2 * rocket.radius):
            body.extend(
                [
                    (
                        "TAFT",
                        "OGIVE"
                        if getattr(tail, "kind", "conical") == "ogive"
                        else "CONE",
                    ),
                    ("LAFT", aft_length),
                    ("DAFT", aft_diameter),
                ]
            )
        if not powered:
            body.append(("DEXIT", 0))
        elif exit_diameter is not None:
            body.append(("DEXIT", exit_diameter))
        reference_items = [
            ("XCG", aft(cg)),
            ("SREF", rocket.area),
            ("LREF", 2 * rocket.radius),
        ]
        if calibration["roughness_height"] > 0:
            # RHR is independent of DIM; the local 3/99 binary disagrees with
            # the 5/97 guide's metric ROUGH note. k = 3 RHR microinches.
            reference_items.append(
                ("RHR", calibration["roughness_height"] / (3e-6 * 0.0254))
            )
        cases = []
        for beta in beta_values:
            cases.append(
                "\n".join(
                    [
                        "DIM M",
                        _namelist(
                            "FLTCON",
                            [
                                ("NMACH", len(mach)),
                                ("MACH", mach),
                                ("ALT", altitude[powered])
                                if reynolds is None
                                else ("REN", ren[powered]),
                                ("NALPHA", len(alpha)),
                                ("ALPHA", alpha),
                                ("BETA", beta),
                            ],
                        ),
                        _namelist("REFQ", reference_items),
                        _namelist("AXIBOD", body),
                        *fin_items,
                        _namelist("DEFLCT", deflections),
                        *naca_cards,
                        "DAMP",
                        "NEXT CASE",
                        "",
                    ]
                )
            )
        decks[powered] = "".join(cases)
    return decks, {True: filled_cg, False: empty_cg}, timeout, first.datcom_path


def run_datcom(path, deck, timeout=60):
    """Copy and run a user-supplied native executable; always remove all files."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"DATCOM executable not found: {source}")
    with tempfile.TemporaryDirectory(prefix="rocketpy-datcom-") as directory:
        executable = Path(directory, source.name)
        shutil.copy2(source, executable)
        Path(directory, "for005.dat").write_text(deck, encoding="ascii")
        try:
            result = subprocess.run(
                [str(executable)],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=timeout,
                errors="replace",
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"DATCOM exceeded its {timeout:g} second timeout"
            ) from exc
        except OSError as exc:
            raise RuntimeError(
                "Cannot execute DATCOM; provide a native executable with execute permission"
            ) from exc
        if result.returncode:
            raise RuntimeError(
                f"DATCOM failed ({result.returncode}): {result.stderr[-2000:]}"
            )
        output = Path(directory, "for006.dat")
        if not output.is_file():
            raise RuntimeError("DATCOM did not produce for006.dat")
        # Some DATCOM builds return success even after rejecting an input deck.
        raw = output.read_text(errors="replace")
        if re.search(r"\*\*\s*ERROR\s*\*\*|\bFATAL\b", raw, re.IGNORECASE):
            raise RuntimeError("DATCOM reported an input error in for006.dat")
        parsed = parse_datcom_output(directory)
        # A successful exit is insufficient: old DATCOM builds can fall back
        # to defaults after malformed inputs. Verify the requested table grid.
        for variable, actual in (("MACH", parsed[4]), ("ALPHA", parsed[5])):
            requested = re.search(rf"\b{variable}\s*=\s*([\dEe+.,\s-]+)", deck)
            if requested:
                expected = np.array(
                    [float(v) for v in re.split(r"[\s,]+", requested[1].strip(" ,\n"))]
                )
                if actual.shape != expected.shape or not np.allclose(
                    actual, expected, atol=0.005, rtol=0
                ):
                    raise RuntimeError(f"DATCOM returned a different {variable} grid")
        requested_beta = [
            float(match[1]) for match in re.finditer(r"\bBETA\s*=\s*([-+\d.eE]+)", deck)
        ]
        if requested_beta:
            expected_beta = np.unique(requested_beta)
            if parsed.beta.shape != expected_beta.shape or not np.allclose(
                parsed.beta, expected_beta, atol=0.005, rtol=0
            ):
                raise RuntimeError("DATCOM returned a different BETA grid")
        return parsed


class DatcomSurface(GenericSurface):
    """Whole-vehicle body-axis loads, including all eleven DAMP derivatives.

    Native force channels map to RocketPy (CY, -CN, -CA), and moments to
    (CM, CLN, CLL). DATCOM geometry coordinates increase aft from the nose;
    RocketPy body z points toward the nose. Static lateral derivatives and
    DAMP derivatives in for006 are per degree. Rates are nondimensionalized by
    diameter/(2 V). Sideslip uses full tables when available, otherwise a
    linear expansion about beta=0.

    Loads are evaluated at each case's reference CG and transferred to the dry
    CG. Flight explicitly transfers their physical torque to the instantaneous
    wet CG. No fuel-fraction blending of powered and coast base drag is used.
    """

    def __init__(self, rocket, tables, references):
        super().__init__(rocket.area, 2 * rocket.radius, {}, name="DATCOM vehicle")
        self.tables = tables
        self.references = references
        self.dry_reference = float(rocket.center_of_dry_mass_position)
        self.csys = rocket._csys
        self.burn_start = rocket.motor.burn_start_time
        self.burn_out = rocket.motor.burn_out_time
        self._interpolators = {}
        self.outside_alpha_envelope = False
        self.outside_mach_envelope = False
        self.outside_beta_envelope = False
        for powered, table in tables.items():
            mach, alpha = table[4], np.radians(table[5])
            zero_alpha_index(alpha)
            if not alpha[0] < 0 < alpha[-1]:
                raise ValueError("DATCOM needs signed alpha tables")

            def channels(t):
                arrays = [t.body[k] for k in BODY_KEYS]
                arrays += [t.derivatives[k] for k in LATERAL_KEYS]
                arrays += [t.damping[k] for k in DAMPING_KEYS]
                arrays += [t[1]]
                values = np.stack(arrays, axis=-1)
                if (
                    values.shape != (len(mach), len(alpha), 21)
                    or not np.isfinite(values).all()
                ):
                    raise ValueError("DATCOM tables are incomplete or nonfinite")
                return values

            if table.sideslip:
                values = np.stack(
                    [channels(table.sideslip[float(beta)]) for beta in table.beta],
                    axis=2,
                )
                grid = (mach, alpha, np.radians(table.beta))
            else:
                values = channels(table)
                grid = (mach, alpha)
            self._interpolators[powered] = RegularGridInterpolator(grid, values)

    def is_powered(self, time):
        return self.burn_start <= time < self.burn_out

    def reference_offset(self, time):
        """Case reference relative to dry CG, in RocketPy body axes (meters)."""
        return Vector(
            [
                0,
                0,
                (self.references[self.is_powered(time)] - self.dry_reference)
                * self.csys,
            ]
        )

    def coefficients(self, powered, mach, alpha, beta=0):
        """Interpolate Mach and signed incidence; clamp outside the input grid."""
        interp = self._interpolators[powered]
        m, a = interp.grid[:2]
        point = [np.clip(mach, m[0], m[-1]), np.clip(alpha, a[0], a[-1])]
        if len(interp.grid) == 3:
            b = interp.grid[2]
            point.append(np.clip(beta, b[0], b[-1]))
        return interp(point)[0]

    def drag_coefficient(self, powered, alpha, beta, mach, reynolds, q, r, p):
        """DATCOM stability-axis drag for reporting and plots."""
        return float(self.coefficients(powered, mach, alpha, beta)[-1])

    def compute_forces_and_moments_at_time(
        self,
        time,
        alpha_dot,
        stream_velocity,
        stream_speed,
        stream_mach,
        rho,
        cp,
        omega,
        reynolds,
    ):
        """Return body loads at dry CG for a deterministic RHS evaluation."""
        powered = self.is_powered(time)
        velocity = -stream_velocity
        speed = abs(velocity)
        if speed < 1e-12:
            return (0.0,) * 6
        alpha = math.atan2(velocity.y, abs(velocity.z))
        beta = math.atan2(velocity.x, abs(velocity.z))
        interp = self._interpolators[powered]
        m, a = interp.grid[:2]
        physical_alpha = math.atan2(velocity.y, velocity.z)
        self.outside_alpha_envelope |= not a[0] <= physical_alpha <= a[-1]
        self.outside_mach_envelope |= not m[0] <= stream_mach <= m[-1]
        if len(interp.grid) == 3:
            b = interp.grid[2]
            self.outside_beta_envelope |= not b[0] <= beta <= b[-1]
        values = self.coefficients(powered, stream_mach, alpha, beta)
        cn, cm, ca, cy, cln, cll, cyb, clnb, cllb = values[:9]
        beta_deg = math.degrees(beta)
        if len(interp.grid) == 2:
            cy += cyb * beta_deg
            cln += clnb * beta_deg
            cll += cllb * beta_deg
        if speed >= 1:
            cnq, cmq, caq, cnad, cmad, cyr, clnr, cllr, cyp, clnp, cllp = values[9:20]
            factor = math.degrees(self.reference_length / (2 * speed))
            q, r, p = (rate * factor for rate in omega)
            adot = alpha_dot * factor
            cn += cnq * q + cnad * adot
            cm += cmq * q + cmad * adot
            ca += caq * q
            cy += cyr * r + cyp * p
            cln += clnr * r + clnp * p
            cll += cllr * r + cllp * p
        q_area = 0.5 * rho * speed**2 * self.reference_area
        # CA is already a body-axis coefficient. Applying another cosine here
        # would project it twice. Only reverse its direction for tail-first flow.
        axial_direction = math.copysign(1.0, velocity.z)
        forces = Vector([cy, -cn, -ca * axial_direction]) * q_area
        moments = Vector([cm, cln, cll]) * (q_area * self.reference_length)
        moments += cp ^ forces
        return (*forces, *moments)

    def compute_forces_and_moments(self, *args, **kwargs):
        raise RuntimeError(
            "DATCOM loads require Flight to supply time and incidence rate"
        )

    def small_angle_cp(self, mach):
        """Coast small-angle CP in the user's axial frame, in meters."""
        table = self.tables[False]
        zero = zero_alpha_index(table[5])
        cna = np.interp(mach, table[4], table.derivatives["CNA"][:, zero])
        cma = np.interp(mach, table[4], table.derivatives["CMA"][:, zero])
        if abs(cna) < 1e-12:
            return float("nan")
        return self.references[False] + self.csys * self.reference_length * cma / cna
