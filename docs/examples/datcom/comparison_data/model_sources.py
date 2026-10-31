"""Read all source notebooks, their end-of-notebook flight evidence and models.

Only trusted local repository notebooks are supported. Construction stops at
first Flight; plots, installs, online weather and later recovery runs are not
executed. Source notebooks are never rewritten.
"""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

import nbformat
import numpy as np
import pandas as pd

ROOT = next(
    p
    for p in Path(__file__).resolve().parents
    if (p / "rocketpy/rocket/datcom.py").is_file()
)
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location(
    "datcom_example_helpers",
    Path(__file__).with_name("model_geometry.py"),
)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


class SourceUnavailable(ValueError):
    """A faithful offline comparison needs additional data or geometry support."""


def _freeze_mach_drag(function):
    """Capture a solver source before legacy Mach-only aliases are reassigned."""
    from rocketpy import Function

    return Function(lambda mach: function(0, 0, mach, 0, 0, 0, 0))


def _drag_assignment(node, namespace):
    """Find legacy drag writes in a trusted example, without changing equations."""
    targets = (
        node.targets
        if isinstance(node, ast.Assign)
        else [node.target]
        if isinstance(node, ast.AugAssign)
        else []
    )
    writes = []
    for target in targets:
        if isinstance(target, ast.Attribute) and target.attr in {
            "power_off_drag",
            "power_on_drag",
        }:
            owner = eval(
                compile(ast.Expression(target.value), "<drag owner>", "eval"), namespace
            )
            # The initial aliases call self.*_7d dynamically. Freeze them before
            # an expression such as alias *= factor, or rebinding *_7d would
            # make that expression recursively call itself.
            for name in ("power_off_drag", "power_on_drag"):
                if getattr(owner, name) is getattr(owner, name + "_by_mach"):
                    setattr(
                        owner, name, _freeze_mach_drag(getattr(owner, name + "_7d"))
                    )
            writes.append((owner, target.attr))
    return writes


def _tree(source):
    return ast.parse(
        "\n".join(
            line
            for line in source.splitlines()
            if not line.lstrip().startswith(("%", "!"))
        )
    )


def _literal(node, scope):
    if isinstance(node, ast.Name):
        return scope[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _literal(node.left, scope) + _literal(node.right, scope)
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_literal(item, scope) for item in node.elts]
    return ast.literal_eval(node)


def flight_reference(path):
    """Read telemetry AGL and all reported/saved references with cell provenance."""
    path = Path(path).resolve()
    notebook = nbformat.read(path, 4)
    scope = {}
    recorded = None
    reported_cell = None
    saved = []
    measured_summaries = []
    telemetry = None
    notes = []
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "markdown":
            plain = re.sub(r"[`*$]", "", cell.source)
            match = re.search(
                r"(?:official\s+)?recorded\s+apogee(?:\s+altitude)?(?:\s+of)?\s*:?\s*([\d,.]+)\s*m",
                plain,
                re.I,
            )
            if not match:
                match = re.search(
                    r"Measured Altitude[^\n]*?or\s*([\d,.]+)\s*m", plain, re.I
                )
            if match:
                recorded = float(match[1].replace(",", ""))
                reported_cell = index
            continue
        for output in cell.get("outputs", []):
            text = output.get("text", "") or output.get("data", {}).get(
                "text/plain", ""
            )
            for match in re.finditer(
                r"Apogee Altitude:\s*[-+\d.eE]+\s*m\s*\(ASL\)\s*\|\s*([-+\d.eE]+)\s*m\s*\(AGL\)",
                text,
            ):
                saved.append(dict(cell=index, agl_m=float(match[1])))
            if "apogee" in cell.source.lower():
                for match in re.finditer(
                    r"(?:Actual apogee|Real apogee|Real data|Actual):\s*([\d.]+)\s*m",
                    text,
                    re.I,
                ):
                    measured_summaries.append(dict(cell=index, agl_m=float(match[1])))
        for node in _tree(cell.source).body:
            if not isinstance(node, ast.Assign):
                continue
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                try:
                    scope[node.targets[0].id] = _literal(node.value, scope)
                except (ValueError, KeyError, TypeError):
                    pass
            call = node.value
            if telemetry is not None or not isinstance(call, ast.Call):
                continue
            if isinstance(call.func, ast.Name) and call.func.id == "FlightDataImporter":
                options = {kw.arg: kw.value for kw in call.keywords}
                try:
                    mapping = _literal(options["columns_map"], scope)
                    time_col = next(
                        key for key, value in mapping.items() if value == "time"
                    )
                    altitude_col = next(
                        key for key, value in mapping.items() if value == "altitude"
                    )
                    paths = _literal(options["paths"], scope)
                except (KeyError, ValueError, StopIteration):
                    continue
                for filename in [paths] if isinstance(paths, str) else paths:
                    filename = ROOT / filename.replace("../../", "")
                    if not filename.exists():
                        notes.append(
                            f"Referenced telemetry file is absent: {filename.relative_to(ROOT)} (cell {index})."
                        )
                        continue
                    table = pd.read_csv(filename)
                    if time_col not in table or altitude_col not in table:
                        continue
                    telemetry = dict(
                        path=str(filename.relative_to(ROOT)),
                        cell=index,
                        time=table[time_col].to_numpy(float),
                        altitude=table[altitude_col].to_numpy(float),
                        conversion="metres AGL, notebook columns_map",
                    )
                    break
            elif (
                path.stem in ("bella_lui_flight_sim", "ndrt_2020_flight_sim")
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "loadtxt"
            ):
                filename = ROOT / _literal(call.args[0], scope).replace("../../", "")
                options = {kw.arg: _literal(kw.value, scope) for kw in call.keywords}
                values = np.loadtxt(filename, **options)
                if path.stem.startswith("bella_lui"):
                    values = values[:573]
                    altitude = values[:, 1]
                    conversion = "metres AGL, first 573 rows as used in source cell 35"
                else:
                    altitude = values[:, 1] * 0.3048
                    conversion = "feet AGL multiplied by exactly 0.3048; source uses division by 3.281"
                    notes.append(
                        "NDRT source uses approximate ft conversion /3.281; comparison uses exact 0.3048 m/ft."
                    )
                telemetry = dict(
                    path=str(filename.relative_to(ROOT)),
                    cell=index,
                    time=values[:, 0],
                    altitude=altitude,
                    conversion=conversion,
                )
    name = path.name.removesuffix("_flight_sim.ipynb")
    if name == "valetudo":
        recorded = float(scope["actual_data"]["apogee"])
        reported_cell = next(
            i
            for i, c in enumerate(notebook.cells)
            if c.cell_type == "code" and "actual_data =" in c.source
        )
        notes.append(
            "Valetudo flight recordings were lost; only recovered scalar apogee/drift values remain."
        )
    if telemetry is not None:
        mask = np.isfinite(telemetry["time"]) & np.isfinite(telemetry["altitude"])
        time, alt = telemetry["time"][mask], telemetry["altitude"][mask]
        order = np.argsort(time, kind="stable")
        time, alt = time[order], alt[order]
        time, keep = np.unique(time, return_index=True)
        alt = alt[keep]
        telemetry["time"], telemetry["altitude"] = time, alt
        telemetry["sha256"] = hashlib.sha256(
            (ROOT / telemetry["path"]).read_bytes()
        ).hexdigest()
        raw_max = float(alt.max())
        clean = np.ones(len(time), dtype=bool)
        previous = 0
        for i in range(1, len(time)):
            if abs(alt[i] - alt[previous]) / (time[i] - time[previous]) > 2000:
                clean[i] = False
            else:
                previous = i
        telemetry["qa_valid"] = clean
        telemetry["raw_max_agl_m"] = raw_max
        telemetry["qa_max_agl_m"] = float(alt[clean].max())
        telemetry["qa_rejected_count"] = int((~clean).sum())
        if (~clean).any():
            notes.append(
                f"Telemetry QA flags {int((~clean).sum())} samples with altitude jumps above 2000 m/s; raw maximum {raw_max:.3f} m is retained separately."
            )
        actual = recorded if recorded is not None else telemetry["qa_max_agl_m"]
        reference_kind = (
            "notebook reported apogee"
            if recorded is not None
            else "telemetry AGL maximum (QA)"
        )
        if recorded is not None and abs(raw_max - recorded) > 0.01:
            notes.append(
                f"Telemetry maximum {raw_max:.3f} m differs from notebook reported value {recorded:.3f} m; both retained."
            )
    else:
        actual = measured_summaries[0]["agl_m"] if measured_summaries else recorded
        reference_kind = (
            "saved measured-apogee summary"
            if measured_summaries
            else "notebook reported apogee"
        )
    if name == "astra":
        notes.append(
            "Source fin trailing edge overhangs the nozzle-inferred body base by 1.216 mm. Native DATCOM accepts this; the integration now validates root-chord centre attachment instead of rejecting all overhang. Geometry is unchanged."
        )
    if name == "defiance":
        notes.append(
            "Source fin root centre is z=-0.025 m, aft of the tail base z=0.0403 m; native root attachment is outside the declared body."
        )
    if name == "ndrt_2020":
        notes.append(
            "Source transition is internal at z=1.2..1.327 m with fins near z=3.05 m. The current DATCOM builder represents only an aft tail, not the required downstream smaller cylinder."
        )
    if name == "hedy":
        notes.append(
            "Hedy end comparison uses simulated ASL 5406.918 m against measured AGL 5231.53 m; saved simulated AGL is 5246.918 m, a 160 m datum discrepancy."
        )
    if name == "prometheus_2022":
        notes.append(
            "Prometheus prose says June 2022; source environment and local reanalysis are June 2023. Both reruns preserve source conditions."
        )
    if name == "valkyrie":
        notes.append(
            "Valkyrie prose says 2025 but source environment says 2026 and fetches live Windy weather; no archived profile is supplied."
        )
    if name == "juno3":
        notes.append(
            "Source fin airfoil is an empirical lift curve (cell 10), not a supplied section contour. The generic 2% HEX DATCOM assumption does not reproduce that curve; source-drag substitution therefore still has a lateral-model discrepancy."
        )
    if name == "lince":
        notes.append(
            "Lince has three saved trajectories. Compare first/pre-deployment ascent only; retain all saved apogees separately."
        )
    return dict(
        rocket=name,
        source_notebook=str(path.relative_to(ROOT)),
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        actual_agl_m=actual,
        reference_kind=reference_kind,
        reported_agl_m=recorded,
        reported_cell=reported_cell,
        saved_first_agl_m=saved[0]["agl_m"] if saved else None,
        saved_apogees=saved,
        measured_summaries=measured_summaries,
        telemetry=telemetry,
        notes=notes,
    )


class Construction(ast.NodeTransformer):
    """Strip display-only statements and reject online model inputs."""

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id in {
            "SolidMotor",
            "HybridMotor",
            "GenericMotor",
            "LiquidMotor",
        }:
            for keyword in node.keywords:
                if keyword.arg == "thrust_source":
                    keyword.value = ast.Call(
                        func=ast.Name(id="_example_thrust", ctx=ast.Load()),
                        args=[keyword.value],
                        keywords=[],
                    )
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in {"draw", "info", "all_info"}:
                return (
                    node.func.value
                    if isinstance(node.func.value, ast.Call)
                    else ast.copy_location(ast.Constant(None), node)
                )
            if node.func.attr == "set_atmospheric_model":
                kind = next(
                    (
                        kw.value.value
                        for kw in node.keywords
                        if kw.arg == "type" and isinstance(kw.value, ast.Constant)
                    ),
                    "",
                )
                if kind.lower() in {"windy", "forecast", "ensemble"}:
                    raise SourceUnavailable(
                        "Source requests live weather with no archived profile; not replaced by invented weather."
                    )
            if node.func.attr == "set_elevation" and any(
                isinstance(a, ast.Constant) and a.value == "Open-Elevation"
                for a in node.args
            ):
                raise SourceUnavailable(
                    "Source requests online elevation; an explicit archived input is needed."
                )
        return node

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call):
            if any(
                part in ast.unparse(node.value.func) for part in (".plots.", ".prints.")
            ):
                return None
            value = node.value
            while isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
                value = value.func.value
            root = value
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in {"plt", "display", "print"}:
                return None
            if isinstance(node.value.func, ast.Attribute) and node.value.func.attr in {
                "compare_plots",
                "show",
                "savefig",
            }:
                return None
        return self.generic_visit(node)


def build_model(path):
    """Execute only trusted pre-Flight construction; preserve physical inputs."""
    path = Path(path).resolve()
    notebook = nbformat.read(path, 4)
    adaptations = []

    def thrust_source(value):
        if isinstance(value, str) and value.endswith(".eng"):
            from rocketpy.motors.motor import Motor

            _, _, points = Motor.import_eng(value)
            if len(points) > 1 and points[1][0] == 0:
                adaptations.append(
                    "Read original .eng numeric samples directly: source already has t=0; avoid the importer's extra duplicate zero-time row. No measured sample or motor equation changed."
                )
                return np.asarray(points[1:], float)
        return value

    namespace = {"__name__": "__datcom_source__", "_example_thrust": thrust_source}
    executed = []
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        tree = _tree(cell.source)
        for node in tree.body:
            call = node.value if isinstance(node, (ast.Assign, ast.Expr)) else None
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "Flight"
            ):
                options = {
                    kw.arg: eval(
                        compile(ast.Expression(kw.value), str(path), "eval"), namespace
                    )
                    for kw in call.keywords
                }
                rocket, env = options.pop("rocket"), options.pop("environment")
                # Both comparisons stop at apogee. Recovery is irrelevant to
                # supported passive ascent cases and is deliberately omitted.
                rocket.parachutes = []
                rocket._example_adaptations = adaptations
                options.update(terminate_on_apogee=True)
                options.setdefault("max_time", 180)
                options["max_time_step"] = 0.1
                return rocket, env, options, executed
            module = ast.Module(body=[node], type_ignores=[])
            module = h.LocalPaths(ROOT).visit(module)
            module = Construction().visit(module)
            ast.fix_missing_locations(module)
            drag_writes = _drag_assignment(node, namespace)
            exec(compile(module, str(path), "exec"), namespace)
            for owner, name in drag_writes:
                setattr(
                    owner,
                    name + "_7d",
                    owner._Rocket__process_drag_input(getattr(owner, name), name),
                )
                adaptations.append(
                    f"Source cell {index}: preserve the notebook's post-construction {name} update in the seven-input solver coefficient."
                )
            executed.append(index)
    raise SourceUnavailable("No first Flight construction found")
