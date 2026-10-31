"""Fit native DATCOM inputs without replacing aerodynamic coefficients."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar

COEFFICIENTS = ("CD", "CL", "CA", "CN", "CY", "CM", "CLN", "CLL")
INPUTS = {"MACH", "ALPHA", "BETA", "POWER_ON", "REYNOLDS_PER_METER"}


def _measurements(data, power_on):
    """Normalize finite column data, preserving partial measured channels."""
    if isinstance(data, (str, Path)):
        table = np.genfromtxt(data, delimiter=",", names=True, encoding="utf-8-sig")
        if not table.dtype.names:
            raise ValueError("Wind-tunnel CSV requires a header")
        data = {key: np.atleast_1d(table[key]) for key in table.dtype.names}
    if not isinstance(data, Mapping):
        raise ValueError("Wind-tunnel data must be a CSV path or column mapping")
    columns = {}
    for key, value in data.items():
        name = str(key).strip().upper()
        if name in columns:
            raise ValueError(f"Duplicate wind-tunnel column: {name}")
        if name not in INPUTS | set(COEFFICIENTS):
            raise ValueError(f"Unsupported wind-tunnel column: {key}")
        array = np.asarray(value, float)
        if array.ndim != 1 or not len(array):
            raise ValueError(f"Wind-tunnel column {key} must be a nonempty vector")
        columns[name] = array
    if "MACH" not in columns or not set(columns) & set(COEFFICIENTS):
        raise ValueError("Wind-tunnel data needs mach and at least one coefficient")
    count = len(columns["MACH"])
    if any(len(value) != count for value in columns.values()):
        raise ValueError("Wind-tunnel columns must have equal lengths")
    if not np.isfinite(columns["MACH"]).all() or np.any(columns["MACH"] <= 0):
        raise ValueError("Wind-tunnel Mach numbers must be finite and positive")
    for key, default in (("ALPHA", 0), ("BETA", 0), ("POWER_ON", float(power_on))):
        columns.setdefault(key, np.full(count, default))
        if not np.isfinite(columns[key]).all():
            raise ValueError(f"Wind-tunnel {key} must be finite")
    if not np.isin(columns["POWER_ON"], [0, 1]).all():
        raise ValueError("Wind-tunnel power_on must be 0 or 1")
    if "REYNOLDS_PER_METER" in columns:
        ren = columns["REYNOLDS_PER_METER"]
        if not np.isfinite(ren).all() or np.any(ren <= 0):
            raise ValueError(
                "Tunnel Reynolds number per meter must be finite and positive"
            )
    measured = [key for key in COEFFICIENTS if key in columns]
    if any(np.isinf(columns[key]).any() for key in measured):
        raise ValueError("Measured coefficients cannot be infinite")
    if any(not np.isfinite(columns[key]).any() for key in measured):
        raise ValueError("Every coefficient column must contain a measurement")
    if not np.any(
        np.column_stack([np.isfinite(columns[key]) for key in measured]), axis=1
    ).all():
        raise ValueError("Every row needs at least one finite measured coefficient")
    return columns, measured


def _positive(value, name):
    try:
        number = float(value) if np.ndim(value) == 0 else float("nan")
    except (ValueError, TypeError, OverflowError):
        number = float("nan")
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return number


def _predictions(
    model, rocket, columns, measured, area, length, lateral_length, origin, drag_axis
):
    """Transform native coefficients to the tunnel's axes and references."""
    predicted = {key: [] for key in measured}
    for i, mach in enumerate(columns["MACH"]):
        state = bool(columns["POWER_ON"][i])
        alpha, beta = np.radians([columns["ALPHA"][i], columns["BETA"][i]])
        grid = model._interpolators[state].grid
        values = [mach, alpha] + ([beta] if len(grid) == 3 else [])
        if any(
            value < axis[0] - 1e-12 or value > axis[-1] + 1e-12
            for value, axis in zip(values, grid)
        ):
            raise ValueError(
                "Wind-tunnel measurements exceed the DATCOM Mach/angle envelope"
            )
        if len(grid) == 2 and beta != 0:
            raise ValueError("Nonzero tunnel beta requires a DATCOM beta table")
        cn, cm, ca, cy, cln, cll = model.coefficients(state, mach, alpha, beta)[:6]
        lift = cn * np.cos(alpha) - ca * np.sin(alpha)
        if drag_axis == "wind":
            # Unit flow = (tan(beta), tan(alpha), 1) / norm in RocketPy axes.
            cd = (ca + cn * np.tan(alpha) - cy * np.tan(beta)) / np.sqrt(
                1 + np.tan(alpha) ** 2 + np.tan(beta) ** 2
            )
        else:
            cd = ca * np.cos(alpha) + cn * np.sin(alpha)
        result = {"CD": cd, "CL": lift, "CA": ca, "CN": cn, "CY": cy}
        result = {key: value * rocket.area / area for key, value in result.items()}
        if any(key in measured for key in ("CM", "CLN", "CLL")):
            # M_new = M_old + (r_old - r_new) x F. In RocketPy axes
            # F=(CY,-CN,-CA), so the axial shift adds (d*CN,d*CY,0).
            shift = (
                (model.references[state] - origin) * rocket._csys / (2 * rocket.radius)
            )
            scale = rocket.area * (2 * rocket.radius) / area
            result.update(
                CM=(cm + shift * cn) * scale / length,
                CLN=(cln + shift * cy) * scale / lateral_length,
                CLL=cll * scale / lateral_length,
            )
        for key in measured:
            predicted[key].append(float(result[key]))
    return predicted


def _tunnel_conditions(columns, original_mach, model_scale=1):
    """Use measured Reynolds conditions only for fitting, never for Flight."""
    if "REYNOLDS_PER_METER" not in columns:
        return None
    mach = np.unique(columns["MACH"])
    if len(mach) == 1:
        other = np.asarray(original_mach)[np.asarray(original_mach) != mach[0]]
        mach = np.sort([mach[0], other[np.argmin(abs(other - mach[0]))]])
    if len(mach) > 20:
        raise ValueError("Tunnel Reynolds schedule supports at most 20 Mach values")
    schedules = {}
    for state, label in ((True, "power_on"), (False, "power_off")):
        mask = columns["POWER_ON"] == state
        # The unused state still needs a valid deck; it is not evaluated.
        if not mask.any():
            mask = np.ones(len(columns["MACH"]), dtype=bool)
        levels = np.unique(columns["MACH"][mask])
        reynolds = []
        for value in levels:
            values = columns["REYNOLDS_PER_METER"][mask & (columns["MACH"] == value)]
            if not np.allclose(values, values[0], rtol=1e-8, atol=0):
                raise ValueError(
                    "Tunnel data needs one Reynolds number per Mach and power state"
                )
            reynolds.append(values[0] * model_scale)
        schedules[label] = np.interp(mach, levels, reynolds).tolist()
    # ALT is unused with REN but must have the new grid's shape for validation.
    return {"mach": mach.tolist(), "reynolds": schedules, "altitude": 0}


def fit_datcom_calibration(
    rocket,
    data,
    *,
    reference_area=None,
    parameters=("roughness_height",),
    bounds=None,
    reference_length=None,
    moment_reference=None,
    lateral_reference_length=None,
    model_scale=1.0,
    power_on=False,
    weights=None,
    drag_axis="wind",
    max_evaluations=100,
):
    """Bounded coefficient fit; publish calibration only after successful runs."""
    from rocketpy.rocket.datcom import DatcomSurface, datcom_decks, run_datcom

    if not rocket._has_datcom_fins():
        raise ValueError("Wind-tunnel calibration requires DATCOM fins")
    if not isinstance(power_on, (bool, np.bool_)):
        raise ValueError("power_on must be boolean; row values may be 0 or 1")
    columns, measured = _measurements(data, power_on)
    model_scale = _positive(model_scale, "model_scale")
    measured_area = _positive(
        rocket.area * model_scale**2 if reference_area is None else reference_area,
        "reference_area",
    )
    area = measured_area / model_scale**2
    length = lateral_length = origin = None
    if set(measured) & {"CM", "CLN", "CLL"}:
        length = _positive(reference_length, "reference_length") / model_scale
        lateral_length = (
            _positive(
                reference_length
                if lateral_reference_length is None
                else lateral_reference_length,
                "lateral_reference_length",
            )
            / model_scale
        )
        if (
            moment_reference is None
            or np.ndim(moment_reference) != 0
            or not np.isfinite(moment_reference)
        ):
            raise ValueError(
                "Moment measurements require a finite moment_reference in rocket coordinates"
            )
        origin = float(moment_reference)
    if drag_axis not in ("wind", "stability"):
        raise ValueError("drag_axis must be 'wind' or 'stability'")
    if isinstance(parameters, str):
        parameters = (parameters,)
    parameters = tuple(parameters)
    if (
        not parameters
        or len(set(parameters)) != len(parameters)
        or set(parameters) - {"roughness_height", "leading_edge_radius"}
    ):
        raise ValueError(
            "Fit roughness_height and/or leading_edge_radius without duplicates"
        )
    if "leading_edge_radius" in parameters and any(
        getattr(s, "section_type", None) == "NACA"
        for s, _, _ in rocket.aerodynamic_surfaces
    ):
        raise ValueError("NACA sections define their own leading-edge radius")
    limits = dict(bounds or {})
    if set(limits) - set(parameters):
        raise ValueError("bounds contains parameters which are not being fitted")
    limits.setdefault("roughness_height", (0, 300e-6))
    for name in parameters:
        pair = np.asarray(limits.get(name, []), float)
        if (
            pair.shape != (2,)
            or not np.isfinite(pair).all()
            or not 0 <= pair[0] < pair[1]
        ):
            raise ValueError(
                f"{name} requires finite nonnegative (lower, upper) bounds"
            )
        limits[name] = pair
    weights = dict(weights or {})
    if set(weights) - set(measured):
        raise ValueError("weights names must correspond to measured coefficients")
    weight = {key: _positive(weights.get(key, 1), f"weight {key}") for key in measured}
    if (
        isinstance(max_evaluations, bool)
        or not isinstance(max_evaluations, (int, np.integer))
        or max_evaluations < 3
    ):
        raise ValueError("max_evaluations must be an integer of at least 3")
    observation_count = sum(int(np.isfinite(columns[key]).sum()) for key in measured)
    if observation_count < len(parameters):
        raise ValueError(
            "Not enough measured coefficients for the requested parameters"
        )
    original = rocket.datcom_calibration
    # Native geometry setup is evaluated without publishing any trial model or
    # changing the rocket's calibration, cached model, or reporting functions.
    first = next(
        s for s, _, _ in rocket.aerodynamic_surfaces if hasattr(s, "datcom_options")
    )
    original_mach = np.asarray(
        first.datcom_options.get("mach", [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.5, 2])
    )
    conditions = _tunnel_conditions(columns, original_mach, model_scale)
    states = sorted(set(bool(value) for value in columns["POWER_ON"]))
    cache = {}

    def evaluate(settings):
        key = (settings["roughness_height"], settings["leading_edge_radius"])
        if key in cache:
            return cache[key]
        if len(cache) >= max_evaluations:
            raise RuntimeError(
                "Wind-tunnel fit exceeded max_evaluations; calibration was not changed"
            )
        decks, references, timeout, path = datcom_decks(
            rocket, calibration=settings, flight_conditions=conditions
        )
        tables = {state: run_datcom(path, decks[state], timeout) for state in states}
        model = DatcomSurface(rocket, tables, references)
        predictions = _predictions(
            model,
            rocket,
            columns,
            measured,
            area,
            length,
            lateral_length,
            origin,
            drag_axis,
        )
        residuals = np.concatenate(
            [
                (np.asarray(predictions[key]) - columns[key])[np.isfinite(columns[key])]
                * np.sqrt(weight[key])
                for key in measured
            ]
        )
        cache[key] = (float(np.mean(residuals**2)), predictions)
        return cache[key]

    initial_cost, initial_prediction = evaluate(original)

    def settings_for(point):
        settings = dict(original)
        for name, value in zip(parameters, np.atleast_1d(point)):
            low, high = limits[name]
            settings[name] = float(low + value * (high - low))
        return settings

    def objective(point):
        return evaluate(settings_for(point))[0]

    if len(parameters) == 1:
        result = minimize_scalar(
            objective,
            bounds=(0, 1),
            method="bounded",
            options={"xatol": 1e-4, "maxiter": max_evaluations - 3},
        )
        best_point = [result.x]
    else:
        start = [
            np.clip(
                ((original[name] or 0) - limits[name][0]) / np.ptp(limits[name]), 0, 1
            )
            for name in parameters
        ]
        result = minimize(
            objective,
            start,
            method="Powell",
            bounds=[(0, 1)] * len(parameters),
            options={"maxfev": max_evaluations - 1, "xtol": 1e-4, "ftol": 1e-6},
        )
        best_point = result.x
    if not result.success:
        raise RuntimeError(
            f"Wind-tunnel fit failed; calibration was not changed: {result.message}"
        )
    # Explicitly compare endpoints: rounded native coefficients can hide a
    # boundary optimum from an interior bounded search.
    candidates = [settings_for(best_point)]
    if len(parameters) == 1:
        candidates += [settings_for([0]), settings_for([1])]
    fitted = min(candidates, key=lambda settings: evaluate(settings)[0])
    final_cost, prediction = evaluate(fitted)
    # Retain an existing calibration if it is within bounds and already better.
    if (
        all(original[name] is not None for name in parameters)
        and all(
            limits[name][0] <= (original[name] or 0) <= limits[name][1]
            for name in parameters
        )
        and initial_cost < final_cost
    ):
        fitted, final_cost, prediction = original, initial_cost, initial_prediction
    residual_rows = []
    rmse = {}
    for key in measured:
        mask = np.isfinite(columns[key])
        delta = np.asarray(prediction[key])[mask] - columns[key][mask]
        rmse[key] = dict(
            initial=float(
                np.sqrt(
                    np.mean(
                        (np.asarray(initial_prediction[key])[mask] - columns[key][mask])
                        ** 2
                    )
                )
            ),
            fitted=float(np.sqrt(np.mean(delta**2))),
        )
        for i in np.flatnonzero(mask):
            residual_rows.append(
                dict(
                    row=int(i),
                    mach=float(columns["MACH"][i]),
                    alpha_deg=float(columns["ALPHA"][i]),
                    beta_deg=float(columns["BETA"][i]),
                    power_on=bool(columns["POWER_ON"][i]),
                    coefficient=key,
                    observed=float(columns[key][i]),
                    predicted=prediction[key][i],
                    residual=float(prediction[key][i] - columns[key][i]),
                )
            )
    report = dict(
        success=True,
        improved=bool(final_cost < initial_cost - 1e-16),
        observation_count=observation_count,
        calibration=dict(fitted),
        initial_calibration=original,
        parameters=list(parameters),
        bounds={key: limits[key].tolist() for key in parameters},
        reference_area=area,
        measured_reference_area=measured_area,
        model_scale=model_scale,
        reference_length=length,
        lateral_reference_length=lateral_length,
        moment_reference=origin,
        drag_axis=drag_axis,
        reynolds_source="measurements"
        if conditions is not None
        else "rocket DATCOM options",
        weighted_rmse_initial=float(np.sqrt(initial_cost)),
        weighted_rmse_fitted=float(np.sqrt(final_cost)),
        rmse=rmse,
        residuals=residual_rows,
        native_evaluations=len(cache),
        native_runs=len(cache) * len(states),
        at_bound={
            name: bool(
                np.isclose(
                    fitted[name],
                    limits[name][0],
                    rtol=0,
                    atol=np.ptp(limits[name]) * 1e-4,
                )
                or np.isclose(
                    fitted[name],
                    limits[name][1],
                    rtol=0,
                    atol=np.ptp(limits[name]) * 1e-4,
                )
            )
            for name in parameters
        },
    )
    rocket.add_datcom_calibration(**fitted)
    return report
