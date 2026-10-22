"""Native-size DATCOM flight grids, independent of ordinary RocketPy models."""

import numpy as np

MAX_MACH_POINTS = 20


def adaptive_mach_grid(peak_mach, margin=0.15):
    """Twenty positive Mach points with fixed anchors and transonic resolution.

    Mach output has two decimal places in the supported native build, so points
    are separated by at least 0.01. Zero speed uses the 0.01 endpoint. The upper
    bound includes 15% headroom plus 0.02 Mach, and is at least 1.2. Spare points
    bisect gaps, prioritizing the reached flight regime over unreached speeds.
    """
    if not np.isfinite(peak_mach) or peak_mach < 0:
        raise ValueError("Reference peak Mach must be finite and nonnegative")
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("Mach margin must be finite and nonnegative")
    upper = np.ceil(max(1.2, peak_mach * (1 + margin) + 0.02) * 100) / 100
    reached = max(0.01, np.ceil(peak_mach * 100) / 100)
    selected = {
        0.01,
        0.1,
        0.8,
        0.9,
        0.95,
        0.98,
        1.0,
        1.02,
        1.05,
        1.1,
        1.2,
        upper,
        reached,
    }
    while len(selected) < MAX_MACH_POINTS:
        values = np.array(sorted(selected))
        gaps = np.diff(values)
        middle = (values[:-1] + values[1:]) / 2
        priorities = gaps * np.where(middle <= reached, 4, 1)
        priorities[gaps < 0.0199] = -1
        index = int(np.argmax(priorities))
        selected.add(float(np.round(middle[index], 2)))
    return np.array(sorted(selected))


def adaptive_alpha_grid(alpha_degrees):
    """Signed 19-point grid, retaining ±20° minimum and dense small angles.

    Extend symmetrically beyond observed, dynamically meaningful incidence by
    15% plus 2 degrees. Tail-first flow is outside the current forward-flight
    DATCOM load mapping and is rejected instead of silently folded or clipped.
    """
    alpha = np.asarray(alpha_degrees, float)
    if not alpha.size or not np.isfinite(alpha).all():
        raise ValueError("Reference flight needs finite loaded alpha samples")
    peak = float(np.max(np.abs(alpha)))
    extent = max(20, np.ceil((peak * 1.15 + 2) * 100) / 100)
    if extent >= 90:
        raise ValueError(
            "Adaptive DATCOM alpha envelope reaches tail-first flow (90 degrees)"
        )
    positive = np.round(
        extent * np.array([0.0025, 0.005, 0.0125, 0.025, 0.05, 0.1, 0.25, 0.5, 1]), 2
    )
    return np.concatenate((-positive[::-1], [0.0], positive))
