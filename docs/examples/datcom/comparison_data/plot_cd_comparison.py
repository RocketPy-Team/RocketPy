"""Zero-incidence CD curves and engineering bands from saved comparison data."""

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from plot_apogee_comparison import NAMES

OUT = Path(__file__).resolve().parent
ROOT = next(parent for parent in OUT.parents if (parent / "rocketpy").is_dir())
sys.path.insert(0, str(ROOT))
from rocketpy.rocket.datcom_uncertainty import Wu2019Errors  # noqa: E402

BLUE, ORANGE, GREY = "#3979b7", "#d17a19", "#ededed"


def cd_comparison_data(directory=OUT, *, asteria=False):
    """Attach the active zero-AoA drag policy to saved powered/coast curves.

    CD is dimensionless and all curves use alpha=beta=0. Native table limits
    are retained; values outside those limits are unavailable, not extrapolated.
    The apogee-fitted constant is inferred using the ordinary RocketPy model.
    """
    directory = Path(directory)
    source = Wu2019Errors()
    if asteria:
        names = ["asteria"]
        metrics = None
        status = None
    else:
        status = pd.read_csv(directory / "rockets/results.csv").set_index("rocket")
        names = status.index[status.datcom_agl_m.notna()].tolist()
        metrics = pd.read_csv(directory / "rocket_metrics.csv")
    records = []
    for name in names:
        stem = directory / "asteria" if asteria else directory / "rockets" / name
        curves = pd.read_csv(stem.with_name(stem.name + "_cd.csv"))
        provenance = json.loads(
            stem.with_name(stem.name + "_provenance.json").read_text()
        )
        grid = np.asarray(
            provenance["atmosphere"]["mach"] if asteria else provenance["mach"], float
        )
        if asteria:
            phases = provenance["atmosphere"]["coverage"]["phases"]
            peak = max(p["mach_max"] for p in phases.values())
            fitted = float("nan")
        else:
            peak = max(
                status.loc[name, "ordinary_max_mach"],
                status.loc[name, "datcom_max_mach"],
            )
            fit = metrics.loc[
                (metrics.rocket == name)
                & (metrics.metric == "Apogee-fitted constant CD"),
                "inferred_cd",
            ]
            if len(fit) != 1:
                raise ValueError(f"{name}: expected one apogee-fit record")
            fitted = float(fit.iloc[0])
        mach = curves.mach.to_numpy(float)
        if not np.isfinite(mach).all() or not np.all(np.diff(mach) > 0):
            raise ValueError(f"{name}: Mach samples must be finite and increasing")
        # Same rule as DatcomUncertaintySurface.drag_fraction at alpha=0.
        fraction = np.array(
            [
                max(0.2, source.fractions(m, 0)[0])
                if source.mach[0] <= m <= source.mach[-1]
                else 0.2
                for m in mach
            ]
        )
        valid = (mach >= grid[0] - 1e-9) & (mach <= grid[-1] + 1e-9)
        for phase in ("powered", "coast"):
            rocketpy = curves[f"rocketpy_{phase}_cd"].to_numpy(float)
            datcom = curves[f"datcom_{phase}_cd"].to_numpy(float)
            if (
                not np.isfinite(rocketpy).any()
                or np.isinf(rocketpy).any()
                or not np.isfinite(datcom[valid]).all()
            ):
                raise ValueError(f"{name}: nonfinite {phase} curve")
            datcom = np.where(valid, datcom, np.nan)
            records.append(
                pd.DataFrame(
                    dict(
                        rocket=name,
                        phase=phase,
                        mach=mach,
                        alpha_deg=0,
                        beta_deg=0,
                        rocketpy_cd=rocketpy,
                        rocketpy_available=np.isfinite(rocketpy),
                        datcom_cd=datcom,
                        datcom_low_cd=datcom * (1 - fraction),
                        datcom_high_cd=datcom * (1 + fraction),
                        drag_excursion_pct=100 * fraction,
                        apogee_fitted_constant_cd=fitted,
                        reference_peak_mach=peak,
                        above_reference_peak=mach > peak,
                        within_native_grid=valid,
                    )
                )
            )
    if not records:
        raise ValueError("No saved CD comparisons")
    return pd.concat(records, ignore_index=True)


def plot_cd_comparison(frame, *, phase=None, rockets=None):
    """Plot supplied curves and nominal DATCOM with engineering bands."""
    names = list(frame.rocket.unique()) if rockets is None else list(rockets)
    phases = [phase] if phase is not None else ["powered", "coast"]
    panels = [(name, state) for state in phases for name in names]
    columns = min(3, len(panels))
    rows = int(np.ceil(len(panels) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(5.2 * columns, 4.2 * rows + 1), squeeze=False
    )
    has_fit = False
    for axis, (name, state) in zip(axes.flat, panels):
        values = frame[(frame.rocket == name) & (frame.phase == state)]
        if values.empty:
            raise ValueError(f"{name}: no {state} curve")
        mach = values.mach.to_numpy()
        peak = float(values.reference_peak_mach.iloc[0])
        if peak < mach[-1]:
            axis.axvspan(max(peak, mach[0]), mach[-1], color=GREY, zorder=0)
            axis.axvline(peak, color="#999999", linestyle=":", linewidth=1)
        axis.fill_between(
            mach,
            values.datcom_low_cd.to_numpy(),
            values.datcom_high_cd.to_numpy(),
            color=ORANGE,
            alpha=0.22,
            linewidth=0,
            zorder=1,
        )
        axis.plot(mach, values.rocketpy_cd, color=BLUE, linewidth=2, zorder=3)
        axis.plot(mach, values.datcom_cd, color=ORANGE, linewidth=2, zorder=3)
        fitted = float(values.apogee_fitted_constant_cd.iloc[0])
        if np.isfinite(fitted):
            axis.axhline(
                fitted, color="#242424", linestyle=":", linewidth=1.5, zorder=2
            )
            has_fit = True
        title = (
            "AST-HER / Asteria (has not flown)"
            if name == "asteria"
            else NAMES.get(name, name.replace("_", " ").title())
        )
        axis.set_title(f"{title} · {state}", fontsize=11, fontweight="bold", pad=10)
        axis.text(
            0.03,
            0.96,
            f"Reference peak Mach: {peak:.2f}",
            transform=axis.transAxes,
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=2),
        )
        maximum = np.nanmax(values[["rocketpy_cd", "datcom_high_cd"]].to_numpy())
        if np.isfinite(fitted):
            maximum = max(maximum, fitted)
        axis.set_ylim(0, maximum * 1.20)
        axis.set_xlim(mach[0], mach[-1])
        axis.set_xlabel("Mach number")
        axis.set_ylabel("Drag coefficient CD")
        axis.yaxis.grid(True, alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    for axis in axes.flat[len(panels) :]:
        axis.set_visible(False)
    title = "Zero-incidence CD comparison"
    if phase is not None:
        title += f" · {phase}"
    figure.suptitle(title, fontsize=18, fontweight="bold", y=0.995)
    handles = [
        Line2D([], [], color=BLUE, linewidth=2, label="RocketPy + supplied CD"),
        Line2D([], [], color=ORANGE, linewidth=2, label="Nominal DATCOM"),
        Patch(facecolor=ORANGE, alpha=0.22, label="DATCOM CD sensitivity range"),
        Patch(facecolor=GREY, label="Above reference peak Mach"),
    ]
    if has_fit:
        handles.append(
            Line2D(
                [],
                [],
                color="#242424",
                linestyle=":",
                label="Apogee-fitted constant CD (inferred)",
            )
        )
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965 if rows > 2 else 0.935),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    footer = "Alpha = beta = 0°. DATCOM band: ±20% minimum, larger in-envelope Wu errors; not a confidence interval.\n"
    footer += (
        "No measured CD supplied. Apogee-fitted CD is an inferred, nonunique RocketPy calibration.\n"
        if has_fit
        else "AST-HER has not flown; no measured or apogee-fitted CD is available.\n"
    )
    footer += "Curves retain saved atmosphere schedules. The ±2% CP shift does not directly alter zero-incidence CD."
    missing = frame.loc[~frame.rocketpy_available, "mach"].unique()
    if len(missing):
        samples = ", ".join(f"{mach:g}" for mach in sorted(missing))
        footer += f"\nMissing saved RocketPy CD samples remain gaps (Mach: {samples})."
    figure.text(
        0.5,
        0.01,
        footer,
        ha="center",
        va="bottom",
        fontsize=10,
    )
    figure.tight_layout(
        rect=(
            0,
            0.035 if rows > 2 else 0.14 if rows == 1 else 0.09,
            1,
            0.94 if rows > 2 else 0.85,
        ),
        h_pad=2,
    )
    return figure


def save_cd_comparison(figure, stem, directory=OUT):
    """Save PNG and whitespace-clean SVG versions of a CD comparison."""
    directory = Path(directory)
    figure.savefig(directory / f"{stem}.png", dpi=180, bbox_inches="tight")
    path = directory / f"{stem}.svg"
    figure.savefig(path, bbox_inches="tight")
    path.write_text(
        "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
    )


def main():
    frame = cd_comparison_data()
    frame.to_csv(OUT / "cd_uncertainty_comparison.csv", index=False)
    for phase in ("powered", "coast"):
        figure = plot_cd_comparison(frame, phase=phase)
        save_cd_comparison(figure, f"cd_{phase}_comparison")
        plt.close(figure)
    figure = plot_cd_comparison(frame, rockets=["genesis", "astra", "andromeda"])
    save_cd_comparison(figure, "cd_comparison_focus")
    plt.close(figure)
    frame = cd_comparison_data(asteria=True)
    frame.to_csv(OUT / "asteria_cd_uncertainty_comparison.csv", index=False)
    figure = plot_cd_comparison(frame)
    save_cd_comparison(figure, "asteria_cd_uncertainty_comparison")
    plt.close(figure)


if __name__ == "__main__":
    main()
