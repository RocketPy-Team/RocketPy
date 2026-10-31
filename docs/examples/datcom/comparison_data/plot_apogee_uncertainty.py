"""Compare recorded apogees with supplied-CD RocketPy and DATCOM scenarios."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from plot_apogee_comparison import GREEN, NAMES, NEUTRAL, RED, comparison_data

OUT = Path(__file__).resolve().parent


def uncertainty_comparison(directory=OUT):
    """Join complete scenario ascents to matching nominal/measured comparisons.

    All altitudes are meters AGL. Missing scenario runs are not zero-width
    uncertainties. Reject stale nominal results instead of mixing models.
    """
    directory = Path(directory)
    nominal = comparison_data(directory).set_index("rocket")
    stability = pd.read_csv(directory / "rocket_metrics.csv")
    stability = stability.loc[stability.metric == "Initial static margin"].set_index(
        "rocket"
    )
    scenarios = pd.read_csv(directory / "uncertainty_sensitivity.csv")
    if scenarios.duplicated(["rocket", "case"]).any():
        raise ValueError("Duplicate DATCOM sensitivity cases")
    rows = []
    for name, group in scenarios.groupby("rocket", sort=False):
        if set(group.case) != {"nominal", "upper", "lower"}:
            raise ValueError(f"{name}: nominal, upper and lower runs are required")
        if name not in nominal.index:
            raise ValueError(f"{name}: no recorded-rocket comparison")
        row = nominal.loc[name].to_dict()
        values = group.set_index("case")
        heights = values.apogee_agl.to_numpy(float)
        if not np.isfinite(heights).all() or not np.all(heights > 0):
            raise ValueError(f"{name}: incomplete or invalid scenario apogees")
        if not np.isclose(
            values.loc["nominal", "apogee_agl"], row["datcom"], rtol=0, atol=0.001
        ):
            raise ValueError(
                f"{name}: sensitivity and comparison nominal apogees differ"
            )
        if not row["paired"]:
            raise ValueError(
                f"{name}: both predictions and a recorded apogee are required"
            )
        low, high = float(heights.min()), float(heights.max())
        margins = values.initial_stability_margin.to_numpy(float)
        rows.append(
            dict(
                rocket=name,
                **row,
                datcom_scenario_low_m=low,
                datcom_scenario_high_m=high,
                datcom_sensitivity_minus_m=row["datcom"] - low,
                datcom_sensitivity_plus_m=high - row["datcom"],
                datcom_low_error_pct=100 * (low / row["measured"] - 1),
                datcom_high_error_pct=100 * (high / row["measured"] - 1),
                recorded_within_datcom_scenarios=low <= row["measured"] <= high,
                nominal_outside_paper_pct=100
                * values.loc["nominal", "outside_paper_fraction"],
                rocketpy_stability=stability.loc[name, "rocketpy"],
                recorded_stability=stability.loc[name, "measured"],
                datcom_stability=values.loc["nominal", "initial_stability_margin"],
                datcom_stability_low=float(margins.min()),
                datcom_stability_high=float(margins.max()),
            )
        )
    if not rows:
        raise ValueError("No complete DATCOM sensitivity comparisons")
    return pd.DataFrame(rows)


def plot_uncertainty_comparison(frame):
    """Draw asymmetric scenario whiskers on nominal DATCOM predictions."""
    columns = min(3, len(frame))
    rows = int(np.ceil(len(frame) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(5.2 * columns, 5.4 * rows), squeeze=False
    )
    for axis, (_, row) in zip(axes.flat, frame.iterrows()):
        maximum = max(row.measured, row.rocketpy, row.datcom_scenario_high_m)
        for index, (key, label) in enumerate(
            (("rocketpy", "RocketPy + supplied CD"), ("datcom", "DATCOM"))
        ):
            color = (
                NEUTRAL
                if row.closer_prediction == "tie"
                else GREEN
                if row.closer_prediction == label
                else RED
            )
            axis.bar(
                index,
                row[key],
                width=0.52,
                color=color,
                edgecolor="#333333",
                linewidth=0.7,
                hatch="//" if key == "datcom" else "",
                zorder=2,
            )
            height = row.datcom_scenario_high_m if key == "datcom" else row[key]
            axis.text(
                index,
                height + 0.025 * maximum,
                f"{row[key]:,.0f} m\n{row[f'{key}_error_pct']:+.1f}%",
                ha="center",
                va="bottom",
                fontsize=11,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.95, pad=1),
                zorder=5,
            )
        axis.errorbar(
            1,
            row.datcom,
            yerr=np.array(
                [[row.datcom_sensitivity_minus_m], [row.datcom_sensitivity_plus_m]]
            ),
            fmt="none",
            ecolor="#202020",
            elinewidth=2,
            capsize=7,
            capthick=2,
            zorder=4,
        )
        axis.axhline(
            row.measured, color="#202020", linestyle="--", linewidth=1.5, zorder=3
        )
        name = NAMES.get(row.rocket, row.rocket.replace("_", " ").title())
        axis.set_title(name, fontsize=15, fontweight="bold", pad=12)
        inside = "inside" if row.recorded_within_datcom_scenarios else "outside"
        axis.text(
            0.5,
            0.97,
            f"Recorded: {row.measured:,.0f} m\nDATCOM scenarios: {row.datcom_scenario_low_m:,.0f}–{row.datcom_scenario_high_m:,.0f} m\nRecorded apogee {inside} scenario range",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            color="#333333",
        )
        axis.set_xticks(
            [0, 1], ["RocketPy\nsupplied CD", "DATCOM\nwith sensitivity range"]
        )
        axis.set_xlim(-0.7, 1.7)
        axis.set_ylim(0, 1.52 * maximum)
        axis.set_ylabel("Apogee above launch site (m)")
        axis.yaxis.grid(True, alpha=0.2, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
    for axis in axes.flat[len(frame) :]:
        axis.set_visible(False)
    figure.suptitle(
        "RocketPy, DATCOM sensitivity and recorded apogees",
        fontsize=18,
        fontweight="bold",
        y=0.99,
    )
    figure.legend(
        handles=[
            Patch(facecolor=GREEN, label="Closer nominal prediction"),
            Patch(facecolor=RED, label="Farther nominal prediction"),
            Line2D([], [], color="#202020", linestyle="--", label="Recorded apogee"),
            Line2D(
                [],
                [],
                color="#202020",
                marker="_",
                markersize=10,
                label="DATCOM scenario range",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.974 if rows > 1 else 0.925),
        ncol=4,
        frameon=False,
        fontsize=10,
    )
    figure.text(
        0.5,
        0.015,
        "Whiskers: CD at least ±20% (larger Wu errors inside its envelope), nose-to-CP distance ±2%; not confidence intervals.\n"
        "Labels show nominal altitude and signed error versus the recorded apogee. Each panel has its own vertical scale.\n"
        f"{len(frame)} rockets shown; "
        f"{frame.nominal_outside_paper_pct.min():.0f}–{frame.nominal_outside_paper_pct.max():.0f}% "
        "of their nominal ascent time lies outside Wu's test envelope.",
        ha="center",
        va="bottom",
        fontsize=10,
    )
    figure.tight_layout(
        rect=(0, 0.04 if rows > 1 else 0.12, 1, 0.94 if rows > 1 else 0.85), h_pad=2
    )
    return figure


def plot_stability_comparison(frame):
    """Show the margin resulting from 2% CP-distance shifts, in calibers."""
    figure, axis = plt.subplots(figsize=(max(12, 1.3 * len(frame)), 6))
    position = np.arange(len(frame))
    axis.bar(
        position - 0.18,
        frame.rocketpy_stability,
        width=0.34,
        color="#3979b7",
        label="RocketPy + supplied CD",
    )
    axis.bar(
        position + 0.18,
        frame.datcom_stability,
        width=0.34,
        color="#dc8a28",
        label="DATCOM",
        hatch="//",
        yerr=np.array(
            [
                frame.datcom_stability - frame.datcom_stability_low,
                frame.datcom_stability_high - frame.datcom_stability,
            ]
        ),
        capsize=5,
        error_kw={"elinewidth": 1.5},
    )
    observed = np.isfinite(frame.recorded_stability)
    if observed.any():
        axis.scatter(
            position[observed],
            frame.loc[observed, "recorded_stability"],
            color="#222222",
            marker="D",
            label="Recorded stability",
            zorder=4,
        )
    axis.set_xticks(
        position,
        [NAMES.get(name, name.replace("_", " ").title()) for name in frame.rocket],
        rotation=25,
        ha="right",
    )
    axis.set_ylabel("Loaded-ignition stability margin (calibers)")
    axis.set_title(
        "Stability sensitivity from ±2% nose-to-CP distance",
        fontsize=16,
        fontweight="bold",
    )
    axis.legend(loc="upper left", frameon=False)
    axis.yaxis.grid(True, alpha=0.2)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    missing = (
        "No recorded stability margins are available for these rockets."
        if not observed.any()
        else "Missing recorded stability values are omitted; their test conditions may differ."
    )
    figure.text(
        0.5,
        0.015,
        missing
        + "\nWhiskers are scenario ranges, not confidence intervals. A 2% CP-distance shift is not a 2% margin change.",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.10, 1, 1))
    return figure


def save_uncertainty_comparison(
    figure, directory=OUT, stem="apogee_uncertainty_comparison"
):
    """Export both PNG and SVG for the notebook and standalone viewing."""
    directory = Path(directory)
    figure.savefig(directory / f"{stem}.png", dpi=180, bbox_inches="tight")
    svg = directory / f"{stem}.svg"
    figure.savefig(svg, bbox_inches="tight")
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )


def main():
    frame = uncertainty_comparison()
    frame.to_csv(OUT / "apogee_uncertainty_comparison.csv", index=False)
    figure = plot_uncertainty_comparison(frame)
    save_uncertainty_comparison(figure)
    plt.close(figure)
    figure = plot_stability_comparison(frame)
    save_uncertainty_comparison(figure, stem="stability_uncertainty_comparison")
    plt.close(figure)
    focus = frame[frame.rocket.isin(["genesis", "astra", "andromeda"])]
    if len(focus):
        figure = plot_uncertainty_comparison(focus)
        save_uncertainty_comparison(figure, stem="apogee_uncertainty_comparison_focus")
        plt.close(figure)


if __name__ == "__main__":
    main()
