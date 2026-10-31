"""Plot measured apogees and the two supported aerodynamic comparisons."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
GREEN, RED, NEUTRAL = "#25834a", "#c93b3b", "#7d8995"
NAMES = {
    "bella_lui": "Bella Lui",
    "camoes": "Camões",
    "juno3": "Juno III",
    "ndrt_2020": "NDRT 2020",
    "prometheus_2022": "Prometheus 2022",
}


def comparison_data(directory=OUT):
    """Keep all rockets; compare errors only where both predictions exist."""
    directory = Path(directory)
    metrics = pd.read_csv(directory / "rocket_metrics.csv")
    frame = metrics.loc[
        metrics.metric == "Apogee AGL",
        ["rocket", "measured", "rocketpy", "datcom", "condition", "availability"],
    ].copy()
    if frame.rocket.duplicated().any():
        raise ValueError("Duplicate apogee records")
    for key in ("rocketpy", "datcom"):
        frame[f"{key}_error_pct"] = (
            100 * (frame[key] - frame.measured) / frame.measured.abs()
        )
    valid_measurement = np.isfinite(frame.measured) & (frame.measured > 0)
    paired = valid_measurement & np.isfinite(frame.rocketpy) & np.isfinite(frame.datcom)
    frame["closer_prediction"] = "unavailable comparison"
    a = frame.rocketpy_error_pct.abs()
    b = frame.datcom_error_pct.abs()
    tie = paired & np.isclose(a, b, atol=1e-9, rtol=0)
    frame.loc[paired & ~tie & (a < b), "closer_prediction"] = "RocketPy + supplied CD"
    frame.loc[paired & ~tie & (b < a), "closer_prediction"] = "DATCOM"
    frame.loc[tie, "closer_prediction"] = "tie"
    frame["paired"] = paired
    frame["rocketpy_archived"] = frame.condition.fillna("").str.contains(
        "archived", case=False
    )
    # The first-flight Juno source explicitly identifies CFD scaling; Cavour
    # names RASAero II and Valetudo identifies RASAero in its input filenames.
    frame["drag_provenance"] = frame.rocket.map(
        {
            "juno3": "RASAero II curve scaled using a CFD value (source cell 14)",
            "cavour": "RASAero II (source cell 10)",
            "valetudo": "RASAero (source input filenames)",
        }
    ).fillna("Supplied CD; CFD provenance not established")
    return frame.reset_index(drop=True)


def error_percentiles(frame):
    """Empirical error quantiles across the same paired rockets, not flight CIs."""
    paired = frame.loc[frame.paired]
    records = []
    for key, label in [("rocketpy", "RocketPy + supplied CD"), ("datcom", "DATCOM")]:
        errors = paired[f"{key}_error_pct"].abs()
        records.append(
            {
                "model": label,
                "paired_rockets": len(errors),
                **{
                    f"P{int(q * 100)}_absolute_error_pct": errors.quantile(q)
                    for q in [0.5, 0.75, 0.9, 0.95]
                },
            }
        )
    return pd.DataFrame(records)


def plot_apogees(frame):
    """Each panel uses its own metre scale and the recorded apogee reference."""
    columns = 4
    rows = int(np.ceil(len(frame) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(18, 3.8 * rows), squeeze=False)
    for ax, (_, row) in zip(axes.flat, frame.iterrows()):
        heights = np.array([row.rocketpy, row.datcom], dtype=float)
        finite = heights[np.isfinite(heights)]
        maximum = max([row.measured, *finite])
        ax.set_ylim(0, maximum * 1.40)
        ax.axhline(row.measured, color="#242424", ls="--", lw=1.5, zorder=3)
        for index, (key, label) in enumerate(
            [("rocketpy", "RocketPy + supplied CD"), ("datcom", "DATCOM")]
        ):
            value = heights[index]
            if not np.isfinite(value):
                ax.text(
                    index,
                    0.09,
                    "Unavailable",
                    ha="center",
                    va="bottom",
                    color=NEUTRAL,
                    transform=ax.get_xaxis_transform(),
                    fontsize=10,
                )
                continue
            winner = row.closer_prediction
            color = (
                NEUTRAL
                if not row.paired or winner == "tie"
                else GREEN
                if winner == label
                else RED
            )
            ax.bar(
                index,
                value,
                width=0.57,
                color=color,
                edgecolor="#333333",
                linewidth=0.6,
                hatch="//" if key == "datcom" else "",
                zorder=2,
            )
            error = row[f"{key}_error_pct"]
            ax.text(
                index,
                value + maximum * 0.035,
                f"{value:,.0f} m\n{error:+.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1),
                zorder=4,
            )
        title = NAMES.get(row.rocket, row.rocket.replace("_", " ").title())
        ax.set_title(
            f"{title}  ·  recorded {row.measured:,.0f} m",
            fontsize=11,
            fontweight="bold",
            pad=9,
        )
        labels = ["RocketPy\nsupplied CD", "DATCOM"]
        if row.rocketpy_archived:
            labels[0] += "\n(archived)"
        ax.set_xticks([0, 1], labels)
        ax.set_xlim(-0.65, 1.65)
        ax.set_ylabel("Apogee AGL (m)")
        ax.yaxis.grid(True, alpha=0.16, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.flat[len(frame) :]:
        ax.set_visible(False)
    fig.suptitle(
        "Recorded rockets: apogee with supplied CD versus DATCOM",
        fontsize=19,
        fontweight="bold",
        y=0.995,
    )
    fig.legend(
        handles=[
            Patch(facecolor=GREEN, label="Closer to recorded apogee"),
            Patch(facecolor=RED, label="Farther from recorded apogee"),
            Patch(facecolor=NEUTRAL, label="No paired result / tie"),
            Line2D([0], [0], color="#242424", linestyle="--", label="Recorded apogee"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.974),
        ncol=4,
        frameon=False,
        fontsize=10,
    )
    fig.text(
        0.5,
        0.008,
        "Labels: altitude and signed error 100 × (prediction − recorded) / |recorded|. Green/red rank only available paired predictions.\n"
        "Each panel has its own vertical scale. Supplied CD is not universally CFD-derived. Archived outputs and missing DATCOM cases remain identified.",
        ha="center",
        va="bottom",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.945), h_pad=2.3)
    return fig


def save_apogees(figure, directory=OUT):
    """Export raster/vector charts with whitespace-clean SVG markup."""
    directory = Path(directory)
    figure.savefig(directory / "apogee_comparison.png", dpi=170, bbox_inches="tight")
    svg = directory / "apogee_comparison.svg"
    figure.savefig(svg, bbox_inches="tight")
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )


def main():
    frame = comparison_data()
    frame.to_csv(OUT / "apogee_comparison.csv", index=False)
    error_percentiles(frame).to_csv(OUT / "apogee_error_percentiles.csv", index=False)
    figure = plot_apogees(frame)
    save_apogees(figure)
    plt.close(figure)


if __name__ == "__main__":
    main()
