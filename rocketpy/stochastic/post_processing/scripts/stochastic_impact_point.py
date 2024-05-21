from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from rocketpy.stochastic.post_processing.stochastic_cache import \
    SimulationCache

# 1-3 sigma
lower_percentiles = [0.16, 0.03, 0.003]
upper_percentiles = [0.84, 0.97, 0.997]


# Define function to calculate eigen values
def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


def compute_impact(file_name, batch_path, save, show):
    cache = SimulationCache(
        file_name,
        batch_path,
    )
    x_impact = cache.read_outputs('x_impact')
    y_impact = cache.read_outputs('y_impact')

    x_mean_impact = np.nanmean(x_impact, axis=0)
    y_mean_impact = np.nanmean(y_impact, axis=0)

    # Calculate error ellipses for impact
    impact_cov = np.cov(x_impact.flatten(), y_impact.flatten())
    impact_vals, impactVecs = eigsorted(impact_cov)
    impact_theta = np.degrees(np.arctan2(*impactVecs[:, 0][::-1]))
    impact_w, impactH = 2 * np.sqrt(impact_vals)

    fig, ax = plt.subplots()
    ax.scatter(x_impact, y_impact, c='blue')
    ax.scatter(
        x_mean_impact,
        y_mean_impact,
        marker='x',
        c='red',
        label='Mean Impact Point',
    )

    # Draw error ellipses for impact
    impact_ellipses = []
    for j in [1, 2, 3]:
        impactEll = Ellipse(
            xy=(np.mean(x_impact), np.mean(y_impact)),
            width=impact_w * j,
            height=impactH * j,
            angle=impact_theta,
            color="black",
        )
        impactEll.set_facecolor((0, 0, 1, 0.2))
        impact_ellipses.append(impactEll)
        ax.add_artist(impactEll)

    ax.set_xlabel('X Impact Point [m]')
    ax.set_ylabel('Y Impact Point [m]')
    ax.set_title('Impact Point Distribution')
    ax.set_aspect('equal')
    ax.legend()
    ax.grid()

    if save:
        plt.savefig(batch_path / 'mean_impact_distribution.png')

    if show:
        plt.show()
    plt.show()


def run(file_name, batch_path, save, show):
    compute_impact(file_name, batch_path, save, show)


if __name__ == '__main__':
    # import easygui

    batch_path = Path("mc_simulations/")
    file_name = 'monte_carlo_class_example'
    run(file_name, batch_path, save=True, show=True)
