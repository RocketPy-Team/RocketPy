"""Test script to visualize different background map options at Kennedy Space Center.

This script creates simulated Monte Carlo data for Kennedy Space Center
and tests all background map options, saving the results as images.
"""

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import pytest

from rocketpy.simulation import MonteCarlo
from rocketpy.plots.monte_carlo_plots import _MonteCarloPlots

plt.rcParams.update({"figure.max_open_warning": 0})

pytest.importorskip(
    "contextily", reason="This test requires contextily to be installed"
)


class MockMonteCarlo(MonteCarlo):
    """Create a mock class to test the method without running a real simulation.

    This class creates a MonteCarlo object with simulated results data for testing
    background map options. Only includes the minimal attributes needed for plotting.
    """

    def __init__(self, env, filename="kennedy_space_center_test"):
        """Initialize MockMonteCarlo with simulated data.

        Parameters
        ----------
        env : rocketpy.Environment
            Environment object to use for the Monte Carlo simulation.
            Must have latitude and longitude attributes.
        filename : str, optional
            Filename for the MonteCarlo object. Defaults to "kennedy_space_center_test".
        """

        # pylint: disable=super-init-not-called
        # Create a simple environment object with only latitude and longitude
        # needed for background map fetching
        class SimpleEnvironment:
            """Simple environment object with only latitude and longitude."""

            def __init__(self, env):
                self.latitude = env.latitude
                self.longitude = env.longitude

        # Set attributes needed for plotting background maps
        self.filename = filename
        self.environment = SimpleEnvironment(env)
        self.plots = _MonteCarloPlots(self)

        # Generate simulated Monte Carlo results
        # Simulate 50 data points with realistic dispersion
        np.random.seed(42)  # For reproducibility

        # Simulate apogee points (scattered around origin with some dispersion)
        n_points = 50
        apogee_x = np.random.normal(0, 500, n_points)  # meters, std dev 500m
        apogee_y = np.random.normal(0, 500, n_points)  # meters, std dev 500m

        # Simulate impact points (further from origin, more dispersion)
        impact_x = np.random.normal(
            2000, 1000, n_points
        )  # meters, mean 2km, std dev 1km
        impact_y = np.random.normal(
            1500, 1000, n_points
        )  # meters, mean 1.5km, std dev 1km

        # Set simulated results (only what's needed for ellipses plot)
        self.results = {
            "apogee_x": apogee_x.tolist(),
            "apogee_y": apogee_y.tolist(),
            "x_impact": impact_x.tolist(),
            "y_impact": impact_y.tolist(),
        }


@pytest.mark.parametrize(
    "background,name",
    [
        (None, "no_background"),
        ("satellite", "satellite"),
        ("street", "street"),
        ("terrain", "terrain"),
        ("CartoDB.Positron", "cartodb_positron"),
    ],
)
def test_all_background_options(example_kennedy_env, background, name):
    """Test all background map options and save images.

    This function tests:
    - None (no background)
    - satellite
    - street
    - terrain
    - custom provider (CartoDB.Positron)

    Parameters
    ----------
    example_kennedy_env : rocketpy.Environment
        Environment fixture for Kennedy Space Center.
    background : str or None
        Background map option to test.
    name : str
        Name identifier for the background option (used in filename).
    """
    output_dir = "kennedy_background_tests"
    os.makedirs(output_dir, exist_ok=True)

    try:
        monte_carlo = MockMonteCarlo(env=example_kennedy_env)

        # Temporarily change filename to save with desired name
        original_filename = monte_carlo.filename
        monte_carlo.filename = os.path.join(output_dir, f"kennedy_{name}")

        try:
            monte_carlo.plots.ellipses(
                background=background,
                xlim=(-5000, 5000),
                ylim=(-5000, 5000),
                save=True,
            )

            expected_file = f"{monte_carlo.filename}.png"
            if not os.path.exists(expected_file):
                raise FileNotFoundError(
                    f"Expected file {expected_file} was not created after plotting."
                )
        finally:
            monte_carlo.filename = original_filename
    finally:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
