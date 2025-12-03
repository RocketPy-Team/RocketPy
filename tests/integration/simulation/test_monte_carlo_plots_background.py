"""Test script to visualize different background map options at Kennedy Space Center.

This script creates simulated Monte Carlo data for Kennedy Space Center
and tests all background map options, saving the results as images.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import MagicMock

from rocketpy.simulation import MonteCarlo
from rocketpy.stochastic import StochasticEnvironment

plt.rcParams.update({"figure.max_open_warning": 0})


def create_simulated_monte_carlo_data(env):
    """Create a MonteCarlo object with simulated results data.

    Parameters
    ----------
    env : rocketpy.Environment
        Environment object to use for the Monte Carlo simulation.

    Returns
    -------
    MonteCarlo
        MonteCarlo object with simulated apogee and impact data.
    """

    # Create stochastic environment (no randomization for simplicity)
    stochastic_env = StochasticEnvironment(
        environment=env,
        elevation=None,
        gravity=None,
        latitude=None,
        longitude=None,
    )

    stochastic_rocket = MagicMock()
    stochastic_flight = MagicMock()

    monte_carlo = MonteCarlo(
        filename="kennedy_space_center_test",
        environment=stochastic_env,
        rocket=stochastic_rocket,
        flight=stochastic_flight,
    )

    # Generate simulated Monte Carlo results
    # Simulate 50 data points with realistic dispersion
    np.random.seed(42)  # For reproducibility

    # Simulate apogee points (scattered around origin with some dispersion)
    n_points = 50
    apogee_x = np.random.normal(0, 500, n_points)  # meters, std dev 500m
    apogee_y = np.random.normal(0, 500, n_points)  # meters, std dev 500m

    # Simulate impact points (further from origin, more dispersion)
    impact_x = np.random.normal(2000, 1000, n_points)  # meters, mean 2km, std dev 1km
    impact_y = np.random.normal(1500, 1000, n_points)  # meters, mean 1.5km, std dev 1km

    monte_carlo.results = {
        "apogee_x": apogee_x.tolist(),
        "apogee_y": apogee_y.tolist(),
        "x_impact": impact_x.tolist(),
        "y_impact": impact_y.tolist(),
    }

    return monte_carlo


def test_all_background_options(example_kennedy_env):
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
    """
    output_dir = "kennedy_background_tests"
    os.makedirs(output_dir, exist_ok=True)

    monte_carlo = create_simulated_monte_carlo_data(example_kennedy_env)

    background_options = [
        (None, "no_background"),
        ("satellite", "satellite"),
        ("street", "street"),
        ("terrain", "terrain"),
        ("CartoDB.Positron", "cartodb_positron"),
    ]

    print(f"Testing {len(background_options)} background options...")
    print(f"Output directory: {output_dir}/")

    for background, name in background_options:
        try:
            print(f"  Testing {name}...", end=" ")

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

                # Check if file was created
                expected_file = f"{monte_carlo.filename}.png"
                if os.path.exists(expected_file):
                    print(f"✓ Saved to {expected_file}")
                else:
                    print("✗ Failed to save")
            finally:
                # Restore original filename
                monte_carlo.filename = original_filename

        except (ValueError, ImportError, OSError) as e:
            print(f"✗ Error: {str(e)}")

    print(f"\nAll tests completed! Check the '{output_dir}/' directory for results.")
