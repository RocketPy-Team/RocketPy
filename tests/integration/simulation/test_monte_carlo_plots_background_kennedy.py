"""Test script to visualize different background map options at Kennedy Space Center.

This script creates simulated Monte Carlo data for Kennedy Space Center
and tests all background map options, saving the results as images.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from rocketpy import Environment
from rocketpy.simulation import MonteCarlo
from rocketpy.stochastic import StochasticEnvironment

plt.rcParams.update({"figure.max_open_warning": 0})


def create_kennedy_environment():
    """Create an Environment object for Kennedy Space Center.
    
    Kennedy Space Center coordinates:
    - Latitude: 28.5721° N
    - Longitude: -80.6480° W
    - Elevation: ~3 meters
    
    Returns
    -------
    Environment
        Environment object configured for Kennedy Space Center.
    """
    env = Environment(
        latitude=28.5721,
        longitude=-80.6480,
        elevation=3.0,
        datum="WGS84",
    )
    # Set a date for the environment (tomorrow at noon UTC)
    tomorrow = datetime.now() + timedelta(days=1)
    # Use tuple format: (year, month, day, hour)
    env.set_date((tomorrow.year, tomorrow.month, tomorrow.day, 12), timezone="UTC")
    return env


def create_simulated_monte_carlo_data():
    """Create a MonteCarlo object with simulated results data.
    
    Returns
    -------
    MonteCarlo
        MonteCarlo object with simulated apogee and impact data.
    """
    # Create environment for Kennedy Space Center
    env = create_kennedy_environment()

    # Create stochastic environment (no randomization for simplicity)
    stochastic_env = StochasticEnvironment(
        environment=env,
        elevation=None,
        gravity=None,
        latitude=None,
        longitude=None,
    )


    from unittest.mock import MagicMock

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


def test_all_background_options():
    """Test all background map options and save images.
    
    This function tests:
    - None (no background)
    - satellite
    - street
    - terrain
    - custom provider (CartoDB.Positron)
    """
    output_dir = "kennedy_background_tests"
    os.makedirs(output_dir, exist_ok=True)

    monte_carlo = create_simulated_monte_carlo_data()

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


if __name__ == "__main__":
    test_all_background_options()