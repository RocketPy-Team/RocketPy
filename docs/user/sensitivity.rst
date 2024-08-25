Sensitivity Analysis
====================

You can use the results from a Monte Carlo simulation to perform sensitivity analysis.
We will first introduce the concepts of sensitivity analysis and then show how to use the `SensitivityModel` class.

It is highly recommended that you read about the Monte Carlo simulations.

...

.. jupyter-execute::

    analysis_parameters = {
        # Rocket
        "mass": {"mean": 14.426, "std": 0.5},
        "radius": {"mean": 127 / 2000, "std": 1 / 1000},
        # Motor
        "motors_dry_mass": {"mean": 1.815, "std": 1 / 100},
        "motors_grain_density": {"mean": 1815, "std": 50},
        "motors_total_impulse": {"mean": 6500, "std": 50},
        "motors_burn_out_time": {"mean": 3.9, "std": 0.2},
        "motors_nozzle_radius": {"mean": 33 / 1000, "std": 0.5 / 1000},
        "motors_grain_separation": {"mean": 5 / 1000, "std": 1 / 1000},
        "motors_grain_initial_height": {"mean": 120 / 1000, "std": 1 / 100},
        "motors_grain_initial_inner_radius": {"mean": 15 / 1000, "std": 0.375 / 1000},
        "motors_grain_outer_radius": {"mean": 33 / 1000, "std": 0.375 / 1000},
        # Parachutes
        "parachutes_cd_s": {"mean": 10, "std": 0.1},
        "parachutes_lag": {"mean": 1.5, "std": 0.1},
        # Flight
        "heading": {"mean": 53, "std": 2},
        "inclination": {"mean": 84.7, "std": 1},
    }

