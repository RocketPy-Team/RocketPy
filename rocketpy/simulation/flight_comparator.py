import warnings

import matplotlib.pyplot as plt
import numpy as np

from rocketpy.mathutils import Function

from ..plots.plot_helpers import show_or_save_fig


# pylint: disable=too-many-public-methods
# pylint: disable=too-many-statements
class FlightComparator:
    """
    A class to compare a RocketPy Flight simulation against external data sources
    (such as flight logs, OpenRocket simulations, RAS Aero).

    This class handles the time-interpolation required to compare datasets
    recorded at different frequencies.

    Parameters
    ----------
    flight : Flight
        The reference RocketPy Flight object to compare against.

    Attributes
    ----------
    flight : Flight
        The reference RocketPy Flight object to compare against.
    data_sources : dict
        Dictionary storing external data sources in the format
        {'Source Name': {'variable': Function}}.

    Examples
    --------

    .. code-block:: python

        # Assuming you have a Flight object named 'my_flight'
        comparator = FlightComparator(my_flight)

        # Add external data
        comparator.add_data('OpenRocket', {
            'altitude': (time_array, altitude_array),
            'vz': (time_array, velocity_array)
        })

        # Run comparisons
        comparator.compare('altitude')
        comparator.summary()
        events_table = comparator.compare_key_events()
    """

    def __init__(self, flight):
        """
        Initialize the comparator with a reference RocketPy Flight.

        Parameters
        ----------
        flight : rocketpy.Flight
            The reference RocketPy Flight object to compare against.

        Returns
        -------
        None
        """
        self.flight = flight
        self.data_sources = {}  # The format is {'Source Name': {'variable': Function}}

    def add_data(self, label, data_dict):
        """
        Add an external dataset to the comparator.

        Parameters
        ----------
        label : str
            Name of the data source (e.g., "Avionics", "OpenRocket", "RAS Aero").
        data_dict : dict
            Dictionary containing the variables to compare.
            Keys must be variable names (e.g., 'z', 'vz', 'az', 'altitude').
            Values can be:
                - A RocketPy Function object
                - A tuple/list of (time_array, data_array)

        Returns
        -------
        None
        """
        # Check if label already exists
        if label in self.data_sources:
            warnings.warn(f"Data source '{label}' already exists. Overwriting.")

        # Making sure that data_dict is not empty
        if not data_dict:
            raise ValueError("data_dict cannot be empty")

        processed_data = {}

        for key, value in data_dict.items():
            # If already a Function, store it
            if isinstance(value, Function):
                processed_data[key] = value

            # If raw data, convert to a function
            elif isinstance(value, (tuple, list)) and len(value) == 2:
                time_arr, data_arr = value
                # Creating a Function for automatic interpolation
                processed_data[key] = Function(
                    np.column_stack((time_arr, data_arr)),
                    inputs="Time (s)",
                    outputs=key,
                    interpolation="linear",
                )
            else:
                warnings.warn(
                    f"Skipping '{key}' in '{label}'. Format not recognized. "
                    "Expected RocketPy Function or (time, data) tuple."
                )

        self.data_sources[label] = processed_data
        print(
            f"Added data source '{label}' with variables: {list(processed_data.keys())}"
        )

    def compare(
        self, attribute, time_range=None, figsize=(10, 8), legend=True, filename=None
    ):
        """
        Compares a specific attribute across all added data sources.
        Generates a plot and prints error metrics (RMSE, MAE, relative error).

        Parameters
        ----------
        attribute : str
            The attribute name to compare (e.g., 'z', 'vz').
            This must exist as an attribute in the RocketPy Flight object.
        time_range : tuple, optional
            (start_time, end_time) to restrict the comparison.
            If None, uses the full duration of the RocketPy simulation.
        figsize : tuple, optional
            standard matplotlib figsize to be used in the plots, by default
            (10, 8), where the tuple means (width, height).
        legend : bool, optional
            Weather or not to show the legend, by default True
        filename : str, optional
            If a filename is provided, the plot will be saved to a file, by
            default None. Image options are: png, pdf, ps, eps and svg.

        Returns
        -------
        None
        """
        # 1. Get RocketPy Simulation Data
        if not hasattr(self.flight, attribute):
            warnings.warn(
                f"Attribute '{attribute}' not found in the RocketPy Flight object."
            )
            return

        # Get the simulated function
        sim_func = getattr(self.flight, attribute)

        # Determining the duration for comparison
        if time_range:
            t_min, t_max = time_range
        else:
            t_min = 0  # Start at liftoff
            # Default to end of simulation
            t_max = self.flight.t_final

        # Create a 1000-point time grid to evaluate both functions
        t_grid = np.linspace(t_min, t_max, 1000)

        # Interpolate Simulation onto the grid
        y_sim = sim_func(t_grid)

        # 2. Setting up the Plot (Top: Values, Bottom: Error)
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [2, 1]}
        )

        # Plot RocketPy Reference
        ax1.plot(
            t_grid,
            y_sim,
            label="RocketPy Simulation",
            linewidth=2,
            color="black",
            alpha=0.8,
        )

        print(f"\n{'-' * 20}")
        print(f"COMPARISON REPORT: {attribute}")
        print(f"{'-' * 20}")

        # 3. Going through External Sources and comparing
        has_plots = False
        for label, dataset in self.data_sources.items():
            if attribute not in dataset:
                continue

            has_plots = True
            ext_func = dataset[attribute]

            # Interpolate External Data onto the same grid
            y_ext = ext_func(t_grid)

            # Calculate Error (Residuals)
            error = y_sim - y_ext

            # Calculate Metrics
            mae = np.mean(np.abs(error))  # Mean Absolute Error
            rmse = np.sqrt(np.mean(error**2))  # Root Mean Square Error
            max_dev = np.max(np.abs(error))  # Max Deviation

            # Calculate Relative Error Percentage
            mean_abs_y_sim = np.mean(np.abs(y_sim))
            relative_error_pct = (
                (rmse / mean_abs_y_sim) * 100 if mean_abs_y_sim != 0 else np.inf
            )

            # Print Metrics
            print(f"Source: {label}")
            print(f"  - MAE:            {mae:.4f}")
            print(f"  - RMSE:           {rmse:.4f}")
            print(f"  - Max Deviation:  {max_dev:.4f}")
            print(f"  - Relative Error: {relative_error_pct:.2f}%")

            # Plot Data
            ax1.plot(t_grid, y_ext, label=label, linestyle="--")

            # Plot Error
            ax2.plot(t_grid, error, label=f"Error ({label})")

        if not has_plots:
            warnings.warn(f"No external sources have data for variable '{attribute}'.")
            plt.close(fig)
            return

        # Formatting
        ax1.set_title(f"Flight Comparison: {attribute}")
        ax1.set_ylabel(attribute)
        if legend:
            ax1.legend()
        ax1.grid(True, linestyle=":", alpha=0.6)

        ax2.set_title("Residuals (Simulation - External)")
        ax2.set_ylabel("Difference")
        ax2.set_xlabel("Time (s)")
        if legend:
            ax2.legend()
        ax2.grid(True, linestyle=":", alpha=0.6)

        fig.tight_layout()

        # Using the existing helper function
        show_or_save_fig(fig, filename)
        if filename:
            print(f"Plot saved to file: {filename}")

    def compare_key_events(self):
        """
        Compare critical flight events across all data sources.

        Returns
        -------
        dict
            Comparison dictionary with metrics as keys, containing RocketPy values
            and errors for each external data source.
        """
        # Initialize results dictionary
        results = {}

        # Create time grid for interpolation
        t_grid = np.linspace(0, self.flight.t_final, 1000)

        # 1. Compare Apogee Altitude
        rocketpy_apogee = self.flight.apogee
        apogee_results = {"RocketPy": rocketpy_apogee}

        for label, dataset in self.data_sources.items():
            if "altitude" in dataset or "z" in dataset:
                alt_func = dataset.get("altitude", dataset.get("z"))
                altitudes = alt_func(t_grid)
                ext_apogee = np.max(altitudes)
                error = ext_apogee - rocketpy_apogee
                rel_error = (
                    (error / rocketpy_apogee) * 100 if rocketpy_apogee != 0 else np.inf
                )

                apogee_results[label] = {
                    "value": ext_apogee,
                    "error": error,
                    "error_percentage": rel_error,
                }

        results["Apogee Altitude (m)"] = apogee_results

        # 2. Compare Apogee Time
        rocketpy_apogee_time = self.flight.apogee_time
        apogee_time_results = {"RocketPy": rocketpy_apogee_time}

        for label, dataset in self.data_sources.items():
            if "altitude" in dataset or "z" in dataset:
                alt_func = dataset.get("altitude", dataset.get("z"))
                altitudes = alt_func(t_grid)
                ext_apogee_time = t_grid[np.argmax(altitudes)]
                error = ext_apogee_time - rocketpy_apogee_time
                rel_error = (
                    (error / rocketpy_apogee_time) * 100
                    if rocketpy_apogee_time != 0
                    else np.inf
                )

                apogee_time_results[label] = {
                    "value": ext_apogee_time,
                    "error": error,
                    "error_percentage": rel_error,
                }

        results["Apogee Time (s)"] = apogee_time_results

        # 3. Compare Maximum Velocity
        rocketpy_max_vel = self.flight.max_speed
        max_vel_results = {"RocketPy": rocketpy_max_vel}

        for label, dataset in self.data_sources.items():
            if "speed" in dataset:
                speed_func = dataset["speed"]
                speeds = speed_func(t_grid)
                ext_max_vel = np.max(speeds)
                error = ext_max_vel - rocketpy_max_vel
                rel_error = (
                    (error / rocketpy_max_vel) * 100
                    if rocketpy_max_vel != 0
                    else np.inf
                )

                max_vel_results[label] = {
                    "value": ext_max_vel,
                    "error": error,
                    "error_percentage": rel_error,
                    "approximation": False,
                }
            elif "vz" in dataset:
                vz_func = dataset["vz"]
                vz_vals = vz_func(t_grid)
                ext_max_vel = np.max(np.abs(vz_vals))
                error = ext_max_vel - rocketpy_max_vel
                rel_error = (
                    (error / rocketpy_max_vel) * 100
                    if rocketpy_max_vel != 0
                    else np.inf
                )

                max_vel_results[label] = {
                    "value": ext_max_vel,
                    "error": error,
                    "error_percentage": rel_error,
                    "approximation": True,
                }

        results["Max Velocity (m/s)"] = max_vel_results

        # 4. Compare Impact Velocity
        rocketpy_impact_vel = self.flight.impact_velocity
        impact_vel_results = {"RocketPy": rocketpy_impact_vel}

        for label, dataset in self.data_sources.items():
            if "speed" in dataset:
                speed_func = dataset["speed"]
                ext_impact_vel = abs(speed_func(t_grid[-1]))
                error = ext_impact_vel - rocketpy_impact_vel
                rel_error = (
                    (error / rocketpy_impact_vel) * 100
                    if rocketpy_impact_vel != 0
                    else np.inf
                )

                impact_vel_results[label] = {
                    "value": ext_impact_vel,
                    "error": error,
                    "error_percentage": rel_error,
                    "approximation": False,
                }
            elif "vz" in dataset:
                vz_func = dataset["vz"]
                ext_impact_vel = abs(vz_func(t_grid[-1]))
                error = ext_impact_vel - rocketpy_impact_vel
                rel_error = (
                    (error / rocketpy_impact_vel) * 100
                    if rocketpy_impact_vel != 0
                    else np.inf
                )

                impact_vel_results[label] = {
                    "value": ext_impact_vel,
                    "error": error,
                    "error_percentage": rel_error,
                    "approximation": True,
                }

        results["Impact Velocity (m/s)"] = impact_vel_results

        return results

    def _format_key_events_table(self, results):
        """
        Format key events results as a string table.

        Parameters
        ----------
        results : dict
            Results from compare_key_events()

        Returns
        -------
        str
            Formatted table string
        """
        lines = []

        # Get all source names
        sources = []
        for metric_data in results.values():
            for key in metric_data.keys():
                if key != "RocketPy" and key not in sources:
                    sources.append(key)

        # Header
        header = f"{'Metric':<25} {'RocketPy':<15}"
        for source in sources:
            header += (
                f" {source:<15} {source + ' Error':<15} {source + ' Error (%)':<15}"
            )
        lines.append(header)
        lines.append("-" * len(header))

        # Rows
        for metric, data in results.items():
            row = f"{metric:<25} {data['RocketPy']:<15.2f}"

            for source in sources:
                if source in data:
                    value = data[source]["value"]
                    error = data[source]["error"]
                    error_pct = data[source]["error_percentage"]
                    approx = "*" if data[source].get("approximation", False) else ""
                    row += f" {value:<15.2f}{approx} {error:<15.2f} {error_pct:<15.2f}"
                else:
                    row += f" {'N/A':<15} {'N/A':<15} {'N/A':<15}"

            lines.append(row)

        return "\n".join(lines)

    def summary(self):
        """
        Print comprehensive comparison summary including key events and metrics.

        Returns
        -------
        None
        """
        print("\n" + "=" * 60)
        print("FLIGHT COMPARISON SUMMARY")
        print("=" * 60)

        print("\nRocketPy Simulation:")
        print(
            f"  - Apogee: {self.flight.apogee:.2f} m at t={self.flight.apogee_time:.2f} s"
        )
        print(f"  - Max velocity: {self.flight.max_speed:.2f} m/s")
        print(f"  - Impact velocity: {self.flight.impact_velocity:.2f} m/s")
        print(f"  - Flight duration: {self.flight.t_final:.2f} s")

        print(f"\nExternal Data Sources: {list(self.data_sources.keys())}")

        # Display key events comparison table
        try:
            events_results = self.compare_key_events()
            print("\n" + self._format_key_events_table(events_results))
            print(
                "\nNote: Values marked with * are approximations (e.g., speed from vz only)"
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"Could not generate key events table: {e}")

        print("\n" + "=" * 60)

    def all(self, time_range=None, figsize=(10, 8), legend=True):
        """
        Generate comparison plots for all common variables found in both
        the RocketPy simulation and external data sources.

        Parameters
        ----------
        time_range : tuple, optional
            (start_time, end_time) to restrict comparisons.
            If None, uses the full flight duration.
        figsize : tuple, optional
            standard matplotlib figsize to be used in the plots, by default
            (10, 8), where the tuple means (width, height).
        legend : bool, optional
            Weather or not to show the legend, by default True

        Returns
        -------
        None
        """
        # Common variables to check for
        common_vars = [
            "z",
            "vz",
            "ax",
            "ay",
            "az",
            "altitude",
            "speed",
            "vx",
            "vy",
            "acceleration",
        ]

        # Find which variables are available in both simulation and at least one source
        available_vars = []
        for var in common_vars:
            if hasattr(self.flight, var):
                # Check if at least one source has this variable
                for dataset in self.data_sources.values():
                    if var in dataset:
                        available_vars.append(var)
                        break

        if not available_vars:
            print("No common variables found for comparison.")
            return

        print(f"\nGenerating comparison plots for: {', '.join(available_vars)}\n")

        # Generate a plot for each available variable
        for var in available_vars:
            self.compare(var, time_range=time_range, figsize=figsize, legend=legend)

    def trajectories_2d(self, plane="xz", figsize=(7, 7), legend=True, filename=None):
        """
        Compare 2D flight trajectories between RocketPy simulation and external sources.

        Parameters
        ----------
        plane : str, optional
            Plane to plot: 'xy', 'xz', or 'yz'. Default is 'xz'.
        figsize : tuple, optional
            standard matplotlib figsize to be used in the plots, by default
            (7, 7), where the tuple means (width, height).
        legend : bool, optional
            Weather or not to show the legend, by default True
        filename : str, optional
            If a filename is provided, the plot will be saved to a file, by
            default None. Image options are: png, pdf, ps, eps and svg.

        Returns
        -------
        None
        """
        if plane not in ["xy", "xz", "yz"]:
            raise ValueError("plane must be 'xy', 'xz', or 'yz'")

        axis1, axis2 = plane[0], plane[1]

        # Check if Flight object has the required attributes
        if not hasattr(self.flight, axis1) or not hasattr(self.flight, axis2):
            warnings.warn(f"Flight object missing {axis1} or {axis2} attributes")
            return

        # Create figure
        fig = plt.figure(figsize=figsize)
        fig.suptitle("Flight Trajectories Comparison", fontsize=16, y=0.95, x=0.5)
        ax = plt.subplot(111)

        # Create time grid for evaluation
        t_grid = np.linspace(0, self.flight.t_final, 1000)

        # Plot RocketPy trajectory
        x_sim = getattr(self.flight, axis1)(t_grid)
        y_sim = getattr(self.flight, axis2)(t_grid)

        ax.plot(x_sim, y_sim, label="RocketPy", linewidth=2, color="black", alpha=0.8)

        # Plot external sources
        has_plots = False
        for label, dataset in self.data_sources.items():
            if axis1 in dataset and axis2 in dataset:
                has_plots = True
                x_ext = dataset[axis1](t_grid)
                y_ext = dataset[axis2](t_grid)
                ax.plot(x_ext, y_ext, label=label, linestyle="--", linewidth=1.5)

        if not has_plots:
            warnings.warn(f"No external sources have both {axis1} and {axis2} data.")
            plt.close(fig)
            return

        # Formatting
        axis_labels = {"x": "X - East (m)", "y": "Y - North (m)", "z": "Z - Up (m)"}
        ax.set_xlabel(axis_labels.get(axis1, f"{axis1} (m)"))
        ax.set_ylabel(axis_labels.get(axis2, f"{axis2} (m)"))
        ax.scatter(0, 0, color="black", s=10, marker="o")
        ax.grid(True)

        # Add legend
        if legend:
            fig.legend()

        fig.tight_layout()

        show_or_save_fig(fig, filename)
        if filename:
            print(f"Plot saved to file: {filename}")
