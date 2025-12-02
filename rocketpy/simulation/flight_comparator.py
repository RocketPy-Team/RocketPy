import warnings

import matplotlib.pyplot as plt
import numpy as np

from rocketpy.mathutils import Function
from rocketpy.simulation.flight import Flight
from rocketpy.simulation.flight_data_importer import FlightDataImporter

from ..plots.plot_helpers import show_or_save_fig


class FlightComparator:
    """
    A class to compare a RocketPy Flight simulation against external data sources
    (such as flight logs, OpenRocket simulations, RASAero).

    This class handles the time-interpolation required to compare datasets
    recorded at different frequencies, and computes error metrics (RMSE, MAE, etc.)
    between your RocketPy simulation and external or reference data.


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

        from rocketpy.simulation import FlightComparator

        # Suppose you have a Flight object named 'my_flight'
        comparator = FlightComparator(my_flight)

        # Add external data (e.g., from OpenRocket or logs)
        comparator.add_data('OpenRocket', {
            'altitude': (time_array, altitude_array),
            'vz': (time_array, velocity_array)
        })

        # You can also add another RocketPy Flight directly:
        comparator.add_data('OtherSimulation', other_flight)

        # Run comparisons
        comparator.compare('altitude')
        comparator.summary()
        events_table = comparator.compare_key_events()
    """

    DEFAULT_GRID_POINTS = 1000  # number of points for interpolation grids

    def __init__(self, flight: Flight):
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
        # Duck-typed validation gives clear errors for Flight-like objects, more flexible than an isinstance check
        required_attrs = ("t_final", "apogee", "apogee_time", "impact_velocity")
        missing = [attr for attr in required_attrs if not hasattr(flight, attr)]
        if missing:
            raise TypeError(
                "flight must be a rocketpy.Flight or Flight-like object with attributes "
                f"{required_attrs}. Missing: {', '.join(missing)}"
            )

        self.flight = flight
        self.data_sources = {}  # The format is {'Source Name': {'variable': Function}}

    def add_data(self, label, data_dict):  # pylint: disable=too-many-statements
        """
        Add an external dataset to the comparator.

        Parameters
        ----------
        label : str
            Name of the data source (e.g., "Avionics", "OpenRocket", "RASAero").
        data_dict : dict, Flight, or FlightDataImporter
            External data to be compared.

            If a dict, keys must be variable names (e.g., 'z', 'vz', 'az', 'altitude')
            and values can be:
                - A RocketPy Function object
                - A tuple/list of (time_array, data_array)

            If a Flight object is provided, standard Flight attributes such as
            'z', 'vz', 'x', 'y', 'speed', 'vx', 'vy', 'ax', 'ay', 'az', 'acceleration'
            will be registered automatically when available.

            If a FlightDataImporter object is provided, all flight attributes will be
            registered automatically. In both cases, 'altitude' will be aliased to 'z'
            if present.
        """

        if isinstance(data_dict, dict) and not data_dict:
            raise ValueError("data_dict cannot be empty")

        processed_data = {}

        # Case 1: dict
        if isinstance(data_dict, dict):
            for key, value in data_dict.items():
                if isinstance(value, Function):
                    processed_data[key] = value
                elif isinstance(value, (tuple, list)) and len(value) == 2:
                    time_arr, data_arr = value
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

        # Case 2: Flight instance
        elif isinstance(data_dict, Flight):
            external_flight = data_dict
            candidate_vars = [
                "z",
                "vz",
                "x",
                "y",
                "speed",
                "vx",
                "vy",
                "ax",
                "ay",
                "az",
                "acceleration",
            ]
            for var in candidate_vars:
                if hasattr(external_flight, var):
                    value = getattr(external_flight, var)
                    if isinstance(value, Function):
                        processed_data[var] = value

            # Provide 'altitude' alias for convenience if 'z' exists
            if "z" in processed_data and "altitude" not in processed_data:
                processed_data["altitude"] = processed_data["z"]

            if not processed_data:
                warnings.warn(
                    f"No comparable variables found when using Flight "
                    f"object for data source '{label}'."
                )

        # Case 3: FlightDataImporter instance
        elif isinstance(data_dict, FlightDataImporter):
            importer = data_dict
            candidate_vars = [
                "z",
                "vz",
                "x",
                "y",
                "speed",
                "vx",
                "vy",
                "ax",
                "ay",
                "az",
                "acceleration",
            ]
            for var in candidate_vars:
                if hasattr(importer, var):
                    value = getattr(importer, var)
                    if isinstance(value, Function):
                        processed_data[var] = value

            # Provide 'altitude' alias for convenience if 'z' exists
            if "z" in processed_data and "altitude" not in processed_data:
                processed_data["altitude"] = processed_data["z"]

            if not processed_data:
                warnings.warn(
                    f"No comparable variables found when using FlightDataImporter "
                    f"for data source '{label}'."
                )

        else:
            warnings.warn(
                f"Data source '{label}' not recognized. Expected a dict, Flight, "
                "or FlightDataImporter object."
            )

        if label in self.data_sources:
            warnings.warn(f"Data source '{label}' already exists. Overwriting.")

        self.data_sources[label] = processed_data
        print(
            f"Added data source '{label}' with variables: {list(processed_data.keys())}"
        )

    def _process_time_range(self, time_range):
        """
        Validate and normalize the time_range argument.

        Parameters
        ----------
        time_range : tuple of (float, float) or list of (float, float) or None
            Tuple or list specifying the start and end times (in seconds) for the comparison.
            If None, the full flight duration [0, flight.t_final] is used.

        Returns
        -------
        tuple of (float, float)
            The validated (t_min, t_max) time range in seconds, where
            0.0 <= t_min < t_max <= flight.t_final.

        Raises
        ------
        TypeError
            If time_range is not a tuple or list of two numeric values.
        ValueError
            If time_range values are invalid or out of bounds.
        """
        if time_range is None:
            return 0.0, self.flight.t_final

        if not isinstance(time_range, (tuple, list)) or len(time_range) != 2:
            raise TypeError(
                "time_range must be a (start_time, end_time) tuple or list."
            )

        t_min, t_max = time_range
        if not isinstance(t_min, (int, float)) or not isinstance(t_max, (int, float)):
            raise TypeError("time_range values must be numeric.")

        if t_min >= t_max:
            raise ValueError("time_range[0] must be strictly less than time_range[1].")

        if t_min < 0 or t_max > self.flight.t_final:
            raise ValueError(
                "time_range must lie within [0, flight.t_final]. "
                f"Got [{t_min}, {t_max}], flight.t_final={self.flight.t_final}."
            )

        return float(t_min), float(t_max)

    def _build_time_grid(self, t_min, t_max):
        """
        Build a time grid for interpolation between t_min and t_max.

        Parameters
        ----------
        t_min : float
            Start time of the grid, in seconds.
        t_max : float
            End time of the grid, in seconds.

        Returns
        -------
        numpy.ndarray
            Array of time points (in seconds) linearly spaced between t_min and t_max,
            with length equal to DEFAULT_GRID_POINTS.
        """
        return np.linspace(t_min, t_max, self.DEFAULT_GRID_POINTS)

    def _setup_compare_figure(self, figsize, attribute):
        """
        Create a matplotlib figure and axes for the compare() method.

        Parameters
        ----------
        figsize : tuple of float
            Size of the figure in inches as (width, height).
        attribute : str
            Name of the attribute being compared, used for the plot title.

        Returns
        -------
        tuple
            A tuple containing:
            - fig : matplotlib.figure.Figure
                The created figure object.
            - ax1 : matplotlib.axes.Axes
                The axes object for the main comparison plot.
            - ax2 : matplotlib.axes.Axes
                The axes object for the residuals (error) plot.
        """
        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=figsize,
            sharex=True,
            gridspec_kw={"height_ratios": [2, 1]},
        )
        ax1.set_title(f"Flight Comparison: {attribute}")
        ax2.set_title("Residuals (Simulation - External)")
        ax2.set_xlabel("Time (s)")
        return fig, ax1, ax2

    def _plot_reference_series(self, ax, t_grid, y_sim):
        """
        Plot RocketPy reference curve on the given axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            he axes object on which to plot the reference curve.
        t_grid : numpy.ndarray
            Array of time points, in seconds.
        y_sim : numpy.ndarray
            Array of simulated values corresponding to t_grid.

        Returns
        -------
        None

        """
        ax.plot(
            t_grid,
            y_sim,
            label="RocketPy Simulation",
            linewidth=2,
            color="black",
            alpha=0.8,
        )

    def _plot_external_sources(
        self,
        attribute,
        t_grid,
        y_sim,
        ax_values,
        ax_errors,
    ):
        """
        Plot external data sources and print error metrics.

        Parameters
        ----------
        attribute : str
            Name of the attribute to compare (e.g., 'altitude', 'vz').
        t_grid : np.ndarray
            1D array of time points (in seconds) at which to evaluate and plot the data.
        y_sim : np.ndarray
            1D array of simulated values corresponding to t_grid.
        ax_values : matplotlib.axes.Axes
            Axes object to plot the simulation and external data values.
        ax_errors : matplotlib.axes.Axes
            Axes object to plot the error (residuals) between simulation and external data.

        Returns
        -------
        bool
            True if at least one external source had the specified attribute and data was plotted.
        """
        has_plots = False

        print(f"\n{'-' * 20}")
        print(f"COMPARISON REPORT: {attribute}")
        print(f"{'-' * 20}")

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
            ax_values.plot(t_grid, y_ext, label=label, linestyle="--")

            # Plot Error
            ax_errors.plot(t_grid, error, label=f"Error ({label})")

        return has_plots

    def _finalize_compare_figure(
        self, fig, ax_values, ax_errors, attribute, legend, filename
    ):
        """
        Apply formatting, legends, and show/save the comparison figure.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure object containing the comparison plots.
        ax_values : matplotlib.axes.Axes
            The axes object displaying the compared values.
        ax_errors : matplotlib.axes.Axes
            The axes object displaying the residuals/errors.
        attribute : str
            Name of the attribute being compared (used for labels).
        legend : bool
            Whether to display legends on both axes.
        filename : str or None
            If provided, save the figure to this file path. If None, display the figure.

        Returns
        -------
        None
        """
        ax_values.set_ylabel(attribute)
        ax_values.grid(True, linestyle=":", alpha=0.6)

        ax_errors.set_ylabel("Difference")
        ax_errors.grid(True, linestyle=":", alpha=0.6)

        if legend:
            ax_values.legend()
            ax_errors.legend()

        fig.tight_layout()
        show_or_save_fig(fig, filename)
        if filename:
            print(f"Plot saved to file: {filename}")

    def compare(  # pylint: disable=too-many-statements
        self,
        attribute,
        time_range=None,
        figsize=(10, 8),
        legend=True,
        filename=None,
    ):
        """
        Compare a specific attribute (e.g., altitude, velocity) across all added data sources.

        This method generates a plot comparing the specified attribute from the reference
        RocketPy Flight object and all added external data sources (e.g., OpenRocket, flight logs).
        It interpolates all data onto a common time grid, computes error metrics (RMSE, MAE,
        relative error), and displays or saves the resulting plot.

        Parameters
        ----------
        attribute : str
            Name of the attribute to compare (e.g., "altitude", "vz", "ax").
            The attribute must be present as a callable (function or property) in the
            reference Flight object and in each external data source.
        time_range : tuple of float, optional
            Tuple specifying the time range (t_min, t_max) in seconds for the comparison.
            If None (default), uses the full time range of the reference Flight.
        figsize : tuple of float, optional
            Size of the figure in inches, as (width, height). Default is (10, 8).
        legend : bool, optional
            Whether to display a legend on the plot. Default is True.
        filename : str or None, optional
            If provided, saves the plot to the specified file path. If None (default),
            the plot is shown interactively.

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

        sim_func = getattr(self.flight, attribute)

        # 2. Process time range and build grid
        t_min, t_max = self._process_time_range(time_range)
        t_grid = self._build_time_grid(t_min, t_max)

        # Interpolate Simulation onto the grid
        y_sim = sim_func(t_grid)

        # 3. Set up figure and plot reference
        fig, ax_values, ax_errors = self._setup_compare_figure(figsize, attribute)
        self._plot_reference_series(ax_values, t_grid, y_sim)

        # 4. Plot external sources and metrics
        has_plots = self._plot_external_sources(
            attribute=attribute,
            t_grid=t_grid,
            y_sim=y_sim,
            ax_values=ax_values,
            ax_errors=ax_errors,
        )

        if not has_plots:
            warnings.warn(f"No external sources have data for variable '{attribute}'.")
            plt.close(fig)
            return

        # 5. Final formatting and save/show
        self._finalize_compare_figure(
            fig=fig,
            ax_values=ax_values,
            ax_errors=ax_errors,
            attribute=attribute,
            legend=legend,
            filename=filename,
        )

    def compare_key_events(self):  # pylint: disable=too-many-statements
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
        t_grid = np.linspace(0, self.flight.t_final, self.DEFAULT_GRID_POINTS)
        altitude_cache = {}
        for label, dataset in self.data_sources.items():
            if "altitude" in dataset or "z" in dataset:
                alt_func = dataset.get("altitude", dataset.get("z"))
                altitude_cache[label] = alt_func(t_grid)
        # 1. Compare Apogee Altitude
        rocketpy_apogee = self.flight.apogee
        apogee_results = {"RocketPy": rocketpy_apogee}

        for label, dataset in self.data_sources.items():
            if label in altitude_cache:
                altitudes = altitude_cache[label]
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
            if label in altitude_cache:
                altitudes = altitude_cache[label]
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

    def summary(self):  # pylint: disable=too-many-statements
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

        try:
            events_results = self.compare_key_events()
            print("\n" + self._format_key_events_table(events_results))
            print(
                "\nNote: Values marked with * are approximations "
                "(e.g., speed from vz only)"
            )
        except (KeyError, AttributeError, ValueError) as exc:
            print(
                "Could not generate key events table. "
                "Ensure external data sources contain compatible variables "
                "such as 'altitude' or 'z' for altitude and 'speed' or 'vz' "
                "for velocity. Details: "
                f"{exc}"
            )

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
            Whether or not to show the legend, by default True

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

    def trajectories_2d(self, plane="xz", figsize=(7, 7), legend=True, filename=None):  # pylint: disable=too-many-statements
        """
        Compare 2D flight trajectories between RocketPy simulation and external sources.
        Coordinates are plotted in the inertial NED-like frame used by Flight:
        x is East, y is North and z is Up.

        Parameters
        ----------
        plane : str, optional
            Plane to plot: 'xy', 'xz', or 'yz'. Default is 'xz'.
        figsize : tuple, optional
            standard matplotlib figsize to be used in the plots, by default
            (7, 7), where the tuple means (width, height).
        legend : bool, optional
            Whether or not to show the legend, by default True
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
        t_grid = np.linspace(0, self.flight.t_final, self.DEFAULT_GRID_POINTS)

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
