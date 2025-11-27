"""
Monte Carlo Simulation Module for RocketPy

This module defines the `MonteCarlo` class, which is used to perform Monte Carlo
simulations of rocket flights. The Monte Carlo simulation is a powerful tool for
understanding the variability and uncertainty in the performance of rocket flights
by running multiple simulations with varied input parameters.

Notes
-----
This module is still under active development, and some features or attributes may
change in future versions. Users are encouraged to check for updates and read the
latest documentation.
"""

import json
import os
import traceback
import warnings
from pathlib import Path
from time import time

import numpy as np
import simplekml

from rocketpy._encoders import RocketPyEncoder
from rocketpy.plots.monte_carlo_plots import _MonteCarloPlots
from rocketpy.prints.monte_carlo_prints import _MonteCarloPrints
from rocketpy.simulation.flight import Flight
from rocketpy.tools import (
    generate_monte_carlo_ellipses,
    generate_monte_carlo_ellipses_coordinates,
    import_optional_dependency,
)

# TODO: Create evolution plots to analyze convergence


class MonteCarlo:
    """Class to run a Monte Carlo simulation of a rocket flight.

    Attributes
    ----------
    filename : str
        Represents the initial part of the export filenames or the .txt file
        containing the outputs of a previous simulation.
    environment : StochasticEnvironment
        The stochastic environment object to be iterated over.
    rocket : StochasticRocket
        The stochastic rocket object to be iterated over.
    flight : StochasticFlight
        The stochastic flight object to be iterated over.
    export_list : list
        The list of variables to export at each simulation.
    data_collector : dict
        A dictionary whose keys are the names of the additional
        exported variables and the values are callback functions.
    inputs_log : list
        List of dictionaries with the inputs used in each simulation.
    outputs_log : list
        List of dictionaries with the outputs of each simulation.
    errors_log : list
        List of dictionaries with the errors of each simulation.
    num_of_loaded_sims : int
        Number of simulations loaded from output_file currently being used.
    results : dict
        Monte Carlo analysis results organized in a dictionary where the keys
        are the names of the saved attributes, and the values are lists with all
        the result numbers of the respective attributes.
    processed_results : dict
        Dictionary with the mean and standard deviation of each parameter
        available in the results.
    prints : _MonteCarloPrints
        Object with methods to print information about the Monte Carlo simulation.
        Use help(MonteCarlo.prints) for more information.
    plots : _MonteCarloPlots
        Object with methods to plot information about the Monte Carlo simulation.
        Use help(MonteCarlo.plots) for more information.
    number_of_simulations : int
        Number of simulations to be run.
    total_wall_time : float
        The total elapsed real-world time from the start to the end of the
        simulation, including all waiting times and delays.
    total_cpu_time : float
        The total CPU time spent running the simulation, excluding the time
        spent waiting for I/O operations or other processes to complete.
    """

    def __init__(
        self,
        filename,
        environment,
        rocket,
        flight,
        export_list=None,
        data_collector=None,
    ):  # pylint: disable=too-many-statements
        """
        Initialize a MonteCarlo object.

        Parameters
        ----------
        filename : str
            Represents the initial part of the export filenames or the .txt file
            containing the outputs of a previous simulation.
        environment : StochasticEnvironment
            The stochastic environment object to be iterated over.
        rocket : StochasticRocket
            The stochastic rocket object to be iterated over.
        flight : StochasticFlight
            The stochastic flight object to be iterated over.
        export_list : list, optional
            The list of variables to export. If None, the default list will be
            used, which includes the following variables: `apogee`, `apogee_time`,
            `apogee_x`, `apogee_y`, `t_final`, `x_impact`, `y_impact`,
            `impact_velocity`, `initial_stability_margin`,
            `out_of_rail_stability_margin`, `out_of_rail_time`,
            `out_of_rail_velocity`, `max_mach_number`, `frontal_surface_wind`,
            `lateral_surface_wind`. Default is None.
        data_collector : dict, optional
            A dictionary whose keys are the names of the exported variables
            and the values are callback functions. The keys (variable names) must not
            overwrite the default names on 'export_list'. The callback functions receive
            a Flight object and returns a value of that variable. For instance

            .. code-block:: python

                custom_data_collector = {
                    "max_acceleration": lambda flight: max(flight.acceleration(flight.time)),
                    "date": lambda flight: flight.env.date,
                }

        Returns
        -------
        None
        """
        warnings.warn(
            "This class is still under testing and some attributes may be "
            "changed in next versions",
            UserWarning,
        )

        self.filename = Path(filename)
        self.environment = environment
        self.rocket = rocket
        self.flight = flight
        self.export_list = []
        self.inputs_log = []
        self.outputs_log = []
        self.errors_log = []
        self.num_of_loaded_sims = 0
        self.results = {}
        self.processed_results = {}
        self.prints = _MonteCarloPrints(self)
        self.plots = _MonteCarloPlots(self)

        self.export_list = self.__check_export_list(export_list)
        self._check_data_collector(data_collector)
        self.data_collector = data_collector

        self.import_inputs(self.filename.with_suffix(".inputs.txt"))
        self.import_outputs(self.filename.with_suffix(".outputs.txt"))
        self.import_errors(self.filename.with_suffix(".errors.txt"))

    def simulate(
        self,
        number_of_simulations,
        append=False,
        parallel=False,
        n_workers=None,
        **kwargs,
    ):  # pylint: disable=too-many-statements
        """
        Runs the Monte Carlo simulation and saves all data.

        Parameters
        ----------
        number_of_simulations : int
            Number of simulations to be run, must be non-negative.
        append : bool, optional
            If True, the results will be appended to the existing files. If
            False, the files will be overwritten. Default is False.
        parallel : bool, optional
            If True, the simulations will be run in parallel. Default is False.
        n_workers : int, optional
            Number of workers to be used if ``parallel=True``. If None, the
            number of workers will be equal to the number of CPUs available.
            A minimum of 2 workers is required for parallel mode.
            Default is None.
        kwargs : dict
            Custom arguments for simulation export of the ``inputs`` file. Options
            are:

                * ``include_outputs``: whether to also include outputs data of the
                  simulation. Default is ``False``.

                * ``include_function_data``: whether to include ``rocketpy.Function``
                  results into the export. Default is ``True``.

            See ``rocketpy._encoders.RocketPyEncoder`` for more information.

        Returns
        -------
        None

        Notes
        -----
        If you need to stop the simulations after starting them, you can
        interrupt the process and the files will be saved with the results
        until the last iteration. You can then load the results and continue
        the simulation by running the ``simulate`` method again with the
        same number of simulations and setting `append=True`.

        Important
        ---------
        If you use `append=False` and the files already exist, they will be
        overwritten. Make sure to save the files with the results before
        running the simulation again with `append=False`.
        """
        self._export_config = kwargs
        self.number_of_simulations = number_of_simulations
        self._initial_sim_idx = self.num_of_loaded_sims if append else 0

        _SimMonitor.reprint("Starting Monte Carlo analysis")

        self.__setup_files(append)

        if parallel:
            self.__run_in_parallel(n_workers)
        else:
            self.__run_in_serial()

        self.__terminate_simulation()

    def __setup_files(self, append):
        """
        Sets up the files for the simulation, creating them if necessary.

        Parameters
        ----------
        append : bool
            If ``True``, the results will be appended to the existing files. If
            ``False``, the files will be overwritten.

        Returns
        -------
        None
        """
        # Create data files for inputs, outputs and error logging
        open_mode = "r+" if append else "w+"

        try:
            with open(self._input_file, open_mode, encoding="utf-8") as input_file:
                idx_i = len(input_file.readlines())
            with open(self._output_file, open_mode, encoding="utf-8") as output_file:
                idx_o = len(output_file.readlines())
            with open(self._error_file, open_mode, encoding="utf-8"):
                pass

            if idx_i != idx_o and not append:
                warnings.warn(
                    "Input and output files are not synchronized", UserWarning
                )

        except OSError as error:
            raise OSError(f"Error creating files: {error}") from error

    def __run_in_serial(self):
        """
        Runs the monte carlo simulation in serial mode.

        Returns
        -------
        None
        """
        sim_monitor = _SimMonitor(
            initial_count=self._initial_sim_idx,
            n_simulations=self.number_of_simulations,
            start_time=time(),
        )
        try:
            while sim_monitor.keep_simulating():
                sim_monitor.increment()
                inputs_json, outputs_json = "", ""

                flight = self.__run_single_simulation()
                inputs_json = self.__evaluate_flight_inputs(sim_monitor.count)
                outputs_json = self.__evaluate_flight_outputs(flight, sim_monitor.count)

                with open(self.input_file, "a", encoding="utf-8") as f:
                    f.write(inputs_json)
                with open(self.output_file, "a", encoding="utf-8") as f:
                    f.write(outputs_json)

                sim_monitor.print_update_status()

            sim_monitor.print_final_status()

        except KeyboardInterrupt:
            _SimMonitor.reprint("Keyboard Interrupt, files saved.")
            with open(self._error_file, "a", encoding="utf-8") as f:
                f.write(inputs_json)

        except Exception as error:
            _SimMonitor.reprint(f"Error on iteration {sim_monitor.count}: {error}")
            with open(self._error_file, "a", encoding="utf-8") as f:
                f.write(inputs_json)
            raise error

    def __run_in_parallel(self, n_workers=None):
        """
        Runs the monte carlo simulation in parallel.

        Parameters
        ----------
        n_workers: int, optional
            Number of workers to be used. If None, the number of workers
            will be equal to the number of CPUs available. Default is None.

        Returns
        -------
        None
        """
        n_workers = self.__validate_number_of_workers(n_workers)

        _SimMonitor.reprint(f"Running Monte Carlo simulation with {n_workers} workers.")

        multiprocess, managers = _import_multiprocess()

        with _create_multiprocess_manager(multiprocess, managers) as manager:
            mutex = manager.Lock()
            simulation_error_event = manager.Event()
            sim_monitor = manager._SimMonitor(
                initial_count=self._initial_sim_idx,
                n_simulations=self.number_of_simulations,
                start_time=time(),
            )

            processes = []
            seeds = np.random.SeedSequence().spawn(n_workers)

            for seed in seeds:
                sim_producer = multiprocess.Process(
                    target=self.__sim_producer,
                    args=(
                        seed,
                        sim_monitor,
                        mutex,
                        simulation_error_event,
                    ),
                )
                processes.append(sim_producer)
                sim_producer.start()

            try:
                for sim_producer in processes:
                    sim_producer.join()

                # Handle error from the child processes
                if simulation_error_event.is_set():
                    raise RuntimeError(
                        "An error occurred during the simulation. \n"
                        f"Check the logs and error file {self.error_file} "
                        "for more information."
                    )

                sim_monitor.print_final_status()

            # Handle error from the main process
            # pylint: disable=broad-except
            except (Exception, KeyboardInterrupt) as error:
                simulation_error_event.set()

                for sim_producer in processes:
                    sim_producer.join()

                if not isinstance(error, KeyboardInterrupt):
                    raise error

    def __validate_number_of_workers(self, n_workers):
        if n_workers is None or n_workers > os.cpu_count():
            n_workers = os.cpu_count()

        if n_workers < 2:
            raise ValueError("Number of workers must be at least 2 for parallel mode.")
        return n_workers

    def __sim_producer(self, seed, sim_monitor, mutex, error_event):  # pylint: disable=too-many-statements
        """Simulation producer to be used in parallel by multiprocessing.

        Parameters
        ----------
        seed : int
            The seed to set the random number generator.
        sim_monitor : _SimMonitor
            The simulation monitor object to keep track of the simulations.
        mutex : multiprocess.Lock
            The mutex to lock access to critical regions.
        error_event : multiprocess.Event
            Event signaling an error occurred during the simulation.
        """
        try:
            # Ensure Processes generate different random numbers
            self.environment._set_stochastic(seed)
            self.rocket._set_stochastic(seed)
            self.flight._set_stochastic(seed)

            while sim_monitor.keep_simulating():
                sim_idx = sim_monitor.increment() - 1
                inputs_json, outputs_json = "", ""

                flight = self.__run_single_simulation()
                inputs_json = self.__evaluate_flight_inputs(sim_idx)
                outputs_json = self.__evaluate_flight_outputs(flight, sim_idx)

                try:
                    mutex.acquire()
                    if error_event.is_set():
                        sim_monitor.reprint(
                            "Simulation Interrupt, files from simulation "
                            f"{sim_idx} saved."
                        )
                        with open(self.error_file, "a", encoding="utf-8") as f:
                            f.write(inputs_json)

                        break

                    with open(self.input_file, "a", encoding="utf-8") as f:
                        f.write(inputs_json)
                    with open(self.output_file, "a", encoding="utf-8") as f:
                        f.write(outputs_json)

                    sim_monitor.print_update_status()
                finally:
                    mutex.release()

        except Exception:  # pylint: disable=broad-except
            mutex.acquire()
            with open(self.error_file, "a", encoding="utf-8") as f:
                f.write(inputs_json)

            sim_monitor.reprint(f"Error on iteration {sim_idx}:")
            sim_monitor.reprint(traceback.format_exc())
            error_event.set()
            mutex.release()

    def __run_single_simulation(self):
        """Runs a single simulation and returns the inputs and outputs.

        Returns
        -------
        Flight
            The flight object of the simulation.
        """
        return Flight(
            rocket=self.rocket.create_object(),
            environment=self.environment.create_object(),
            rail_length=self.flight._randomize_rail_length(),
            inclination=self.flight._randomize_inclination(),
            heading=self.flight._randomize_heading(),
            initial_solution=self.flight.initial_solution,
            terminate_on_apogee=self.flight.terminate_on_apogee,
            time_overshoot=self.flight.time_overshoot,
        )

    def __evaluate_flight_inputs(self, sim_idx):
        """Evaluates the inputs of a single flight simulation.

        Parameters
        ----------
        sim_idx : int
            The index of the simulation.

        Returns
        -------
        str
            A JSON compatible dictionary with the inputs of the simulation.
        """
        inputs_dict = dict(
            item
            for d in [
                self.environment.last_rnd_dict,
                self.rocket.last_rnd_dict,
                self.flight.last_rnd_dict,
            ]
            for item in d.items()
        )
        inputs_dict["index"] = sim_idx
        return (
            json.dumps(inputs_dict, cls=RocketPyEncoder, **self._export_config) + "\n"
        )

    def __evaluate_flight_outputs(self, flight, sim_idx):
        """Evaluates the outputs of a single flight simulation.

        Parameters
        ----------
        flight : Flight
            The flight object to be evaluated.
        sim_idx : int
            The index of the simulation.

        Returns
        -------
        str
            A JSON compatible dictionary with the outputs of the simulation.
        """
        outputs_dict = {
            export_item: getattr(flight, export_item)
            for export_item in self.export_list
        }
        outputs_dict["index"] = sim_idx

        if self.data_collector is not None:
            additional_exports = {}
            for key, callback in self.data_collector.items():
                try:
                    additional_exports[key] = callback(flight)
                except Exception as e:
                    raise ValueError(
                        f"An error was encountered running 'data_collector' callback {key}. "
                    ) from e
            outputs_dict = outputs_dict | additional_exports

        return (
            json.dumps(outputs_dict, cls=RocketPyEncoder, **self._export_config) + "\n"
        )

    def __terminate_simulation(self):
        """
        Terminates the simulation, closes the files and prints the results.

        Returns
        -------
        None
        """
        # resave the files on self and calculate post simulation attributes
        self.input_file = self._input_file
        self.output_file = self._output_file
        self.error_file = self._error_file

        _SimMonitor.reprint(f"Results saved to {self._output_file}")

    def __check_export_list(self, export_list):
        """
        Checks if the export_list is valid and returns a valid list. If no
        export_list is provided, the standard list is used.

        Parameters
        ----------
        export_list : list
            The list of variables to export. If None, the default list will be
            used. Default is None.

        Returns
        -------
        list
            Validated export list.
        """
        standard_output = set(
            {
                "apogee",
                "apogee_time",
                "apogee_x",
                "apogee_y",
                "t_final",
                "x_impact",
                "y_impact",
                "impact_velocity",
                "initial_stability_margin",
                "out_of_rail_stability_margin",
                "out_of_rail_time",
                "out_of_rail_velocity",
                "max_mach_number",
                "frontal_surface_wind",
                "lateral_surface_wind",
            }
        )
        # NOTE: this list needs to be updated with Flight numerical properties
        #       example: You added the property 'inclination' to Flight.
        #       But don't add other types.
        can_be_exported = set(
            {
                "inclination",
                "heading",
                "effective1rl",
                "effective2rl",
                "out_of_rail_time",
                "out_of_rail_time_index",
                "out_of_rail_state",
                "out_of_rail_velocity",
                "rail_button1_normal_force",
                "max_rail_button1_normal_force",
                "rail_button1_shear_force",
                "max_rail_button1_shear_force",
                "rail_button2_normal_force",
                "max_rail_button2_normal_force",
                "rail_button2_shear_force",
                "max_rail_button2_shear_force",
                "out_of_rail_static_margin",
                "apogee_state",
                "apogee_time",
                "apogee_x",
                "apogee_y",
                "apogee",
                "x_impact",
                "y_impact",
                "z_impact",
                "impact_velocity",
                "impact_state",
                "parachute_events",
                "apogee_freestream_speed",
                "final_static_margin",
                "frontal_surface_wind",
                "initial_static_margin",
                "lateral_surface_wind",
                "max_acceleration",
                "max_acceleration_time",
                "max_dynamic_pressure_time",
                "max_dynamic_pressure",
                "max_mach_number_time",
                "max_mach_number",
                "max_reynolds_number_time",
                "max_reynolds_number",
                "max_speed_time",
                "max_speed",
                "max_total_pressure_time",
                "max_total_pressure",
                "t_final",
            }
        )
        if export_list:
            for attr in set(export_list):
                if not isinstance(attr, str):
                    raise TypeError("Variables in export_list must be strings.")

                # Checks if attribute is not valid
                if attr not in can_be_exported:
                    raise ValueError(
                        f"Attribute '{attr}' can not be exported. Check export_list."
                    )
        else:
            # No export list provided, using default list instead.
            export_list = standard_output

        return export_list

    def _check_data_collector(self, data_collector):
        """Check if data collector provided is a valid

        Parameters
        ----------
        data_collector : dict
            A dictionary whose keys are the names of the exported variables
            and the values are callback functions that receive a Flight object
            and returns a value of that variable
        """

        if data_collector is not None:
            if not isinstance(data_collector, dict):
                raise ValueError(
                    "Invalid 'data_collector' argument! "
                    "It must be a dict of callback functions."
                )

            for key, callback in data_collector.items():
                if key in self.export_list:
                    raise ValueError(
                        "Invalid 'data_collector' key! "
                        f"Variable names overwrites 'export_list' key '{key}'."
                    )
                if not callable(callback):
                    raise ValueError(
                        f"Invalid value in 'data_collector' for key '{key}'! "
                        "Values must be python callables (callback functions)."
                    )

    @property
    def input_file(self):
        """String representing the filepath of the input file"""
        return self._input_file

    @input_file.setter
    def input_file(self, value):
        """
        Setter for input_file. Sets/updates inputs_log.

        Parameters
        ----------
        value : str
            The filepath of the input file.

        Returns
        -------
        None
        """
        self._input_file = value
        self.set_inputs_log()

    @property
    def output_file(self):
        """String representing the filepath of the output file"""
        return self._output_file

    @output_file.setter
    def output_file(self, value):
        """
        Setter for output_file. Sets/updates outputs_log, num_of_loaded_sims,
        results, and processed_results.

        Parameters
        ----------
        value : str
            The filepath of the output file.

        Returns
        -------
        None
        """
        self._output_file = value
        self.set_outputs_log()
        self.set_num_of_loaded_sims()
        self.set_results()
        self.set_processed_results()

    @property
    def error_file(self):
        """String representing the filepath of the error file"""
        return self._error_file

    @error_file.setter
    def error_file(self, value):
        """
        Setter for error_file. Sets/updates errors_log.

        Parameters
        ----------
        value : str
            The filepath of the error file.

        Returns
        -------
        None
        """
        self._error_file = value
        self.set_errors_log()

    # Setters for post simulation attributes

    def set_inputs_log(self):
        """
        Sets inputs_log from a file into an attribute for easy access.

        Returns
        -------
        None
        """
        self.inputs_log = []
        with open(self.input_file, mode="r", encoding="utf-8") as rows:
            for line in rows:
                self.inputs_log.append(json.loads(line))

    def set_outputs_log(self):
        """
        Sets outputs_log from a file into an attribute for easy access.

        Returns
        -------
        None
        """
        self.outputs_log = []
        with open(self.output_file, mode="r", encoding="utf-8") as rows:
            for line in rows:
                self.outputs_log.append(json.loads(line))

    def set_errors_log(self):
        """
        Sets errors_log from a file into an attribute for easy access.

        Returns
        -------
        None
        """
        self.errors_log = []
        with open(self.error_file, mode="r", encoding="utf-8") as errors:
            for line in errors:
                self.errors_log.append(json.loads(line))

    def set_num_of_loaded_sims(self):
        """
        Determines the number of simulations loaded from output_file being
        currently used.

        Returns
        -------
        None
        """
        with open(self.output_file, mode="r", encoding="utf-8") as outputs:
            self.num_of_loaded_sims = sum(1 for _ in outputs)

    def set_results(self):
        """
        Monte Carlo results organized in a dictionary where the keys are the
        names of the saved attributes, and the values are lists with all the
        result numbers of the respective attributes. For instance:

            .. code-block:: python

                {
                    'apogee': [1000, 1001, 1002, ...],
                    'max_speed': [100, 101, 102, ...],
                }

        Returns
        -------
        None
        """
        self.results = {}
        for result in self.outputs_log:
            for key, value in result.items():
                if key in self.results:
                    self.results[key].append(value)
                else:
                    self.results[key] = [value]

    def set_processed_results(self):
        """
        Creates a dictionary with the mean and standard deviation of each
        parameter available in the results.

        Returns
        -------
        None
        """
        self.processed_results = {}
        for result, values in self.results.items():
            try:
                mean = np.mean(values)
                stdev = np.std(values)
                self.processed_results[result] = (mean, stdev)
                pi_low = np.quantile(values, 0.025)
                pi_high = np.quantile(values, 0.975)
                median = np.median(values)
            except TypeError:
                mean = None
                stdev = None
                pi_low = None
                pi_high = None
                median = None
            self.processed_results[result] = (mean, median, stdev, pi_low, pi_high)

    # Import methods

    def import_outputs(self, filename=None):
        """
        Import Monte Carlo results from .txt file and save it into a dictionary.

        Parameters
        ----------
        filename : str, optional
            Name or directory path to the file to be imported. If none,
            self.filename will be used.

        Returns
        -------
        None

        Notes
        -----
        Notice that you can import the outputs, inputs, and errors from the a
        file without the need to run simulations. You can use previously saved
        files to process analyze the results or to continue a simulation.
        """
        filepath = filename if filename else self.filename.with_suffix(".outputs.txt")

        try:
            with open(filepath, "r+", encoding="utf-8"):
                self.output_file = filepath
        except FileNotFoundError:
            with open(filepath, "w+", encoding="utf-8"):
                self.output_file = filepath

        _SimMonitor.reprint(
            f"A total of {self.num_of_loaded_sims} simulations results were "
            f"loaded from the following output file: {self.output_file}\n"
        )

    def import_inputs(self, filename=None):
        """
        Import Monte Carlo inputs from .txt file and save it into a dictionary.

        Parameters
        ----------
        filename : str, optional
            Name or directory path to the file to be imported. If none,
            self.filename will be used.

        Returns
        -------
        None
        """
        filepath = filename if filename else self.filename.with_suffix(".inputs.txt")

        try:
            with open(filepath, "r+", encoding="utf-8"):
                self.input_file = filepath
        except FileNotFoundError:
            with open(filepath, "w+", encoding="utf-8"):
                self.input_file = filepath

        _SimMonitor.reprint(f"The following input file was imported: {self.input_file}")

    def import_errors(self, filename=None):
        """
        Import Monte Carlo errors from .txt file and save it into a dictionary.

        Parameters
        ----------
        filename : str, optional
            Name or directory path to the file to be imported. If none,
            self.filename will be used.

        Returns
        -------
        None
        """
        filepath = filename if filename else self.filename.with_suffix(".errors.txt")

        try:
            with open(filepath, "r+", encoding="utf-8"):
                self.error_file = filepath
        except FileNotFoundError:
            with open(filepath, "w+", encoding="utf-8"):
                self.error_file = filepath

        _SimMonitor.reprint(f"The following error file was imported: {self.error_file}")

    def import_results(self, filename=None):
        """
        Import Monte Carlo results from .txt file and save it into a dictionary.

        Parameters
        ----------
        filename : str, optional
            Name or directory path to the file to be imported. If ``None``,
            self.filename will be used.

        Returns
        -------
        None
        """
        self.import_outputs(filename=filename)
        self.import_inputs(filename=filename)
        self.import_errors(filename=filename)

    # Export methods

    def export_ellipses_to_kml(  # pylint: disable=too-many-statements
        self,
        filename,
        origin_lat,
        origin_lon,
        type="all",  # TODO: Don't use "type" as a parameter name, it's a reserved word  # pylint: disable=redefined-builtin
        resolution=100,
        colors=("ffff0000", "ff00ff00"),  # impact, apogee
    ):
        """
        Generates a KML file with the ellipses on the impact point, which can be
        used to visualize the dispersion ellipses on Google Earth.

        Parameters
        ----------
        filename : str
            Name to the KML exported file.
        origin_lat : float
            Latitude coordinate of Ellipses' geometric center, in degrees.
        origin_lon : float
            Longitude coordinate of Ellipses' geometric center, in degrees.
        type : str, optional
            Type of ellipses to be exported. Options are: 'all', 'impact' and
            'apogee'. Default is 'all', it exports both apogee and impact ellipses.
        resolution : int, optional
            Number of points to be used to draw the ellipse. Default is 100. You
            can increase this number to make the ellipse smoother, but it will
            increase the file size. It is recommended to keep it below 1000.
        colors : tuple[str, str], optional
            Colors of the ellipses. Default is ['ffff0000', 'ff00ff00'], which
            are blue and green, respectively. The first element is the color of
            the impact ellipses, and the second element is the color of the
            apogee. The colors are in hexadecimal format (aabbggrr).

        Returns
        -------
        None

        Notes
        -----
        - For further understanding on .kml files, see the official documentation:\
            https://developers.google.com/kml/documentation/kmlreference
        - You can set a pair of origin coordinates different from the launch site\
            to visualize the dispersion as if the rocket was launched from that\
            point. This is useful to visualize the dispersion ellipses in a\
            different location. However, this approach is not accurate for\
            large distances offsets, as the atmospheric conditions may change.
        """
        # TODO: The lat and lon should be optional arguments, we can get it from the env
        # Retrieve monte carlo data por apogee and impact XY position
        if type not in ["all", "impact", "apogee"]:
            raise ValueError("Invalid type. Options are 'all', 'impact' and 'apogee'")

        apogee_x = np.array([])
        apogee_y = np.array([])
        impact_x = np.array([])
        impact_y = np.array([])
        if type in ["all", "apogee"]:
            try:
                apogee_x = np.array(self.results["apogee_x"])
                apogee_y = np.array(self.results["apogee_y"])
            except KeyError as e:
                raise KeyError("No apogee data found. Skipping apogee ellipses.") from e

        if type in ["all", "impact"]:
            try:
                impact_x = np.array(self.results["x_impact"])
                impact_y = np.array(self.results["y_impact"])
            except KeyError as e:
                raise KeyError("No impact data found. Skipping impact ellipses.") from e

        (apogee_ellipses, impact_ellipses) = generate_monte_carlo_ellipses(
            impact_x,
            impact_y,
            apogee_x,
            apogee_y,
        )

        outputs = []

        if type in ["all", "impact"]:
            outputs.extend(
                generate_monte_carlo_ellipses_coordinates(
                    impact_ellipses, origin_lat, origin_lon, resolution=resolution
                )
            )

        if type in ["all", "apogee"]:
            outputs.extend(
                generate_monte_carlo_ellipses_coordinates(
                    apogee_ellipses, origin_lat, origin_lon, resolution=resolution
                )
            )

        if all(isinstance(output, list) for output in outputs):
            kml_data = [
                [(coord[1], coord[0]) for coord in output] for output in outputs
            ]
        else:
            raise ValueError("Each element in outputs must be a list")

        kml = simplekml.Kml()

        for i, points in enumerate(kml_data):
            if i < len(impact_ellipses):
                name = f"Impact Ellipse {i + 1}"
                ellipse_color = colors[0]  # default is blue
            else:
                name = f"Apogee Ellipse {i + 1 - len(impact_ellipses)}"
                ellipse_color = colors[1]  # default is green

            mult_ell = kml.newmultigeometry(name=name)
            mult_ell.newpolygon(
                outerboundaryis=points,
                name=name,
            )
            # Setting ellipse style
            mult_ell.tessellate = 1
            mult_ell.visibility = 1
            mult_ell.style.linestyle.color = ellipse_color
            mult_ell.style.linestyle.width = 3
            mult_ell.style.polystyle.color = simplekml.Color.changealphaint(
                80, ellipse_color
            )

        kml.newpoint(
            name="Launch Pad",
            coords=[(origin_lon, origin_lat)],
            description="Flight initial position",
        )

        kml.save(filename)

    def info(self):
        """
        Print information about the Monte Carlo simulation.

        Returns
        -------
        None
        """
        self.prints.all()

    def all_info(self):
        """
        Print and plot information about the Monte Carlo simulation and its results.

        Returns
        -------
        None
        """
        self.info()
        self.plots.ellipses()
        self.plots.all()

    def compare_info(self, other_monte_carlo):
        """
        Prints the comparison of the information  of the Monte Carlo simulation
        against the information of another Monte Carlo simulation.
        Parameters
        ----------
        other_monte_carlo : MonteCarlo
            MonteCarlo object which the current one will be compared to.
        Returns
        -------
        None
        """
        self.prints.print_comparison(other_monte_carlo)

    def compare_plots(self, other_monte_carlo):
        """
        Plots the comparison of the information of the Monte Carlo simulation
        against the information of another Monte Carlo simulation.
        Parameters
        ----------
        other_monte_carlo : MonteCarlo
            MonteCarlo object which the current one will be compared to.
        Returns
        -------
        None
        """
        self.plots.plot_comparison(other_monte_carlo)

    def compare_ellipses(self, other_monte_carlo, **kwargs):
        """
        Plots the comparison of the ellipses of the Monte Carlo simulation
        against the ellipses of another Monte Carlo simulation.
        Parameters
        ----------
        other_monte_carlo : MonteCarlo
            MonteCarlo object which the current one will be compared to.
        Returns
        -------
        None
        """
        self.plots.ellipses_comparison(other_monte_carlo, **kwargs)


def _import_multiprocess():
    """Import the necessary modules and submodules for the
    multiprocess library.

    Returns
    -------
    tuple
        Tuple containing the imported modules.
    """
    multiprocess = import_optional_dependency("multiprocess")
    managers = import_optional_dependency("multiprocess.managers")

    return multiprocess, managers


def _create_multiprocess_manager(multiprocess, managers):
    """Creates a manager for the multiprocess control of the
    Monte Carlo simulation.

    Parameters
    ----------
    multiprocess : module
        Multiprocess module.
    managers : module
        Managing submodules of the multiprocess module.

    Returns
    -------
    MonteCarloManager
        Subclass of BaseManager with the necessary classes registered.
    """

    class MonteCarloManager(managers.BaseManager):
        """Custom manager for shared objects in the Monte Carlo simulation."""

        def __init__(self):
            super().__init__()
            self.register("Lock", multiprocess.Lock)
            self.register("Queue", multiprocess.Queue)
            self.register("Event", multiprocess.Event)
            self.register("_SimMonitor", _SimMonitor)

    return MonteCarloManager()


class _SimMonitor:
    """Class to monitor the simulation progress and display the status."""

    _last_print_len = 0

    def __init__(self, initial_count, n_simulations, start_time):
        self.initial_count = initial_count
        self.count = initial_count
        self.n_simulations = n_simulations
        self.start_time = start_time
        self.completed_count = 0

    def keep_simulating(self):
        return self.count < self.n_simulations

    def increment(self):
        self.count += 1
        return self.count

    def print_update_status(self):
        """Prints a message on the same line as the previous one and replaces
        the previous message with the new one, deleting the extra characters
        from the previous message. This method increments the completed_count
        to track how many simulations have finished (thread-safe when called
        within a mutex-protected section).

        Returns
        -------
        None
        """
        self.completed_count += 1

        average_time = (time() - self.start_time) / self.completed_count
        remaining = self.n_simulations - self.initial_count - self.completed_count
        estimated_time = int(remaining * average_time)

        msg = f"Iterations completed: {self.completed_count:06d}"
        msg += f" | Average Time per Iteration: {average_time:.3f} s"
        msg += f" | Estimated time left: {estimated_time} s"

        _SimMonitor.reprint(msg, end="\r", flush=True)

    def print_final_status(self):
        """Prints the final status of the simulation."""
        print()
        msg = f"Completed {self.count - self.initial_count} iterations."
        msg += f" In total, {self.count} simulations are exported.\n"
        msg += f"Total wall time: {time() - self.start_time:.1f} s"

        _SimMonitor.reprint(msg, end="\n", flush=True)

    @staticmethod
    def reprint(msg, end="\n", flush=True):
        """
        Prints a message on the same line as the previous one and replaces the
        previous message with the new one, deleting the extra characters from
        the previous message.

        Parameters
        ----------
        msg : str
            Message to be printed.
        end : str, optional
            String appended after the message. Default is a new line.
        flush : bool, optional
            If True, the output is flushed. Default is True.

        Returns
        -------
        None
        """

        padding = ""

        if len(msg) < _SimMonitor._last_print_len:
            padding = " " * (_SimMonitor._last_print_len - len(msg))

        print(msg + padding, end=end, flush=flush)

        _SimMonitor._last_print_len = len(msg)
