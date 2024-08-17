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
import queue
import warnings
from copy import deepcopy
from pathlib import Path
from time import time
import multiprocessing as mp
from multiprocessing.managers import BaseManager

import numpy as np
import simplekml

from rocketpy import Function
from rocketpy._encoders import RocketPyEncoder
from rocketpy.plots.monte_carlo_plots import _MonteCarloPlots
from rocketpy.prints.monte_carlo_prints import _MonteCarloPrints
from rocketpy.simulation.flight import Flight
from rocketpy.tools import (
    generate_monte_carlo_ellipses,
    generate_monte_carlo_ellipses_coordinates,
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

    _last_print_len = 0  # used to print on the same line

    def __init__(
        self,
        filename,
        environment,
        rocket,
        flight,
        export_list=None,
        batch_path=None,
        export_sample_time=0.1,
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
        batch_path : str, optional
            Path to the batch folder to be used in the simulation. Export file
            will be saved in this folder. Default is None.
        export_sample_time : float, optional
            Sample time to downsample the arrays in seconds. Used to automatically
            discretize inputs that contain callable ``rocketpy.Function`` objects.
            Default is 0.1.

        Returns
        -------
        None
        """
        warnings.warn(
            "This class is still under testing and some attributes may be "
            "changed in next versions",
            UserWarning,
        )

        self.filename = filename
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
        self.export_sample_time = export_sample_time

        if batch_path is None:
            self.batch_path = Path.cwd()
        else:
            self.batch_path = Path(batch_path)

        if not os.path.exists(self.batch_path):
            os.makedirs(self.batch_path)

        self.export_list = self.__check_export_list(export_list)

        try:
            self.import_inputs()
        except FileNotFoundError:
            self._input_file = self.batch_path / f"{filename}.inputs.txt"

        try:
            self.import_outputs()
        except FileNotFoundError:
            self._output_file = self.batch_path / f"{filename}.outputs.txt"

        try:
            self.import_errors()
        except FileNotFoundError:
            self._error_file = self.batch_path / f"{filename}.errors.txt"

    # pylint: disable=consider-using-with
    def simulate(
        self,
        number_of_simulations,
        append=False,
        parallel=False,
        n_workers=None,
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
            Default is None.

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
        self.number_of_simulations = number_of_simulations
        self._initial_sim_idx = self.num_of_loaded_sims if append else 0

        # Begin display
        _SimMonitor._reprint("Starting Monte Carlo analysis")

        # Setup files
        self.__setup_files(append)

        # Run simulations
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
            open(self._error_file, open_mode, encoding="utf-8").close()

            if idx_i != idx_o and not append:
                warnings.warn(
                    "Input and output files are not synchronized", UserWarning
                )

        except OSError as error:
            raise OSError(f"Error creating files: {error}") from error

    def __run_in_serial(self):
        """
        Runs the monte carlo simulation in serial mode.

        Parameters
        ----------
        start_index : int
            The index of the first simulation to be run.

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

                inputs_dict, outputs_dict = self.__run_single_simulation(
                    sim_monitor.count,
                )

                self.__export_flight_data(inputs_dict, outputs_dict)

                sim_monitor.print_update_status(sim_monitor.count)

            sim_monitor.print_final_status()

        except KeyboardInterrupt:
            _SimMonitor._reprint("Keyboard Interrupt, files saved.")
            with open(self._error_file, "a", encoding="utf-8") as file:
                file.write(json.dumps(inputs_dict, cls=RocketPyEncoder) + "\n")

        except Exception as error:
            _SimMonitor._reprint(
                f"Error on iteration {self.__sim_monitor.count}: {error}"
            )
            with open(self._error_file, "a", encoding="utf-8") as file:
                file.write(json.dumps(inputs_dict, cls=RocketPyEncoder) + "\n")
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
        if n_workers is None or n_workers > os.cpu_count():
            n_workers = os.cpu_count()

        if n_workers < 2:
            raise ValueError("Number of workers must be at least 2 for parallel mode.")

        with MonteCarloManager() as manager:
            export_queue = manager.Queue()
            mutex = manager.Lock()
            consumer_stop_event = manager.Event()

            sim_monitor = manager._SimMonitor(
                initial_count=self._initial_sim_idx,
                n_simulations=self.number_of_simulations,
                start_time=time(),
            )

            processes = []
            seeds = np.random.SeedSequence().spawn(n_workers - 1)

            for seed in seeds:
                sim_producer = mp.Process(
                    target=self.__sim_producer,
                    args=(
                        sim_monitor,
                        export_queue,
                        mutex,
                        seed,
                    ),
                )
                processes.append(sim_producer)

            for sim_producer in processes:
                sim_producer.start()

            sim_consumer = mp.Process(
                target=self.__sim_consumer,
                args=(
                    export_queue,
                    mutex,
                    consumer_stop_event,
                ),
            )

            sim_consumer.start()

            for sim_producer in processes:
                sim_producer.join()

            consumer_stop_event.set()

            sim_consumer.join()

            sim_monitor.print_final_status()

    def __sim_producer(self, sim_monitor, export_queue, mutex, seed):
        """Simulation producer to be used in parallel by multiprocessing.

        Parameters
        ----------
        sim_monitor : _SimMonitor
            The simulation monitor object to keep track of the simulations.
        export_queue : multiprocess.Queue
            The queue to export the results.
        error_file : str
            The file to write the errors.
        mutex : multiprocess.Lock
            The mutex to lock access to critical regions.
        seed : int
            The seed to set the random number generator.
        """
        try:
            while sim_monitor.keep_simulating():
                sim_idx = sim_monitor.increment() - 1

                self.environment._set_stochastic(seed)
                self.rocket._set_stochastic(seed)
                self.flight._set_stochastic(seed)

                inputs_dict, outputs_dict = self.__run_single_simulation(sim_idx)

                export_queue.put((inputs_dict, outputs_dict))

                mutex.acquire()
                sim_monitor.print_update_status(sim_idx)
                mutex.release()

        except Exception as error:
            mutex.acquire()
            with open(self.error_file, "a", encoding="utf-8") as file:
                file.write(json.dumps(inputs_dict, cls=RocketPyEncoder) + "\n")

            _SimMonitor._reprint(f"Error on iteration {sim_idx}: {error}")
            mutex.release()

            raise error

    def __sim_consumer(
        self,
        export_queue,
        mutex,
        stop_event,
    ):
        """Simulation consumer to be used in parallel by multiprocessing.
        It consumes the results from the queue and writes them to the files.
        If no results are received for 30 seconds, a TimeoutError is raised.

        Parameters
        ----------
        export_queue : multiprocess.Queue
            The queue to export the results.
        inputs_file : str
            The file path to write the inputs.
        outputs_file : str
            The file path to write the outputs.
        mutex : multiprocess.Lock
            The mutex to lock access to critical regions.
        stop_event : multiprocess.Event
            The event indicating that the simulations are done.
        """
        trials = 0
        while not stop_event.is_set():
            try:
                mutex.acquire()
                inputs_dict, outputs_dict = export_queue.get(timeout=3)

                self.__export_flight_data(inputs_dict, outputs_dict)

            except queue.Empty as exc:
                trials += 1

                if trials > 10:
                    raise TimeoutError(
                        "No simulations were received for 30 seconds."
                    ) from exc

            finally:
                mutex.release()

    def __run_single_simulation(self, sim_idx):
        """Runs a single simulation and returns the inputs and outputs.

        Parameters
        ----------
        sim_idx : int
            The index of the simulation.
        """
        monte_carlo_flight = Flight(
            rocket=self.rocket.create_object(),
            environment=self.environment.create_object(),
            rail_length=self.flight._randomize_rail_length(),
            inclination=self.flight._randomize_inclination(),
            heading=self.flight._randomize_heading(),
            initial_solution=self.flight.initial_solution,
            terminate_on_apogee=self.flight.terminate_on_apogee,
        )

        inputs_dict = dict(
            item
            for d in [
                self.environment.last_rnd_dict,
                self.rocket.last_rnd_dict,
                self.flight.last_rnd_dict,
            ]
            for item in d.items()
        )
        inputs_dict["idx"] = sim_idx

        inputs_dict = MonteCarlo._prepare_export_data(
            inputs_dict, self.export_sample_time, remove_functions=True
        )

        outputs_dict = {
            export_item: getattr(monte_carlo_flight, export_item)
            for export_item in self.export_list
        }

        return inputs_dict, outputs_dict

    def __export_flight_data(self, inputs_dict, outputs_dict):
        """
        Exports the flight data to the respective files.

        Parameters
        ----------
        inputs_dict : dict
            Dictionary with the inputs of the simulation.
        outputs_dict : dict
            Dictionary with the outputs of the simulation.

        Returns
        -------
        None
        """
        with open(self.input_file, "a", encoding="utf-8") as file:
            file.write(json.dumps(inputs_dict, cls=RocketPyEncoder) + "\n")
        with open(self.output_file, "a", encoding="utf-8") as file:
            file.write(json.dumps(outputs_dict, cls=RocketPyEncoder) + "\n")

    def __terminate_simulation(self):
        """
        Terminates the simulation, closes the files and prints the results.

        Returns
        -------
        None
        """
        # resave the files on self and calculate post simulation attributes
        self.input_file = self.batch_path / f"{self.filename}.inputs.txt"
        self.output_file = self.batch_path / f"{self.filename}.outputs.txt"
        self.error_file = self.batch_path / f"{self.filename}.errors.txt"

        _SimMonitor._reprint(f"Results saved to {self._output_file}")

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

    # Properties and setters

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
            mean = np.mean(values)
            stdev = np.std(values)
            self.processed_results[result] = (mean, stdev)

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
        filepath = filename if filename else self.filename

        try:
            with open(f"{filepath}.outputs.txt", "r+", encoding="utf-8"):
                self.output_file = f"{filepath}.outputs.txt"
        except FileNotFoundError:
            with open(filepath, "r+", encoding="utf-8"):
                self.output_file = filepath

        _SimMonitor._reprint(
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
        filepath = filename if filename else self.filename

        try:
            with open(f"{filepath}.inputs.txt", "r+", encoding="utf-8"):
                self.input_file = f"{filepath}.inputs.txt"
        except FileNotFoundError:
            with open(filepath, "r+", encoding="utf-8"):
                self.input_file = filepath

        _SimMonitor._reprint(
            f"The following input file was imported: {self.input_file}"
        )

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
        filepath = filename if filename else self.filename

        try:
            with open(f"{filepath}.errors.txt", "r+", encoding="utf-8"):
                self.error_file = f"{filepath}.errors.txt"
        except FileNotFoundError:
            with open(filepath, "r+", encoding="utf-8"):
                self.error_file = filepath

        _SimMonitor._reprint(
            f"The following error file was imported: {self.error_file}"
        )

    def import_results(self, filename=None):
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
        """
        filepath = filename if filename else self.filename

        self.import_outputs(filename=filepath)
        self.import_inputs(filename=filepath)
        self.import_errors(filename=filepath)

    # Export methods

    def export_ellipses_to_kml(  # pylint: disable=too-many-statements
        self,
        filename,
        origin_lat,
        origin_lon,
        type="all",  # TODO: Don't use "type" as a parameter name, it's a reserved word  # pylint: disable=redefined-builtin
        resolution=100,
        color="ff0000ff",
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
        color : str, optional
            Color of the ellipse. Default is 'ff0000ff', which is red. Kml files
            use an 8 digit HEX color format, see its docs.

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
        (
            impact_ellipses,
            apogee_ellipses,
            *_,
        ) = generate_monte_carlo_ellipses(self.results)
        outputs = []

        if type in ["all", "impact"]:
            outputs = outputs + generate_monte_carlo_ellipses_coordinates(
                impact_ellipses, origin_lat, origin_lon, resolution=resolution
            )

        if type in ["all", "apogee"]:
            outputs = outputs + generate_monte_carlo_ellipses_coordinates(
                apogee_ellipses, origin_lat, origin_lon, resolution=resolution
            )

        # TODO: Non-iterable value output is used in an iterating context PylintE1133:
        kml_data = [[(coord[1], coord[0]) for coord in output] for output in outputs]

        kml = simplekml.Kml()

        for i in range(len(outputs)):
            if (type == "all" and i < 3) or (type == "impact"):
                ellipse_name = "Impact \u03C3" + str(i + 1)
            elif type == "all" and i >= 3:
                ellipse_name = "Apogee \u03C3" + str(i - 2)
            else:
                ellipse_name = "Apogee \u03C3" + str(i + 1)

            mult_ell = kml.newmultigeometry(name=ellipse_name)
            mult_ell.newpolygon(
                outerboundaryis=kml_data[i],
                name="Ellipse " + str(i),
            )
            # Setting ellipse style
            mult_ell.tessellate = 1
            mult_ell.visibility = 1
            mult_ell.style.linestyle.color = color
            mult_ell.style.linestyle.width = 3
            mult_ell.style.polystyle.color = simplekml.Color.changealphaint(
                100, simplekml.Color.blue
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

    @staticmethod
    def _time_function_serializer(function_object, t_range=None, sample_time=None):
        """
        Method to serialize a Function object into a numpy array. If the function is
        callable, it will be discretized. If the downsample_time is specified, the
        function will be downsampled. This serializer should not be used for
        function that are not time dependent.

        Parameters
        ----------
        function_object : Function
            Function object to be serialized.
        t_range : tuple, optional
            Tuple with the initial and final time of the function. Default is None.
        sample_time : float, optional
            Time interval between samples. Default is None.

        Returns
        -------
        np.ndarray
            Serialized function as a numpy array.
        """
        func = deepcopy(function_object)

        # Discretize the function if it is callable
        if callable(function_object.source):
            if t_range is not None:
                func.set_discrete(*t_range, (t_range[1] - t_range[0]) / sample_time)
            else:
                raise ValueError("t_range must be specified for callable functions")

        source = func.get_source()

        # Ensure the downsampling is applied
        if sample_time is not None:
            t0 = source[0, 0]
            tf = source[-1, 0]
            t = np.arange(t0, tf, sample_time)
            y = func(t)
            source = np.column_stack((t, y))

        return source

    @staticmethod
    def _prepare_export_data(obj, sample_time=0.1, remove_functions=False):
        """
        Inspects the attributes of an object and returns a dictionary of its
        attributes.

        Parameters
        ----------
        obj : object
            The object whose attributes are to be inspected.
        sample_time : float, optional
            Time interval between samples. Default is 0.1.
        remove_functions : bool, optional
            If True, the Function objects will not be serialized. Default is False.

        Returns
        -------
        dict
            A dictionary containing the attributes of the object.
            If the attribute is a Function object, it is serialized using
            `function_serializer`. If the attribute is a dictionary, it is recursively
            inspected using `inspect_object_attributes`. Only includes attributes that
            are integers, floats, dictionaries or Function objects.
        """
        result = {}

        if isinstance(obj, dict):
            # Iterate through all attributes of the object
            for attr_name, attr_value in obj.items():
                # Filter out private attributes and check if the attribute is of a type we are interested in
                if not attr_name.startswith('_') and isinstance(
                    attr_value, (int, float, dict, Function)
                ):
                    if isinstance(attr_value, (int, float)):
                        result[attr_name] = attr_value

                    elif isinstance(attr_value, dict):
                        result[attr_name] = MonteCarlo._prepare_export_data(
                            attr_value, sample_time
                        )

                    elif not remove_functions and isinstance(attr_value, Function):
                        # Serialize the Functions
                        result[attr_name] = MonteCarlo._time_function_serializer(
                            attr_value, None, sample_time
                        )
        else:
            # Iterate through all attributes of the object
            for attr_name in dir(obj):
                attr_value = getattr(obj, attr_name)

                # Filter out private attributes and check if the attribute is of a type we are interested in
                if not attr_name.startswith('_') and isinstance(
                    attr_value, (int, float, dict, Function)
                ):
                    if isinstance(attr_value, (int, float)):
                        result[attr_name] = attr_value

                    elif isinstance(attr_value, dict):
                        result[attr_name] = MonteCarlo._prepare_export_data(
                            attr_value, sample_time
                        )

                    elif not remove_functions and isinstance(attr_value, Function):
                        # Serialize the Functions
                        result[attr_name] = MonteCarlo._time_function_serializer(
                            attr_value, None, sample_time
                        )

        return result


class MonteCarloManager(BaseManager):
    """Custom manager for shared objects in the Monte Carlo simulation."""

    def __init__(self):
        super().__init__()
        self.register('Lock', mp.Lock)
        self.register('Queue', mp.Queue)
        self.register('Event', mp.Event)
        self.register('_SimMonitor', _SimMonitor)


class _SimMonitor:
    """Class to monitor the simulation progress and display the status."""

    _last_print_len = 0

    def __init__(self, initial_count, n_simulations, start_time):
        self.initial_count = initial_count
        self.count = initial_count
        self.n_simulations = n_simulations
        self.start_time = start_time

    def keep_simulating(self):
        return self.count < self.n_simulations

    def increment(self):
        self.count += 1
        return self.count

    def print_update_status(self, sim_idx):
        """Prints a message on the same line as the previous one and replaces
        the previous message with the new one, deleting the extra characters
        from the previous message.

        Parameters
        ----------
        sim_idx : int
            Index of the current simulation.

        Returns
        -------
        None
        """
        average_time = (time() - self.start_time) / (self.count - self.initial_count)
        estimated_time = int((self.n_simulations - self.count) * average_time)

        msg = f"Current iteration: {sim_idx:06d}"
        msg += f" | Average Time per Iteration: {average_time:.3f} s"
        msg += f" | Estimated time left: {estimated_time} s"

        _SimMonitor._reprint(msg, end="\r", flush=True)

    def print_final_status(self):
        """Prints the final status of the simulation."""
        print()
        performed_sims = self.count - self.initial_count
        msg = f"Completed {performed_sims} iterations."
        msg += f" In total, {self.n_simulations} simulations are exported.\n"
        msg += f"Total wall time: {time() - self.start_time:.1f} s"

        _SimMonitor._reprint(msg, end="\n", flush=True)

    @staticmethod
    def _reprint(msg, end="\n", flush=False):
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
            If True, the output is flushed. Default is False.

        Returns
        -------
        None
        """
        padding = ""

        if len(msg) < _SimMonitor._last_print_len:
            padding = " " * (_SimMonitor._last_print_len - len(msg))

        _SimMonitor._last_print_len = len(msg)

        print(msg + padding, end=end, flush=flush)
