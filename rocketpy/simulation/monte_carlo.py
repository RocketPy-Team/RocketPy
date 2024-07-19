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

import ctypes
import json
import os
import pickle
from copy import deepcopy
from pathlib import Path
import warnings
from time import process_time, time

import h5py
import numpy as np
import simplekml
from multiprocess import Event, Lock, Process, Semaphore, shared_memory
from multiprocess.managers import BaseManager

from rocketpy import Function
from rocketpy._encoders import RocketPyEncoder
from rocketpy.plots.monte_carlo_plots import _MonteCarloPlots
from rocketpy.prints.monte_carlo_prints import _MonteCarloPrints
from rocketpy.simulation.flight import Flight
from rocketpy.stochastic import (
    StochasticEnvironment,
    StochasticFlight,
    StochasticRocket,
)
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

    def __init__(
        self,
        filename,
        environment,
        rocket,
        flight,
        export_list=None,
        batch_path=None,
        export_sample_time=0.1,
    ): # pylint: disable=too-many-statements
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
            Sample time to downsample the arrays in seconds. Default is 0.1.

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
        self._inputs_dict = {}
        self._last_print_len = 0  # used to print on the same line
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
        light_mode=False,
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
        light_mode : bool, optional
            If True, only variables from the export_list will be saved to
            the output file as a .txt file. If False, all variables will be
            saved to the output file as a .h5 file. Default is False.
        parallel : bool, optional
            If True, the simulations will be run in parallel. Default is False.
        n_workers : int, optional
            Number of workers to be used. If None, the number of workers
            will be equal to the number of CPUs available. Default is None.

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
        # initialize counters
        self.number_of_simulations = number_of_simulations
        self.__iteration_count = self.num_of_loaded_sims if append else 0
        self.__start_time = time()
        self.__start_cpu_time = process_time()

        # Begin display
        print("Starting Monte Carlo analysis", end="\r")

        # Run simulations
        if parallel:
            self.__run_in_parallel(append, light_mode=light_mode, n_workers=n_workers)
        else:
            self.__run_in_serial(append, light_mode=light_mode)


    def __run_in_serial(self, append, light_mode):
        """
        Runs the monte carlo simulation in serial mode.

        Parameters
        ----------
        append: bool
            If True, the results will be appended to the existing files. If
            False, the files will be overwritten.
        light_mode: bool
            If True, only variables from the export_list will be saved to
            the output file as a .txt file. If False, all variables will be
            saved to the output file as a .h5 file.

        Returns
        -------
        None
        """
        # Create data files for inputs, outputs and error logging
        open_mode = "a" if append else "w"

        # Open files
        if light_mode:
            # open files in write/append mode
            input_file = open(self._input_file, open_mode, encoding="utf-8")
            output_file = open(self._output_file, open_mode, encoding="utf-8")
            error_file = open(self._error_file, open_mode, encoding="utf-8")

        else:
            input_file = h5py.File(Path(self._input_file).with_suffix(".h5"), open_mode)
            output_file = h5py.File(
                Path(self._output_file).with_suffix(".h5"), open_mode
            )
            error_file = open(self._error_file, open_mode, encoding="utf-8")

        idx_i = self.__get_initial_sim_idx(
            input_file, append=append, light_mode=light_mode
        )
        idx_o = self.__get_initial_sim_idx(
            output_file, append=append, light_mode=light_mode
        )

        if idx_i != idx_o:
            raise ValueError(
                "Input and output files are not synchronized. Append mode is not available."
            )

        # Run simulations
        try:
            while self.__iteration_count < self.number_of_simulations:
                self.__run_single_simulation(
                    self.iteration_count + idx_i,
                    input_file,
                    output_file,
                    light_mode=light_mode,
                )

        except KeyboardInterrupt:
            print("Keyboard Interrupt, files saved.")
            error_file.write(json.dumps(self._inputs_dict, cls=RocketPyEncoder) + "\n")
            self.__close_files(input_file, output_file, error_file)

        except Exception as error:
            print(f"Error on iteration {self.__iteration_count}: {error}")
            error_file.write(json.dumps(self._inputs_dict, cls=RocketPyEncoder) + "\n")
            self.__close_files(input_file, output_file, error_file)
            raise error

        self.__terminate_simulation(
            input_file, output_file, error_file, light_mode=light_mode
        )

    def __run_in_parallel(self, append, light_mode, n_workers=None):
        """
        Runs the monte carlo simulation in parallel.

        Parameters
        ----------
        append: bool
            If True, the results will be appended to the existing files. If
            False, the files will be overwritten.
        light_mode: bool
            If True, only variables from the export_list will be saved to
            the output file as a .txt file. If False, all variables will be
            saved to the output file as a .h5 file.
        n_workers: int, optional
            Number of workers to be used. If None, the number of workers
            will be equal to the number of CPUs available. Default is None.

        Returns
        -------
        None
        """
        parallel_start_time = time()
        processes = []

        if (
            n_workers is None or n_workers > os.cpu_count()
        ):  # leave 2 cores for the writer workers
            n_workers = os.cpu_count()

        if n_workers < 3:
            raise ValueError("Number of workers must be at least 3 for parallel mode.")

        # get the size of the serialized dictionary
        inputs_size, results_size = self.__get_export_size(light_mode)

        # add safety margin to the buffer size
        inputs_size += 1024
        results_size += 1024

        # calculate the number of simulations that can be stored in memory
        n_sim_memory = max(n_workers - 2, 2)  # at least a double buffer

        # initialize shared memory buffer
        shared_inputs_buffer = shared_memory.SharedMemory(
            create=True, size=inputs_size * n_sim_memory, name="shared_inputs"
        )
        shared_results_buffer = shared_memory.SharedMemory(
            create=True, size=results_size * n_sim_memory, name="shared_results"
        )

        try:
            with MonteCarloManager() as manager:
                # initialize queue
                errors_lock = manager.Lock()

                # initialize semaphores to control the shared memory buffer
                # input file semaphores
                go_write_inputs = [
                    manager.Semaphore(value=1) for _ in range(n_sim_memory)
                ]
                go_read_inputs = [
                    manager.Semaphore(value=1) for _ in range(n_sim_memory)
                ]

                # output file semaphores
                go_write_results = [
                    manager.Semaphore(value=1) for _ in range(n_sim_memory)
                ]
                go_read_results = [
                    manager.Semaphore(value=1) for _ in range(n_sim_memory)
                ]

                # acquire all read semaphores to make sure the readers will wait for data
                for sem in go_read_inputs:
                    sem.acquire()

                for sem in go_read_results:
                    sem.acquire()

                # Initialize write file
                open_mode = "a" if append else "w"

                file_paths = {
                    "input_file": Path(self._input_file),
                    "output_file": Path(self._output_file),
                    "error_file": Path(self._error_file),
                    "export_list": self.export_list,
                }

                # Initialize files
                if light_mode:
                    # open files in write/append mode
                    with open(self._input_file, mode=open_mode) as f:
                        pass

                    with open(self._output_file, mode=open_mode) as f:
                        pass

                    # get the initial simulation index - read mode is required
                    with open(self._input_file, mode='r') as f:
                        idx_i = self.__get_initial_sim_idx(
                            f, append=append, light_mode=light_mode
                        )

                    with open(self._output_file, mode='r') as f:
                        idx_o = self.__get_initial_sim_idx(
                            f, append=append, light_mode=light_mode
                        )

                else:
                    # Change file extensions to .h5
                    file_paths["input_file"] = file_paths["input_file"].with_suffix(
                        ".h5"
                    )
                    file_paths["output_file"] = file_paths["output_file"].with_suffix(
                        ".h5"
                    )
                    file_paths["error_file"] = file_paths["error_file"].with_suffix(
                        ".h5"
                    )

                    # Initialize files and get initial simulation index
                    with h5py.File(file_paths["input_file"], open_mode) as f:
                        idx_i = self.__get_initial_sim_idx(
                            f, append=append, light_mode=light_mode
                        )
                    with h5py.File(file_paths["output_file"], open_mode) as f:
                        idx_o = self.__get_initial_sim_idx(
                            f, append=append, light_mode=light_mode
                        )

                if idx_i != idx_o:
                    raise ValueError(
                        "Input and output files are not synchronized. Append mode is not available."
                    )

                # Initialize error file - always a .txt file
                with open(self._error_file, mode=open_mode) as _:
                    pass  # initialize file

                # Initialize simulation counter
                sim_counter = manager.SimCounter(
                    idx_i, self.number_of_simulations, parallel_start_time
                )

                print("\nStarting monte carlo analysis", end="\r")
                print(f"Number of simulations: {self.number_of_simulations}")

                # Creates n_workers processes then starts them
                for i in range(n_workers - 2):  # leave 2 cores for the writer workers
                    p = Process(
                        target=self.__run_simulation_worker,
                        args=(
                            light_mode,
                            file_paths,
                            self.environment,
                            self.rocket,
                            self.flight,
                            sim_counter,
                            errors_lock,
                            go_write_inputs,
                            go_write_results,
                            go_read_inputs,
                            go_read_results,
                            shared_inputs_buffer.name,
                            shared_results_buffer.name,
                            inputs_size,
                            results_size,
                            n_sim_memory,
                            self.export_sample_time,
                        ),
                    )
                    processes.append(p)

                # Starts all the processes
                for p in processes:
                    p.start()

                # create writer workers
                input_writer_stop_event = manager.Event()
                results_writer_stop_event = manager.Event()

                input_writer = Process(
                    target=self._write_data_worker,
                    args=(
                        file_paths["input_file"],
                        go_write_inputs,
                        go_read_inputs,
                        shared_inputs_buffer.name,
                        inputs_size,
                        input_writer_stop_event,
                        n_sim_memory,
                        light_mode,
                    ),
                )

                results_writer = Process(
                    target=self._write_data_worker,
                    args=(
                        file_paths["output_file"],
                        go_write_results,
                        go_read_results,
                        shared_results_buffer.name,
                        results_size,
                        results_writer_stop_event,
                        n_sim_memory,
                        light_mode,
                    ),
                )

                # start the writer workers
                input_writer.start()
                results_writer.start()

                # Joins all the processes
                for p in processes:
                    p.join()

                print("Joining writer workers.")
                # stop the writer workers
                input_writer_stop_event.set()
                results_writer_stop_event.set()

                print("Waiting for writer workers to join.")
                # join the writer workers
                input_writer.join()
                results_writer.join()

                self.number_of_simulations = sim_counter.get_count()

                parallel_end = time()

                print("-" * 80 + "\nAll workers joined, simulation complete.")
                print(
                    f"In total, {sim_counter.get_count() - idx_i} simulations were performed."
                )
                print(
                    "Simulation took",
                    parallel_end - parallel_start_time,
                    "seconds to run.",
                )

        finally:
            # ensure shared memory is realeased
            shared_inputs_buffer.close()
            shared_results_buffer.close()
            shared_inputs_buffer.unlink()
            shared_results_buffer.unlink()

    @staticmethod
    def __run_simulation_worker(
        light_mode,
        file_paths,
        sto_env,
        sto_rocket,
        sto_flight,
        sim_counter,
        errors_lock,
        go_write_inputs,
        go_write_results,
        go_read_inputs,
        go_read_results,
        shared_inputs_name,
        shared_results_name,
        inputs_size,
        results_size,
        n_sim_memory,
        export_sample_time,
    ):
        """
        Runs a single simulation worker.

        Parameters
        ----------
        light_mode : bool
            If True, only variables from the export_list will be saved to
            the output file as a .txt file. If False, all variables will be
            saved to the output file as a .h5 file.
        file_paths : dict
            Dictionary with the file paths.
        sto_env : StochasticEnvironment
            Stochastic environment object.
        sto_rocket : StochasticRocket
            Stochastic rocket object.
        sto_flight : StochasticFlight
            Stochastic flight object.
        sim_counter : SimCounter
            Simulation counter object.
        errors_lock : Lock
            Lock to write errors to the error file.
        go_write_inputs : list
            List of semaphores to write the inputs.
        go_write_results : list
            List of semaphores to write the results.
        go_read_inputs : list
            List of semaphores to read the inputs.
        go_read_results : list
            List of semaphores to read the results.
        shared_inputs_name : str
            Name of the shared memory buffer for the inputs.
        shared_results_name : str
            Name of the shared memory buffer for the results.
        inputs_size : int
            Size of the inputs to be written.
        results_size : int
            Size of the results to be written.
        n_sim_memory : int
            Number of simulations that can be stored in memory.

        Returns
        -------
        None
        """
        # open shared memory buffers
        shm_inputs = shared_memory.SharedMemory(shared_inputs_name)
        shm_results = shared_memory.SharedMemory(shared_results_name)

        shared_inputs_buffer = np.ndarray(
            (n_sim_memory, inputs_size), dtype=ctypes.c_ubyte, buffer=shm_inputs.buf
        )
        shared_results_buffer = np.ndarray(
            (n_sim_memory, results_size), dtype=ctypes.c_ubyte, buffer=shm_results.buf
        )

        try:
            while True:
                sim_idx = sim_counter.increment()
                if sim_idx == -1:
                    break

                env = sto_env.create_object()
                rocket = sto_rocket.create_object()
                rail_length = sto_flight._randomize_rail_length()
                inclination = sto_flight._randomize_inclination()
                heading = sto_flight._randomize_heading()
                initial_solution = sto_flight.initial_solution
                terminate_on_apogee = sto_flight.terminate_on_apogee

                monte_carlo_flight = Flight(
                    rocket=rocket,
                    environment=env,
                    rail_length=rail_length,
                    inclination=inclination,
                    heading=heading,
                    initial_solution=initial_solution,
                    terminate_on_apogee=terminate_on_apogee,
                )

                # Export to file
                inputs_dict = dict(
                    item
                    for d in [
                        sto_env.last_rnd_dict,
                        sto_rocket.last_rnd_dict,
                        sto_flight.last_rnd_dict,
                    ]
                    for item in d.items()
                )

                inputs_dict["idx"] = sim_idx
                inputs_dict = MonteCarlo.prepare_export_data(
                    inputs_dict, export_sample_time, remove_functions=True
                )

                if light_mode:
                    # Construct the dict with the results from the flight
                    results = {
                        export_item: getattr(monte_carlo_flight, export_item)
                        for export_item in file_paths["export_list"]
                    }

                    export_inputs = json.dumps(inputs_dict, cls=RocketPyEncoder)
                    export_outputs = json.dumps(results, cls=RocketPyEncoder)

                else:
                    # serialize data
                    flight_results = MonteCarlo.prepare_export_data(
                        monte_carlo_flight, sample_time=export_sample_time
                    )

                    # place data in dictionary as it will be found in output file
                    export_inputs = {
                        str(sim_idx): inputs_dict,
                    }

                    export_outputs = {
                        str(sim_idx): flight_results,
                    }

                # convert to bytes
                export_inputs_bytes = pickle.dumps(export_inputs)
                export_outputs_bytes = pickle.dumps(export_outputs)

                if len(export_inputs_bytes) > inputs_size:
                    raise ValueError(
                        "Input data is too large to fit in the shared memory buffer."
                    )

                if len(export_outputs_bytes) > results_size:
                    raise ValueError(
                        "Output data is too large to fit in the shared memory buffer."
                    )

                # add padding to make sure the byte stream fits in the allocated space
                export_inputs_bytes = export_inputs_bytes.ljust(inputs_size, b'\0')
                export_outputs_bytes = export_outputs_bytes.ljust(results_size, b'\0')

                # write to shared memory
                MonteCarlo.__export_simulation_data(
                    go_write_inputs,
                    go_read_inputs,
                    shared_inputs_buffer,
                    export_inputs_bytes,
                )

                # write data to the shared buffer
                MonteCarlo.__export_simulation_data(
                    go_write_results,
                    go_read_results,
                    shared_results_buffer,
                    export_outputs_bytes,
                )

                # update user on progress
                sim_counter.reprint(
                    sim_idx,
                    end="\n",
                    flush=False,
                )

        except Exception as error:
            print(f"Error on iteration {sim_idx}: {error}")

            # write error to file
            errors_lock.acquire()
            with open(file_paths["error_file"], mode='a', encoding="utf-8") as f:
                f.write(json.dumps(inputs_dict, cls=RocketPyEncoder) + "\n")
            errors_lock.release()

            raise error

        finally:
            print("Worker stopped.")

    def __run_single_simulation(
        self, sim_idx, input_file, output_file, light_mode=False
    ):
        """
        Runs a single simulation and saves the inputs and outputs to the
        respective files.

        Parameters
        ----------
        input_file : str
            The file object to write the inputs.
        output_file : str
            The file object to write the outputs.

        Returns
        -------
        None
        """
        self.__iteration_count += 1

        monte_carlo_flight = Flight(
            rocket=self.rocket.create_object(),
            environment=self.environment.create_object(),
            rail_length=self.flight._randomize_rail_length(),
            inclination=self.flight._randomize_inclination(),
            heading=self.flight._randomize_heading(),
            initial_solution=self.flight.initial_solution,
            terminate_on_apogee=self.flight.terminate_on_apogee,
        )

        self._inputs_dict = dict(
            item
            for d in [
                self.environment.last_rnd_dict,
                self.rocket.last_rnd_dict,
                self.flight.last_rnd_dict,
            ]
            for item in d.items()
        )
        self._inputs_dict["idx"] = sim_idx

        # Export inputs and outputs to file
        if light_mode:
            self.__export_flight_data(
                flight=monte_carlo_flight,
                inputs_dict=self._inputs_dict,
                input_file=input_file,
                output_file=output_file,
            )
        else:
            # serialize data
            flight_results = MonteCarlo.prepare_export_data(
                monte_carlo_flight, sample_time=self.export_sample_time
            )

            # place data in dictionary as it will be found in output file
            export_inputs = {
                str(sim_idx): self._inputs_dict,
            }
            export_outputs = {
                str(sim_idx): flight_results,
            }

            self.__dict_to_h5(input_file, '/', export_inputs)
            self.__dict_to_h5(output_file, '/', export_outputs)

        average_time = (process_time() - self.__start_cpu_time) / self.__iteration_count
        estimated_time = int(
            (self.number_of_simulations - self.__iteration_count) * average_time
        )
        self.__reprint(
            f"Current iteration: {self.__iteration_count:06d} | "
            f"Average Time per Iteration: {average_time:.3f} s | "
            f"Estimated time left: {estimated_time} s",
            end="\r",
            flush=True,
        )

    @staticmethod
    def __export_simulation_data(go_write, go_read, shared_buffer, export_bytes):
        """
        Export the simulation data to the shared memory buffer. This function
        will loop through the shared memory buffer to find an empty slot, write
        the data to the shared buffer, and signal the input reader that the data
        is ready.

        Parameters
        ----------
        go_write : list
            List of semaphores to write the data.
        go_read : list
            List of semaphores to read the data.
        shared_buffer : np.ndarray
            Shared memory buffer with the data.
        export_bytes : bytes
            Data to be written to the shared buffer.

        Returns
        -------
        bool
            True if the data was saved. An error will be raised otherwise.
        """
        i = 0
        found_slot = False

        # loop through the shared memory buffer to find an empty slot
        while not found_slot:
            if i >= len(go_write):
                i = 0

            # try to acquire the semaphore, skip if it is already acquired
            if go_write[i].acquire(timeout=1e-3):
                # write data to the shared buffer
                shared_buffer[i] = np.frombuffer(export_bytes, dtype=ctypes.c_ubyte)

                # signal the input reader that the data is ready
                go_read[i].release()
                found_slot = True
            else:
                i += 1

        return True

    @staticmethod
    def __loop_though_buffer(
        file,
        shared_buffer,
        go_read_semaphores,
        go_write_semaphores,
        light_mode,
    ):
        """
        Loop through the shared buffer, writing the data to the file.

        Parameters
        ----------
        file : h5py.File or TextIOWrapper
            File object to write the data.
        shared_buffer : np.ndarray
            Shared memory buffer with the data.
        go_read_semaphores : list
            List of semaphores to read the data.
        go_write_semaphores : list
            List of semaphores to write the data.
        light_mode : bool
            If True, only variables from the export_list will be saved to
            the output file as a .txt file. If False, all variables will be
            saved to the output file as a .h5 file.

        Returns
        -------
        None
        """
        # loop through all the semaphores
        for i, sem in enumerate(go_read_semaphores):
            # try to acquire the semaphore, skip if it is already acquired
            if sem.acquire(timeout=1e-3):
                # retrieve the data from the shared buffer
                data = shared_buffer[i]
                data_deserialized = pickle.loads(bytes(data))

                # write data to the file
                if light_mode:
                    file.write(data_deserialized + "\n")
                else:
                    MonteCarlo.__dict_to_h5(file, "/", data_deserialized)

                # release the write semaphore // tell worker it can write again
                go_write_semaphores[i].release()

    @staticmethod
    def _write_data_worker(
        file_path,
        go_write_semaphores,
        go_read_semaphores,
        shared_name,
        data_size,
        stop_event,
        n_sim_memory,
        light_mode,
    ):
        """
        Worker function to write data to the file.

        Parameters
        ----------
        file_path : str
            Path to the file to write the data.
        go_write_semaphores : list
            List of semaphores to write the data.
        go_read_semaphores : list
            List of semaphores to read the data.
        shared_name : str
            Name of the shared memory buffer.
        data_size : int
            Size of the data to be written.
        stop_event : Event
            Event to stop the worker.
        n_sim_memory : int
            Number of simulations that can be stored in memory.
        light_mode : bool
            If True, only variables from the export_list will be saved to
            the output file as a .txt file. If False, all variables will be
            saved to the output file as a .h5 file.
        """
        shm = shared_memory.SharedMemory(shared_name)
        shared_buffer = np.ndarray(
            (n_sim_memory, data_size), dtype=ctypes.c_ubyte, buffer=shm.buf
        )
        if light_mode:
            with open(file_path, mode="a", encoding="utf-8") as f:
                while not stop_event.is_set():
                    MonteCarlo.__loop_though_buffer(
                        f,
                        shared_buffer,
                        go_read_semaphores,
                        go_write_semaphores,
                        light_mode,
                    )

                # loop through the remaining data
                MonteCarlo.__loop_though_buffer(
                    f,
                    shared_buffer,
                    go_read_semaphores,
                    go_write_semaphores,
                    light_mode,
                )

        else:
            with h5py.File(file_path, 'a') as h5_file:
                # loop until the stop event is set
                while not stop_event.is_set():
                    MonteCarlo.__loop_though_buffer(
                        h5_file,
                        shared_buffer,
                        go_read_semaphores,
                        go_write_semaphores,
                        light_mode,
                    )

                # loop through the remaining data
                MonteCarlo.__loop_though_buffer(
                    h5_file,
                    shared_buffer,
                    go_read_semaphores,
                    go_write_semaphores,
                    light_mode,
                )

    @staticmethod
    def __downsample_recursive(data_dict, max_time, sample_time):
        """
        Given a dictionary, this function will downsample all arrays in the
        dictionary to the sample_time, filling the arrays up to the max_time.
        The function is recursive, so it will go through all the nested
        dictionaries.

        Parameters
        ----------
        data_dict : dict
            Dictionary to be downsampled.
        max_time : float
            Maximum time to fill the arrays.
        sample_time : float
            Sample time to downsample the arrays.

        Returns
        -------
        dict
            Downsampled dictionary.
        """
        # calculate the new size of the arrays
        new_size = int(max_time / sample_time) + 1

        # downsample the arrays
        for key, value in data_dict.items():
            if isinstance(value, dict):
                data_dict[key] = MonteCarlo.__downsample_recursive(
                    value, max_time, sample_time
                )
            elif isinstance(value, np.ndarray):
                if len(value.shape) > 1:
                    new_array = np.zeros((new_size, value.shape[1]), dtype=value.dtype)
                else:
                    new_array = np.zeros((new_size, 1), dtype=value.dtype)

                data_dict[key] = new_array
            else:
                data_dict[key] = value

        return data_dict

    def __get_export_size(self, light_mode):
        """
        This function runs a simulation, fills all exported arrays up to the max
        time, serializes the dictionary, and returns the size of the serialized
        dictionary. The purpose is to estimate the size of the exported data.
        """
        # Run trajectory simulation
        env = self.environment.create_object()
        rocket = self.rocket.create_object()
        rail_length = self.flight._randomize_rail_length()
        inclination = self.flight._randomize_inclination()
        heading = self.flight._randomize_heading()
        initial_solution = self.flight.initial_solution
        terminate_on_apogee = self.flight.terminate_on_apogee

        monte_carlo_flight = Flight(
            rocket=rocket,
            environment=env,
            rail_length=rail_length,
            inclination=inclination,
            heading=heading,
            initial_solution=initial_solution,
            terminate_on_apogee=terminate_on_apogee,
        )

        if monte_carlo_flight.max_time is None or monte_carlo_flight.max_time <= 0:
            raise ValueError(
                "The max_time attribute must be greater than zero. To use parallel mode."
            )

        # Export inputs and outputs to file
        export_inputs = dict(
            item
            for d in [
                self.environment.last_rnd_dict,
                self.rocket.last_rnd_dict,
                self.flight.last_rnd_dict,
            ]
            for item in d.items()
        )
        export_inputs["idx"] = 123456789

        export_inputs = self.prepare_export_data(
            export_inputs, self.export_sample_time, remove_functions=True
        )

        export_inputs = self.__downsample_recursive(
            data_dict=export_inputs,
            max_time=monte_carlo_flight.max_time,
            sample_time=self.export_sample_time,
        )

        if light_mode:
            results = {
                export_item: getattr(monte_carlo_flight, export_item)
                for export_item in self.export_list
            }

            export_inputs_bytes = json.dumps(export_inputs, cls=RocketPyEncoder)
            results_bytes = json.dumps(results, cls=RocketPyEncoder)
        else:
            flight_results = self.prepare_export_data(
                monte_carlo_flight, self.export_sample_time
            )
            results = {"probe_flight": flight_results}

            # downsample the arrays, filling them up to the max time
            results = self.__downsample_recursive(
                data_dict=results,
                max_time=monte_carlo_flight.max_time,
                sample_time=self.export_sample_time,
            )

        # serialize the dictionary
        export_inputs_bytes = pickle.dumps(export_inputs)
        results_bytes = pickle.dumps(results)

        # get the size of the serialized dictionary
        export_inputs_size = len(export_inputs_bytes)
        results_size = len(results_bytes)

        return export_inputs_size, results_size

    def __close_files(self, input_file, output_file, error_file):
        """
        Closes all the files.

        Parameters
        ----------
        input_file : str
            The file object to write the inputs.
        output_file : str
            The file object to write the outputs.
        error_file : str
            The file object to write the errors.

        Returns
        -------
        None
        """
        input_file.close()
        output_file.close()
        error_file.close()

    def __terminate_simulation(self, input_file, output_file, error_file, light_mode):
        """
        Terminates the simulation, closes the files and prints the results.

        Parameters
        ----------
        input_file : str
            The file object to write the inputs.
        output_file : str
            The file object to write the outputs.
        error_file : str
            The file object to write the errors.
        light_mode : bool
            If True, only variables from the export_list will be saved to
            the output file as a .txt file. If False, all variables will be
            saved to the output file as a .h5 file.

        Returns
        -------
        None
        """
        final_string = (
            f"Completed {self.__iteration_count} iterations. Total CPU time: "
            f"{process_time() - self.__start_cpu_time:.1f} s. Total wall time: "
            f"{time() - self.__start_time:.1f} s\n"
        )

        self.__reprint(final_string + "Saving results.", flush=True)

        # close files to guarantee saving
        self.__close_files(input_file, output_file, error_file)

        if light_mode:
            # resave the files on self and calculate post simulation attributes
            self.input_file = self.batch_path / f"{self.filename}.inputs.txt"
            self.output_file = self.batch_path / f"{self.filename}.outputs.txt"
            self.error_file = self.batch_path / f"{self.filename}.errors.txt"

        print(f"Results saved to {self._output_file}")

    def __export_flight_data(
        self,
        flight,
        inputs_dict,
        input_file,
        output_file,
    ):
        """
        Exports the flight data to the respective files.

        Parameters
        ----------
        flight : Flight
            The Flight object containing the flight data.
        inputs_dict : dict
            Dictionary containing the inputs used in the simulation.
        input_file : str
            The file object to write the inputs.
        output_file : str
            The file object to write the outputs.

        Returns
        -------
        None
        """
        results = {
            export_item: getattr(flight, export_item)
            for export_item in self.export_list
        }

        input_file.write(json.dumps(inputs_dict, cls=RocketPyEncoder) + "\n")
        output_file.write(json.dumps(results, cls=RocketPyEncoder) + "\n")

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

    def __reprint(self, msg, end="\n", flush=False):
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
        len_msg = len(msg)
        if len_msg < self._last_print_len:
            msg += " " * (self._last_print_len - len_msg)
        else:
            self._last_print_len = len_msg

        print(msg, end=end, flush=flush)

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

        print(
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

        print(f"The following input file was imported: {self.input_file}")

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
        print(f"The following error file was imported: {self.error_file}")

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
    def time_function_serializer(function_object, t_range=None, sample_time=None):
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
    def prepare_export_data(obj, sample_time=0.1, remove_functions=False):
        """
        Inspects the attributes of an object and returns a dictionary of its
        attributes.

        Parameters
        ----------
        obj : object
            The object whose attributes are to be inspected.

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
                        result[attr_name] = MonteCarlo.prepare_export_data(
                            attr_value, sample_time
                        )

                    elif not remove_functions and isinstance(attr_value, Function):
                        # Serialize the Functions
                        result[attr_name] = MonteCarlo.time_function_serializer(
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
                        result[attr_name] = MonteCarlo.prepare_export_data(
                            attr_value, sample_time
                        )

                    elif not remove_functions and isinstance(attr_value, Function):
                        # Serialize the Functions
                        result[attr_name] = MonteCarlo.time_function_serializer(
                            attr_value, None, sample_time
                        )

        return result

    @staticmethod
    def __get_initial_sim_idx(file, append, light_mode):
        """
        Get the initial simulation index from the filename.

        Parameters
        ----------
        filename : str
            Name of the file to be analyzed.
        append : bool
            If True, the file will be opened in append mode. Default is False.
        light_mode : bool
            If True, the file will be opened in light mode. Default is False.

        Returns
        -------
        int
            Initial simulation index.
        """
        if append is False:
            return 0

        if light_mode:  # txt file / light mode
            lines = file.readlines()
            idx = len(lines)

        else:  # h5 file / heavy mode
            if len(file.keys()) == 0:
                idx = 0
            else:
                # avoid overwriting since parallel mode does not save in order
                keys = [int(key) for key in file.keys()]
                idx = max(keys) + 1

        return idx

    @staticmethod
    def __dict_to_h5(h5_file, path, dic):
        """
        Converts a dictionary to a h5 file.

        Parameters
        ----------
        h5_file : h5py.File
            File object to be written.
        path : str
            Path to the group to be created.
        dic : dict
            Dictionary to be converted.

        Returns
        -------
        None
        """
        for key, item in dic.items():
            if isinstance(item, (np.int64, np.float64, int, float)):
                data = np.array([[item]])
                h5_file.create_dataset(
                    path + key, data=data, shape=data.shape, dtype=data.dtype
                )
            elif isinstance(item, np.ndarray):
                if len(item.shape) < 2:
                    item = item.reshape(-1, 1)  # Ensure it is a column vector
                h5_file.create_dataset(
                    path + key,
                    data=item,
                    shape=item.shape,
                    dtype=item.dtype,
                )
            elif isinstance(item, (str, bytes)):
                h5_file.create_dataset(
                    path + key,
                    data=item,
                )
            elif isinstance(item, Function):
                raise TypeError(
                    "Function objects should be preprocessed before saving."
                )
            elif isinstance(item, dict):
                MonteCarlo.__dict_to_h5(h5_file, path + key + '/', item)
            else:
                pass  # Implement other types as needed


class MonteCarloManager(BaseManager):
    def __init__(self):
        super().__init__()
        self.register('Lock', Lock)
        self.register('Event', Event)
        self.register('Semaphore', Semaphore)
        self.register('SimCounter', SimCounter)
        self.register('StochasticEnvironment', StochasticEnvironment)
        self.register('StochasticRocket', StochasticRocket)
        self.register('StochasticFlight', StochasticFlight)


class SimCounter:
    def __init__(self, initial_count, n_simulations, parallel_start_time):
        self.initial_count = initial_count
        self.count = initial_count
        self.n_simulations = n_simulations
        self._last_print_len = 0  # used to print on the same line
        self.initial_time = parallel_start_time

    def increment(self):
        if self.count >= self.n_simulations:
            return -1

        self.count += 1
        return self.count - 1

    def set_count(self, count):
        self.count = count

    def get_count(self):
        return self.count

    def get_n_simulations(self):
        return self.n_simulations

    def get_intial_time(self):
        return self.initial_time

    def reprint(self, sim_idx, end="\n", flush=False):
        """Prints a message on the same line as the previous one and replaces
        the previous message with the new one, deleting the extra characters
        from the previous message.

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
        average_time = (time() - self.initial_time) / (self.count - self.initial_count)
        estimated_time = int(
            (self.n_simulations - (self.count - self.initial_count)) * average_time
        )

        msg = f"Current iteration: {sim_idx:06d}"
        msg += f" | Average Time per Iteration: {average_time:.3f} s"
        msg += f" | Estimated time left: {estimated_time} s"

        len_msg = len(msg)
        if len_msg < self._last_print_len:
            msg += " " * (self._last_print_len - len_msg)
        else:
            self._last_print_len = len_msg

        print(msg, end=end, flush=flush)
