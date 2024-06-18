"""Defines the MonteCarlo class."""
import ctypes
import json
import os
import pickle
from pathlib import Path
from time import process_time, time

import h5py
import numpy as np
import simplekml
from multiprocess import Array, Lock, Process, Semaphore, Event
from multiprocess.managers import BaseManager

from rocketpy import Function
from rocketpy._encoders import RocketPyEncoder
from rocketpy.plots.monte_carlo_plots import _MonteCarloPlots
from rocketpy.prints.monte_carlo_prints import _MonteCarloPrints
from rocketpy.simulation.flight import Flight
from rocketpy.simulation.sim_config.flight2serializer import flightv1_serializer
from rocketpy.simulation.sim_config.serializer import function_serializer
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
        When running a new simulation, this parameter represents the
        initial part of the export filenames. For example, if the value
        is 'filename', the exported output files will be named
        'filename.outputs.txt'. When analyzing the results of a
        previous simulation, this parameter should be set to the .txt
        file containing the outputs of the previous monte carlo analysis.
    environment : StochasticEnvironment
        The stochastic environment object to be iterated over.
    rocket : StochasticRocket
        The stochastic rocket object to be iterated over.
    flight : StochasticFlight
        The stochastic flight object to be iterated over.
    export_list : list
        The list of variables to export. If None, the default list will
        be used. Default is None. # TODO: improve docs to explain the
        default list, and what can be exported.
    inputs_log : list
        List of dictionaries with the inputs used in each simulation.
    outputs_log : list
        List of dictionaries with the outputs of each simulation.
    errors_log : list
        List of dictionaries with the errors of each simulation.
    num_of_loaded_sims : int
        Number of simulations loaded from output_file being currently used.
    results : dict
        Monte carlo analysis results organized in a dictionary where the keys
        are the names of the saved attributes, and the values are a list with
        all the result number of the respective attribute
    processed_results : dict
        Creates a dictionary with the mean and standard deviation of each
        parameter available in the results
    prints : _MonteCarloPrints
        Object with methods to print information about the monte carlo
        simulation.
    plots : _MonteCarloPlots
        Object with methods to plot information about the monte carlo
        simulation.
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
    ):
        """
        Initialize a MonteCarlo object.

        Parameters
        ----------
        filename : str
            When running a new simulation, this parameter represents the
            initial part of the export filenames. For example, if the value
            is 'filename', the exported output files will be named
            'filename.outputs.txt'. When analyzing the results of a
            previous simulation, this parameter should be set to the .txt
            file containing the outputs of the previous monte carlo
            analysis.
        environment : StochasticEnvironment
            The stochastic environment object to be iterated over.
        rocket : StochasticRocket
            The stochastic rocket object to be iterated over.
        flight : StochasticFlight
            The stochastic flight object to be iterated over.
        export_list : list, optional
            The list of variables to export. If None, the default list will
            be used. Default is None. # TODO: improve docs to explain the
            default list, and what can be exported.
        batch_path : str, optional
            Path to the batch folder to be used in the simulation. Export file
            will be saved in this folder. Default is None.
        export_sample_time : float, optional
            Sample time to downsample the arrays in seconds. Default is 0.1.

        Returns
        -------
        None
        """
        # Save and initialize parameters
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

        # Checks export_list
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

    def simulate(
        self,
        number_of_simulations,
        append=False,
        light_mode=False,
        parallel=False,
        n_workers=None,
    ):
        """
        Runs the monte carlo simulation and saves all data.

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

        Returns
        -------
        None
        """
        # initialize counters
        self.number_of_simulations = number_of_simulations
        self.iteration_count = self.num_of_loaded_sims if append else 0
        self.start_time = time()
        self.start_cpu_time = process_time()

        # Begin display
        print("Starting monte carlo analysis", end="\r")

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
            # get initial simulation index
            idx_i, idx_o = self.__get_light_indexes(
                self._input_file, self._output_file, append=append
            )

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

            idx_i = self.__get_initial_sim_idx(input_file, light_mode=light_mode)
            idx_o = self.__get_initial_sim_idx(output_file, light_mode=light_mode)

        if idx_i != idx_o:
            raise ValueError(
                "Input and output files are not synchronized. Append mode is not available."
            )

        # Run simulations
        try:
            while self.iteration_count < self.number_of_simulations:
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
            print(f"Error on iteration {self.iteration_count}: {error}")
            error_file.write(json.dumps(self._inputs_dict, cls=RocketPyEncoder) + "\n")
            self.__close_files(input_file, output_file, error_file)
            raise error

        self.__finalize_simulation(
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
        parallel_start = time()
        processes = []

        if n_workers is None or n_workers > os.cpu_count() - 2: # leave 2 cores for the writer workers
            n_workers = os.cpu_count() - 2

        # get the size of the serialized dictionary
        inputs_size, results_size = self.__get_export_size(light_mode)

        # add safety margin to the buffer size
        inputs_size += 1024
        results_size += 1024

        # initialize shared memory buffer
        shared_inputs_buffer = Array(ctypes.c_ubyte, inputs_size * n_workers)
        shared_results_buffer = Array(ctypes.c_ubyte, results_size * n_workers)

        with MonteCarloManager() as manager:
            # initialize queue
            inputs_lock = manager.Lock()
            outputs_lock = manager.Lock()
            errors_lock = manager.Lock()

            go_write_inputs = [manager.Semaphore(value=1) for _ in range(n_workers)]
            go_read_inputs = [manager.Semaphore(value=1) for _ in range(n_workers)]
            
            go_write_results = [manager.Semaphore(value=1) for _ in range(n_workers)]
            go_read_results = [manager.Semaphore(value=1) for _ in range(n_workers)]

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
                # get initial simulation index
                idx_i, idx_o = self.__get_light_indexes(
                    self._input_file, self._output_file, append=append
                )
                # open files in write/append mode
                with open(self._input_file, mode=open_mode) as f:
                    pass  # initialize file
                with open(self._output_file, mode=open_mode) as f:
                    pass  # initialize file

            else:
                # Change file extensions to .h5
                file_paths["input_file"] = file_paths["input_file"].with_suffix(".h5")
                file_paths["output_file"] = file_paths["output_file"].with_suffix(".h5")
                file_paths["error_file"] = file_paths["error_file"].with_suffix(".h5")

                # Initialize files and get initial simulation index
                with h5py.File(file_paths["input_file"], open_mode) as f:
                    idx_i = self.__get_initial_sim_idx(f, light_mode=light_mode)
                with h5py.File(file_paths["output_file"], open_mode) as f:
                    idx_o = self.__get_initial_sim_idx(f, light_mode=light_mode)

            if idx_i != idx_o:
                raise ValueError(
                    "Input and output files are not synchronized. Append mode is not available."
                )

            # Initialize error file - always a .txt file
            with open(self._error_file, mode=open_mode) as _:
                pass  # initialize file

            # Initialize simulation counter
            sim_counter = manager.SimCounter(
                idx_i, self.number_of_simulations, parallel_start
            )

            print("\nStarting monte carlo analysis", end="\r")
            print(f"Number of simulations: {self.number_of_simulations}")

            # Creates n_workers processes then starts them
            for i in range(n_workers):
                p = Process(
                    target=self.__run_simulation_worker,
                    args=(
                        i,
                        self.environment,
                        self.rocket,
                        self.flight,
                        sim_counter,
                        inputs_lock,
                        outputs_lock,
                        errors_lock,
                        go_write_inputs[i],
                        go_write_results[i],
                        go_read_inputs[i],
                        go_read_results[i],
                        light_mode,
                        file_paths,
                        shared_inputs_buffer,
                        shared_results_buffer,
                        inputs_size,
                        results_size,
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
                    shared_inputs_buffer,
                    inputs_size,
                    input_writer_stop_event,
                ),
            )
            
            results_writer = Process(
                target=self._write_data_worker,
                args=(
                    file_paths["output_file"],
                    go_write_results,
                    go_read_results,
                    shared_results_buffer,
                    results_size,
                    results_writer_stop_event,
                ),
            )
            
            # start the writer workers
            input_writer.start()
            results_writer.start()

            # Joins all the processes
            for p in processes:
                p.join()
            
            print("All workers joined, simulation complete.")
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
            print("Simulation took", parallel_end - parallel_start, "seconds to run.")

    @staticmethod
    def __run_simulation_worker(
        worker_no,
        sto_env,
        sto_rocket,
        sto_flight,
        sim_counter,
        inputs_lock,
        outputs_lock,
        errors_lock,
        go_write_inputs,
        go_write_results,
        go_read_inputs,
        go_read_results,
        light_mode,
        file_paths,
        shared_inputs_buffer,
        shared_results_buffer,
        inputs_size,
        results_size,
    ):
        """
        Runs a single simulation worker.

        Parameters
        ----------
        worker_no : int
            Worker number.
        sto_env : StochasticEnvironment
            Stochastic environment object.
        sto_rocket : StochasticRocket
            Stochastic rocket object.
        sto_flight : StochasticFlight
            Stochastic flight object.
        sim_counter : SimCounter
            Simulation counter object.
        inputs_lock : Lock
            Lock object for inputs file.
        outputs_lock : Lock
            Lock object for outputs file.
        errors_lock : Lock
            Lock object for errors file.
        buffer_lock : Lock
            Lock object for shared memory buffer.
        light_mode : bool
            If True, only variables from the export_list will be saved to
            the output file as a .txt file. If False, all variables will be
            saved to the output file as a .h5 file.
        file_paths : dict
            Dictionary with the file paths.
        shared_inputs_buffer : Array
            Shared memory buffer for inputs.
        shared_results_buffer : Array
            Shared memory buffer for results.
        inputs_size : int
            Size of the serialized dictionary.
        results_size : int
            Size of the serialized dictionary.

        Returns
        -------
        None
        """
        # get the size of the serialized dictionary
        begin_mem_i = worker_no * inputs_size
        end_mem_i = (worker_no + 1) * inputs_size
        begin_mem_r = worker_no * results_size
        end_mem_r = (worker_no + 1) * results_size

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

                flight = Flight(
                    rocket=rocket,
                    environment=env,
                    rail_length=rail_length,
                    inclination=inclination,
                    heading=heading,
                    initial_solution=initial_solution,
                    terminate_on_apogee=terminate_on_apogee,
                )

                # # Export to file
                # if light_mode:
                #     inputs_dict = dict(
                #         item
                #         for d in [
                #             sto_env.last_rnd_dict,
                #             sto_rocket.last_rnd_dict,
                #             sto_flight.last_rnd_dict,
                #         ]
                #         for item in d.items()
                #     )

                #     # Construct the dict with the results from the flight
                #     results = {
                #         export_item: getattr(flight, export_item)
                #         for export_item in file_paths["export_list"]
                #     }

                #     # Write flight setting and results to file
                #     inputs_lock.acquire()
                #     with open(
                #         file_paths["input_file"], mode='a', encoding="utf-8"
                #     ) as f:
                #         f.write(json.dumps(inputs_dict, cls=RocketPyEncoder) + "\n")
                #     inputs_lock.release()

                #     outputs_lock.acquire()
                #     with open(
                #         file_paths["output_file"], mode='a', encoding="utf-8"
                #     ) as f:
                #         f.write(json.dumps(results, cls=RocketPyEncoder) + "\n")
                #     outputs_lock.release()

                # else:
                input_parameters = flightv1_serializer(
                    flight, f"Simulation_{sim_idx}", return_dict=True
                )

                flight_results = MonteCarlo.inspect_object_attributes(flight)

                export_inputs = {
                    str(sim_idx): input_parameters,
                }

                export_outputs = {
                    str(sim_idx): flight_results,
                }

                # placeholder logic, needs to be implemented
                export_inputs_downsampled = MonteCarlo.__downsample_recursive(
                    data_dict=export_inputs,
                    max_time=flight.max_time,
                    sample_time=0.1,
                )
                export_outputs_downsampled = MonteCarlo.__downsample_recursive(
                    data_dict=export_outputs,
                    max_time=flight.max_time,
                    sample_time=0.1,
                )

                # convert to bytes
                export_inputs_bytes = pickle.dumps(export_inputs_downsampled)
                export_outputs_bytes = pickle.dumps(export_outputs_downsampled)

                # add padding to make sure the byte stream fits in the allocated space
                export_inputs_bytes = export_inputs_bytes.ljust(inputs_size, b'\0')
                export_outputs_bytes = export_outputs_bytes.ljust(
                    results_size, b'\0'
                )

                # write to shared memory
                go_write_inputs.acquire()
                shared_inputs_buffer[begin_mem_i:end_mem_i] = export_inputs_bytes
                go_read_inputs.release()
                
                go_write_results.acquire()
                shared_results_buffer[begin_mem_r:end_mem_r] = export_outputs_bytes
                go_read_results.release()

                    # inputs_lock.acquire()
                    # with h5py.File(file_paths["input_file"], 'a') as h5_file:
                    #     MonteCarlo.__dict_to_h5(h5_file, '/', export_inputs)
                    # inputs_lock.release()

                    # outputs_lock.acquire()
                    # with h5py.File(file_paths["output_file"], 'a') as h5_file:
                    #     MonteCarlo.__dict_to_h5(h5_file, '/', export_outputs)
                    # outputs_lock.release()

                average_time = (
                    time() - sim_counter.get_intial_time()
                ) / sim_counter.get_count()
                estimated_time = int(
                    (sim_counter.get_n_simulations() - sim_counter.get_count())
                    * average_time
                )

                sim_counter.reprint(
                    f"Current iteration: {sim_idx:06d} | "
                    f"Average Time per Iteration: {average_time:.3f} s | "
                    f"Estimated time left: {estimated_time} s",
                    end="\n",
                    flush=False,
                )

        # except Exception as error:
        #     print(f"Error on iteration {sim_idx}: {error}")
        #     errors_lock.acquire()
        #     with open(file_paths["error_file"], mode='a', encoding="utf-8") as f:
        #         f.write(json.dumps(inputs_dict, cls=RocketPyEncoder) + "\n")
        #     errors_lock.release()
        #     raise error

        finally:
            print("Worker stopped.")

    def __run_single_simulation(
        self, sim_idx, input_file, output_file, light_mode=False
    ):
        """Runs a single simulation and saves the inputs and outputs to the
        respective files."""
        # Update iteration count
        self.iteration_count += 1
        # Run trajectory simulation
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

        # Export inputs and outputs to file
        if light_mode:
            self.__export_flight_data(
                flight=monte_carlo_flight,
                inputs_dict=self._inputs_dict,
                input_file=input_file,
                output_file=output_file,
            )
        else:
            input_parameters = flightv1_serializer(
                monte_carlo_flight, f"Simulation_{sim_idx}", return_dict=True
            )

            flight_results = self.inspect_object_attributes(monte_carlo_flight)

            export_inputs = {str(sim_idx): input_parameters}
            export_outputs = {str(sim_idx): flight_results}

            self.__dict_to_h5(input_file, '/', export_inputs)
            self.__dict_to_h5(output_file, '/', export_outputs)

        average_time = (process_time() - self.start_cpu_time) / self.iteration_count
        estimated_time = int(
            (self.number_of_simulations - self.iteration_count) * average_time
        )
        self.__reprint(
            f"Current iteration: {self.iteration_count:06d} | "
            f"Average Time per Iteration: {average_time:.3f} s | "
            f"Estimated time left: {estimated_time} s",
            end="\r",
            flush=True,
        )

    @staticmethod
    def _write_data_worker(
        file_path,
        go_write_semaphores,
        go_read_semaphores,
        shared_buffer,
        data_size,
        stop_event,
    ):
        sim_idx = 0
        with h5py.File(file_path, 'a') as h5_file:
            # loop until the stop event is set
            while not stop_event.is_set():
                # loop through all the semaphores
                for i, sem in enumerate(go_read_semaphores):
                    # try to acquire the semaphore, skip if it is already acquired
                    if sem.acquire(timeout=1e-3):
                        # retrieve the data from the shared buffer
                        data = shared_buffer[i * data_size : (i + 1) * data_size]
                        # data_dict = pickle.loads(bytes(data))

                        # write data to the file
                        grp = h5_file.create_group(f"{sim_idx}")
                        grp.create_dataset("data", data=data)
                        
                        # release the write semaphore // tell worker it can write again
                        go_write_semaphores[i].release()
                        sim_idx += 1
                        # print(f"Wrote data to file. Buffer pos: {i}")
                        
            # loop through all the semaphores to write the remaining data
            for i, sem in enumerate(go_read_semaphores):
                # try to acquire the semaphore, skip if it is already acquired
                if sem.acquire(timeout=1e-3):
                    # retrieve the data from the shared buffer
                    data = shared_buffer[i * data_size : (i + 1) * data_size]
                    # data_dict = pickle.loads(bytes(data))

                    # write data to the file
                    grp = h5_file.create_group(f"{sim_idx}")
                    grp.create_dataset("data", data=data)

                    # release the write semaphore // tell worker it can write again
                    go_write_semaphores[i].release()
                    sim_idx += 1

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
        monte_carlo_flight = self.flight.create_object()

        if monte_carlo_flight.max_time is None or monte_carlo_flight.max_time <= 0:
            raise ValueError("The max_time attribute must be greater than zero.")

        # Export inputs and outputs to file
        if light_mode:
            export_inputs = dict(
                item
                for d in [
                    self.environment.last_rnd_dict,
                    self.rocket.last_rnd_dict,
                    self.flight.last_rnd_dict,
                ]
                for item in d.items()
            )
            results = {
                export_item: getattr(monte_carlo_flight, export_item)
                for export_item in self.export_list
            }
        else:
            input_parameters = flightv1_serializer(
                monte_carlo_flight, f"probe_simulation", return_dict=True
            )

            flight_results = self.inspect_object_attributes(monte_carlo_flight)

            export_inputs = {"probe_flight": input_parameters}
            results = {"probe_flight": flight_results}

            # downsample the arrays, filling them up to the max time
            export_inputs = self.__downsample_recursive(
                data_dict=export_inputs,
                max_time=monte_carlo_flight.max_time,
                sample_time=self.export_sample_time,
            )
            results = self.__downsample_recursive(
                data_dict=results,
                max_time=monte_carlo_flight.max_time,
                sample_time=self.export_sample_time,
            )

        # serialize the dictionary
        export_inputs_bytes = pickle.dumps(export_inputs)
        results_bytes = pickle.dumps(results)

        # load again and check if the downsample worked
        export_inputs = pickle.loads(export_inputs_bytes)
        results = pickle.loads(results_bytes)

        # get the size of the serialized dictionary
        export_inputs_size = len(export_inputs_bytes)
        results_size = len(results_bytes)

        return export_inputs_size, results_size

    def __close_files(self, input_file, output_file, error_file):
        """Closes all the files."""
        input_file.close()
        output_file.close()
        error_file.close()

    def __finalize_simulation(self, input_file, output_file, error_file, light_mode):
        """Finalizes the simulation, closes the files and prints the results."""
        final_string = (
            f"Completed {self.iteration_count} iterations. Total CPU time: "
            f"{process_time() - self.start_cpu_time:.1f} s. Total wall time: "
            f"{time() - self.start_time:.1f} s\n"
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
        """Exports the flight data to the respective files."""
        # Construct the dict with the results from the flight
        results = {
            export_item: getattr(flight, export_item)
            for export_item in self.export_list
        }

        # Write flight setting and results to file
        input_file.write(json.dumps(inputs_dict, cls=RocketPyEncoder) + "\n")
        output_file.write(json.dumps(results, cls=RocketPyEncoder) + "\n")

    def __check_export_list(self, export_list):
        """Checks if the export_list is valid and returns a valid list. If no
        export_list is provided, the default list is used."""
        standard_output = set(
            {
                "apogee",
                "apogee_time",
                "apogee_x",
                "apogee_y",
                # "apogee_freestream_speed",
                "t_final",
                "x_impact",
                "y_impact",
                "impact_velocity",
                # "initial_stability_margin", # Needs to implement it!
                # "out_of_rail_stability_margin", # Needs to implement it!
                "out_of_rail_time",
                "out_of_rail_velocity",
                # "max_speed",
                "max_mach_number",
                # "max_acceleration_power_on",
                "frontal_surface_wind",
                "lateral_surface_wind",
            }
        )
        exportables = set(
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
                if attr not in exportables:
                    raise ValueError(
                        f"Attribute '{attr}' can not be exported. Check export_list."
                    )
        else:
            # No export list provided, using default list instead.
            export_list = standard_output

        return export_list

    def __reprint(self, msg, end="\n", flush=False):
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

        len_msg = len(msg)
        if len_msg < self._last_print_len:
            msg += " " * (self._last_print_len - len_msg)
        else:
            self._last_print_len = len_msg

        print(msg, end=end, flush=flush)

    @property
    def input_file(self):
        """String containing the filepath of the input file"""
        return self._input_file

    @input_file.setter
    def input_file(self, value):
        """Setter for input_file. Sets/updates inputs_log."""
        self._input_file = value
        self.set_inputs_log()

    @property
    def output_file(self):
        """String containing the filepath of the output file"""
        return self._output_file

    @output_file.setter
    def output_file(self, value):
        """Setter for input_file. Sets/updates outputs_log, num_of_loaded_sims,
        results, and processed_results."""
        self._output_file = value
        self.set_outputs_log()
        self.set_num_of_loaded_sims()
        self.set_results()
        self.set_processed_results()

    @property
    def error_file(self):
        """String containing the filepath of the error file"""
        return self._error_file

    @error_file.setter
    def error_file(self, value):
        """Setter for input_file. Sets/updates inputs_log."""
        self._error_file = value
        self.set_errors_log()

    # setters for post simulation attributes
    def set_inputs_log(self):
        """Sets inputs_log from a file into an attribute for easy access"""
        self.inputs_log = []
        with open(self.input_file, mode="r", encoding="utf-8") as rows:
            for line in rows:
                self.inputs_log.append(json.loads(line))

    def set_outputs_log(self):
        """Sets outputs_log from a file into an attribute for easy access"""
        self.outputs_log = []
        with open(self.output_file, mode="r", encoding="utf-8") as rows:
            for line in rows:
                self.outputs_log.append(json.loads(line))

    def set_errors_log(self):
        """Sets errors_log log from a file into an attribute for easy access"""
        self.errors_log = []
        with open(self.error_file, mode="r", encoding="utf-8") as errors:
            for line in errors:
                self.errors_log.append(json.loads(line))

    def set_num_of_loaded_sims(self):
        """Number of simulations loaded from output_file being currently used."""
        with open(self.output_file, mode="r", encoding="utf-8") as outputs:
            self.num_of_loaded_sims = sum(1 for _ in outputs)

    def set_results(self):
        """Monte carlo results organized in a dictionary where the keys are the
        names of the saved attributes, and the values are a list with all the
        result number of the respective attribute"""
        self.results = {}
        for result in self.outputs_log:
            for key, value in result.items():
                if key in self.results:
                    self.results[key].append(value)
                else:
                    self.results[key] = [value]

    def set_processed_results(self):
        """Creates a dictionary with the mean and standard deviation of each
        parameter available in the results"""
        self.processed_results = {}
        for result, values in self.results.items():
            mean = np.mean(values)
            stdev = np.std(values)
            self.processed_results[result] = (mean, stdev)

    def import_outputs(self, filename=None):
        """Import monte carlo results from .txt file and save it into a
        dictionary.

        Parameters
        ----------
        filename : str
            Name or directory path to the file to be imported. If none,
            self.filename will be used.

        Returns
        -------
        None
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
        """Import monte carlo results from .txt file and save it into a
        dictionary.

        Parameters
        ----------
        filename : str
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
        """Import monte carlo results from .txt file and save it into a
        dictionary.

        Parameters
        ----------
        filename : str
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
        """Import monte carlo results from .txt file and save it into a
        dictionary.

        Parameters
        ----------
        filename : str
            Name or directory path to the file to be imported. If none,
            self.filename will be used.

        Returns
        -------
        None
        """
        # select file to use
        filepath = filename if filename else self.filename

        self.import_outputs(filename=filepath)
        self.import_inputs(filename=filepath)
        self.import_errors(filename=filepath)

    def export_ellipses_to_kml(
        self,
        filename,
        origin_lat,
        origin_lon,
        type="all",
        resolution=100,
        color="ff0000ff",
    ):
        """Generates a KML file with the ellipses on the impact point.

        Parameters
        ----------
        results : dict
            Contains results from the Monte Carlo simulation.
        filename : String
            Name to the KML exported file.
        origin_lat : float
            Latitude coordinate of Ellipses' geometric center, in degrees.
        origin_lon : float
            Latitude coordinate of Ellipses' geometric center, in degrees.
        type : String
            Type of ellipses to be exported. Options are: 'all', 'impact' and
            'apogee'. Default is 'all', it exports both apogee and impact
            ellipses.
        resolution : int
            Number of points to be used to draw the ellipse. Default is 100.
        color : String
            Color of the ellipse. Default is 'ff0000ff', which is red.
            Kml files use an 8 digit HEX color format, see its docs.

        Returns
        -------
        None
        """

        (
            impact_ellipses,
            apogee_ellipses,
            *_,
        ) = generate_monte_carlo_ellipses(self.results)
        outputs = []

        if type == "all" or type == "impact":
            outputs = outputs + generate_monte_carlo_ellipses_coordinates(
                impact_ellipses, origin_lat, origin_lon, resolution=resolution
            )

        if type == "all" or type == "apogee":
            outputs = outputs + generate_monte_carlo_ellipses_coordinates(
                apogee_ellipses, origin_lat, origin_lon, resolution=resolution
            )

        # Prepare data to KML file
        kml_data = [[(coord[1], coord[0]) for coord in output] for output in outputs]

        # Export to KML
        kml = simplekml.Kml()

        for i in range(len(outputs)):
            if (type == "all" and i < 3) or (type == "impact"):
                ellipse_name = "Impact σ" + str(i + 1)
            elif type == "all" and i >= 3:
                ellipse_name = "Apogee σ" + str(i - 2)
            else:
                ellipse_name = "Apogee σ" + str(i + 1)

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
        """Print information about the monte carlo simulation."""
        self.prints.all()

    def all_info(self):
        """Print and plot information about the monte carlo simulation
        and its results.

        Returns
        -------
        None
        """
        self.info()
        self.plots.ellipses()
        self.plots.all()

    @staticmethod
    def inspect_object_attributes(obj):
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
        # Iterate through all attributes of the object
        for attr_name in dir(obj):
            attr_value = getattr(obj, attr_name)

            # Check if the attribute is of a type we are interested in and not a private attribute
            if isinstance(
                attr_value, (int, float, dict, Function)
            ) and not attr_name.startswith('_'):
                if isinstance(attr_value, Function):
                    # Serialize the Functions
                    result[attr_name] = function_serializer(attr_value)

                elif isinstance(attr_value, dict):
                    # Recursively inspect the dictionary attributes
                    result[attr_name] = MonteCarlo.inspect_object_attributes(attr_value)

                elif isinstance(attr_value, (int, float)):
                    result[attr_name] = attr_value

                else:
                    # Should never reach this point
                    raise TypeError("Methods should be preprocessed before saving.")
        return result

    @staticmethod
    def __get_initial_sim_idx(file, light_mode):
        """
        Get the initial simulation index from the filename.

        Parameters
        ----------
        filename : str
            Name of the file to be analyzed.

        Returns
        -------
        int
            Initial simulation index.
        """
        if light_mode:
            lines = file.readlines()
            return len(lines)

        if len(file.keys()) == 0:
            return 0  # light mode not using the simulation index

        keys = [int(key) for key in file.keys()]
        return max(keys) + 1

    @staticmethod
    def __get_light_indexes(input_file, output_file, append):
        """
        Get the initial simulation index from the filename.

        Parameters
        ----------
        input_file : str
            Name of the input file to be analyzed.
        output_file : str
            Name of the output file to be analyzed.
        append : bool
            If True, the results will be appended to the existing files. If
            False, the files will be overwritten.

        Returns
        -------
        int
            Initial simulation index.
        """
        if append:
            try:
                with open(input_file, 'r', encoding="utf-8") as f:
                    idx_i = MonteCarlo.__get_initial_sim_idx(f, light_mode=True)
                with open(output_file, 'r', encoding="utf-8") as f:
                    idx_o = MonteCarlo.__get_initial_sim_idx(f, light_mode=True)
            except OSError:  # File not found, return 0
                idx_i = 0
                idx_o = 0
        else:
            idx_i = 0
            idx_o = 0

        return idx_i, idx_o

    @staticmethod
    def __dict_to_h5(h5_file, path, dic):
        """
        ....
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
    def __init__(self, initial_count, n_simulations, parallel_start):
        self.count = initial_count
        self.n_simulations = n_simulations
        self._last_print_len = 0  # used to print on the same line
        self.initial_time = parallel_start

    def increment(self):
        if self.count >= self.n_simulations:
            return -1

        self.count += 1
        return self.count - 1

    def get_count(self):
        return self.count

    def get_n_simulations(self):
        return self.n_simulations

    def get_intial_time(self):
        return self.initial_time

    def reprint(self, msg, end="\n", flush=False):
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

        # len_msg = len(msg)
        # if len_msg < self._last_print_len:
        #     msg += " " * (self._last_print_len - len_msg)
        # else:
            # self._last_print_len = len_msg

        print(msg, end=end, flush=flush)
