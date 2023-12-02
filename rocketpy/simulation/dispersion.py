import ast
from time import process_time, time

import numpy as np
import simplekml

from rocketpy.mathutils.function import Function
from rocketpy.plots.dispersion_plots import _DispersionPlots
from rocketpy.prints.dispersion_prints import _DispersionPrints
from rocketpy.simulation.flight import Flight
from rocketpy.tools import (
    generate_dispersion_ellipses,
    generate_dispersion_ellipses_coordinates,
)

# TODO: How to save Functions? With pickle? Save just the source?

# TODO: create a method that recreates each flight from inputs_log
# and saves it in an attribute that is a list

# TODO: Create evolution plots to analyze convergence


class Dispersion:
    """
    This class is used to perform Monte Carlo analysis on the rocket's flight
    trajectory. It is used to predict the probability distributions of the
    rocket's landing point, apogee and other relevant information.

    Attributes
    ----------
    Dispersion.filename: string
        When running a new simulation, this parameter represents the initial
        part of the export filenames (e.g. 'filename.disp_outputs.txt'). When
        analyzing the results of a previous simulation, this parameter shall be
        the .txt filename containing the outputs of a previous ran dispersion
        analysis.
    Dispersion.environment: McEnvironment
        The environment in which the rocket will be launched.
    Dispersion.rocket: McRocket
        The rocket to be launched.
    Dispersion.flight: McFlight
        The flight conditions of the rocket.
    Dispersion.motors: list of McMotor
        The motors to be used in the rocket during the Flight.
    Dispersion.nosecones : list of McNosecone
        The nosecones to be used in the rocket during the Flight.
    Dispersion.fins : list of McTrapezoidalFins or McEllipticalFins
        The fins to be used in the rocket during the Flight.
    Dispersion.tails : list of McTail objects
        The tails to be used in the rocket during the Flight.
    Dispersion.parachutes :  list of McParachute objects
        The parachutes to be used in the rocket during the Flight.
    Dispersion.rail_buttons : list of McRailButtons objects
        The rail buttons to be used in the rocket during the Flight. Usually
        only one object will be present in this list.
    Dispersion.number_of_simulations : int
        Number of simulations to be performed in the run_dispersion() method.
    Dispersion.dispersion_dictionary : dict
        Dictionary containing the parameters to be used in the Monte Carlo
        simulations.
    Dispersion.export_list : list
        List of parameters to be exported from each flight in the Monte Carlo
        loop.
    Dispersion.input_file : str
        String containing the filepath of the input file created during the
        simulation or that was imported.
    Dispersion.output_file : str
        String containing the filepath of the output file created during the
        simulation or that was imported.
    Dispersion.error_file : str
        String containing the filepath of the error file created during the
        simulation or that was imported.
    Dispersion.inputs_log : list
        List in which each item is a line of the input_file.
    Dispersion.outputs_log : list
        List in which each item is a line of the output_file.
    Dispersion.errors_log : list
        List in which each item is a line of the error_file.
    Dispersion.num_of_loaded_sims : int
        Number of simulations loaded from output_file being currently used.
    Dispersion.results : dict
        A dictionary containing all the output parameters saved from the flight
        simulations.
    Dispersion.processed_results : dict
        Dictionary containing (mean, std. dev.) for each parameter available
        in the dispersion dictionary.
    """

    def __init__(
        self,
        filename,
        environment,
        rocket,
        flight,
    ):
        """Constructor of the Dispersion class.

        Parameters
        ----------
        filename: string
            When running a new simulation, this parameter represents the initial
            part of the export filenames (e.g. 'filename.disp_outputs.txt').
            When analyzing the results of a previous simulation, this parameter
            shall be the .txt filename containing the outputs of a previous ran
            dispersion analysis.
        environment: McEnvironment
            The environment in which the rocket will be launched.
        rocket: McRocket
            The rocket to be launched.
        flight: McFlight
            The flight conditions of the rocket.

        Returns
        -------
        None
        """

        # Save and initialize parameters
        self.filename = filename
        self.environment = environment
        self.rocket = rocket
        self.flight = flight
        self.motors = rocket.motors
        self.nosecones = rocket.nosecones
        self.fins = rocket.fins
        self.tails = rocket.tails
        self.parachutes = rocket.parachutes
        self.rail_buttons = rocket.rail_buttons
        self.export_list = []
        self.inputs_log = []
        self.outputs_log = []
        self.errors_log = []
        self.num_of_loaded_sims = 0
        self.results = {}
        self.processed_results = {}
        self.prints = _DispersionPrints(self)
        self.plots = _DispersionPlots(self)

        try:
            self.import_inputs()
        except FileNotFoundError:
            self._input_file = f"{filename}.disp_inputs.txt"

        try:
            self.import_outputs()
        except FileNotFoundError:
            self._output_file = f"{filename}.disp_outputs.txt"

        try:
            self.import_errors()
        except FileNotFoundError:
            self._error_file = f"{filename}.disp_errors.txt"

        # TODO: Initialize variables so they can be accessed by MATLAB
        return None

    # getters and setters for dispersion input/output/error files
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
        # TODO: add pickle package to deal with parachute triggers and Function objects
        self.inputs_log = []
        with open(self.input_file, mode="r", encoding="utf-8") as disp_inputs:
            # Loop through each line in the file
            for line in disp_inputs:
                # swap "<" and ">" to "'"
                # this is done to interpret the trigger functions
                # Find the index of the first and last occurrences of '<' and '>'
                first_lt_index = line.find("<")
                last_gt_index = line.rfind(">")
                if first_lt_index != -1 and last_gt_index != -1:
                    # Replace the first '<' and last '>' with double quotes
                    line = (
                        line[:first_lt_index]
                        + '"'
                        + line[first_lt_index + 1 : last_gt_index]
                        + '"'
                        + line[last_gt_index + 1 :]
                    )

                # # Skip comments lines
                if line[0] != "{":
                    continue
                # Try to convert the line to a dictionary
                d = ast.literal_eval(line)
                # If successful, append the dictionary to the list
                self.inputs_log.append(d)
        return None

    def set_outputs_log(self):
        """Sets outputs_log from a file into an attribute for easy access"""
        self.outputs_log = []
        # Loop through each line in the file
        with open(self.output_file, mode="r", encoding="utf-8") as disp_outputs:
            for line in disp_outputs:
                # Skip comments lines
                if line[0] != "{":
                    continue
                # Try to convert the line to a dictionary
                d = ast.literal_eval(line)
                # If successful, append the dictionary to the list
                self.outputs_log.append(d)
        return None

    def set_errors_log(self):
        """Sets errors_log log from a file into an attribute for easy access"""
        self.errors_log = []
        # Loop through each line in the file
        with open(self.error_file, mode="r", encoding="utf-8") as disp_errors:
            for line in disp_errors:
                # Skip comments lines
                if line[0] != "{":
                    continue
                # Try to convert the line to a dictionary
                d = ast.literal_eval(line)
                # If successful, append the dictionary to the list
                self.errors_log.append(d)
        return None

    def set_num_of_loaded_sims(self):
        """Number of simulations loaded from output_file being currently used."""
        # Calculate the number of flights simulated
        self.num_of_loaded_sims = 0
        # Loop through each line in the file
        with open(self.output_file, mode="r", encoding="utf-8") as disp_outputs:
            for line in disp_outputs:
                # Skip comments lines
                if line[0] != "{":
                    continue
                self.num_of_loaded_sims += 1
        return None

    def set_results(self):
        """Dispersion results organized in a dictionary where the keys are the
        names of the saved attributes, and the values are a list with all the
        result number of the respective attribute"""
        self.results = {}
        for result in self.outputs_log:
            for key, value in result.items():
                if key in self.results.keys():
                    self.results[key].append(value)
                else:
                    self.results[key] = [value]
        return None

    def set_processed_results(self):
        """Creates a dictionary with the mean and standard deviation of each
        parameter available in the results

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.processed_results = {}
        for result in self.results.keys():
            mean = np.mean(self.results[result])
            stdev = np.std(self.results[result])
            self.processed_results[result] = (mean, stdev)
        return None

    # methods for running dispersion analysis
    def run_dispersion(
        self,
        number_of_simulations,
        export_list=None,
        append=False,
    ):
        """Runs the dispersion simulation and saves all data. For the simulation to be run
        all classes must be defined. This can happen either trough the dispersion_dictionary
        or by inputting objects

        Parameters
        ----------
        number_of_simulations : int
            Number of simulations to be run, must be non negative.
        export_list : list, optional
            A list containing the name of the attributes to be saved on the dispersion
            outputs file. See Examples for all possible attributes
        append : bool, optional
            If True, the results will be appended to the existing files. If False,
            the files will be overwritten. By default False.

        Returns
        -------
        None
        """

        # Saving the arguments as attributes
        self.number_of_simulations = number_of_simulations

        # Create data files for inputs, outputs and error logging
        open_mode = "a" if append else "w"
        input_file = open(self._input_file, open_mode, encoding="utf-8")
        output_file = open(self._output_file, open_mode, encoding="utf-8")
        error_file = open(self._error_file, open_mode, encoding="utf-8")

        # Checks export_list
        self.export_list = self.__check_export_list(export_list)

        # Initializes inputs_dict in case of error in the first iteration
        inputs_dict = {}

        # Initialize counter and timer
        i = self.num_of_loaded_sims if append else 0
        initial_wall_time = time()
        initial_cpu_time = process_time()

        # Begin display
        print("Starting", end="\r")

        # Start the flight simulations
        for _ in range(number_of_simulations):
            start_time = process_time()
            i += 1

            # Run trajectory simulation
            try:
                dispersion_flight = Flight(
                    rocket=self.rocket.create_object(),
                    environment=self.environment.create_object(),
                    inclination=self.flight.rnd_inclination(),
                    heading=self.flight.rnd_heading(),
                    initialSolution=self.flight.initialSolution,
                    terminateOnApogee=self.flight.terminateOnApogee,
                )

                # create inputs dictionary
                inputs_dict = dict(
                    item
                    for d in [
                        self.environment.last_rnd_dict,
                        self.rocket.last_rnd_dict,
                        self.flight.last_rnd_dict,
                    ]
                    for item in d.items()
                )
                # TODO: I believe the positions are not being saved
                # need to check if they are and fix if not
                if self.rocket.motors:
                    for motor in self.rocket.motors.get_components():
                        inputs_dict.update(motor.last_rnd_dict)
                if self.rocket.nosecones:
                    for nosecone in self.rocket.nosecones.get_components():
                        inputs_dict.update(nosecone.last_rnd_dict)
                if self.rocket.fins:
                    for fin in self.rocket.fins.get_components():
                        inputs_dict.update(fin.last_rnd_dict)
                if self.rocket.tails:
                    for tail in self.rocket.tails.get_components():
                        inputs_dict.update(tail.last_rnd_dict)
                if self.rocket.parachutes:
                    for parachute in self.rocket.parachutes:
                        inputs_dict.update(parachute.last_rnd_dict)
                if self.rocket.rail_buttons.get_components():
                    for rail_buttons in self.rocket.rail_buttons.get_components():
                        inputs_dict.update(rail_buttons.last_rnd_dict)
                # Export inputs and outputs to file
                self.__export_flight_data(
                    setting=inputs_dict,
                    flight=dispersion_flight,
                    input_file=input_file,
                    output_file=output_file,
                )
            except (TypeError, ValueError, KeyError, AttributeError) as error:
                print(f"Error on iteration {i}: {error}\n")
                error_file.write(f"{inputs_dict}\n")
                raise error
            except KeyboardInterrupt:
                print("Keyboard Interrupt, file saved.")
                error_file.write(f"{inputs_dict}\n")
                break

            # spaces after the last 's' are necessary to fix a bug with end='\r'
            print(
                f"Current iteration: {i:06d} | Average Time per Iteration: "
                f"{(process_time() - initial_cpu_time)/i:2.6f} s | Estimated time"
                f" left: {int((number_of_simulations - i)*((process_time() - initial_cpu_time)/i))} s      ",
                end="\r",
            )

        ## Print and save total time
        final_string = (
            f"Completed {i} iterations. Total CPU time: "
            f"{process_time() - initial_cpu_time:.1f} s. Total wall time: "
            f"{time() - initial_wall_time:.1f} s"
        )
        print(final_string, end="\r")

        # close files to guarantee saving
        input_file.close()
        output_file.close()
        error_file.close()

        # resave the files on self and calculate post simulation attributes
        self.input_file = f"{self.filename}.disp_inputs.txt"
        self.output_file = f"{self.filename}.disp_outputs.txt"
        self.error_file = f"{self.filename}.disp_errors.txt"

        return None

    # methods for exporting data
    def __check_export_list(self, export_list):
        """Check if export list is valid or if it is None. In case it is
        None, export a standard list of parameters.

        Parameters
        ----------
        export_list : list
            List of strings with the names of the attributes to be exported

        Returns
        -------
        export_list
        """
        standard_output = (
            "apogee",
            "apogeeTime",
            "apogeeX",
            "apogeeY",
            "apogeeFreestreamSpeed",
            "tFinal",
            "xImpact",
            "yImpact",
            "impactVelocity",
            "initialStaticMargin",
            "finalStaticMargin",
            "outOfRailStaticMargin",
            "outOfRailTime",
            "outOfRailVelocity",
            "maxSpeed",
            "maxMachNumber",
            "maxAcceleration",
            "frontalSurfaceWind",
            "lateralSurfaceWind",
        )
        exportables = (
            "inclination",
            "heading",
            "effective1RL",
            "effective2RL",
            "outOfRailTime",
            "outOfRailTimeIndex",
            "outOfRailState",
            "outOfRailVelocity",
            "railButton1NormalForce",
            "maxRailButton1NormalForce",
            "railButton1ShearForce",
            "maxRailButton1ShearForce",
            "railButton2NormalForce",
            "maxRailButton2NormalForce",
            "railButton2ShearForce",
            "maxRailButton2ShearForce",
            "outOfRailStaticMargin",
            "apogeeState",
            "apogeeTime",
            "apogeeX",
            "apogeeY",
            "apogee",
            "xImpact",
            "yImpact",
            "zImpact",
            "impactVelocity",
            "impactState",
            "parachuteEvents",
            "apogeeFreestreamSpeed",
            "finalStaticMargin",
            "frontalSurfaceWind",
            "initialStaticMargin",
            "lateralSurfaceWind",
            "maxAcceleration",
            "maxAccelerationTime",
            "maxDynamicPressureTime",
            "maxDynamicPressure",
            "maxMachNumberTime",
            "maxMachNumber",
            "maxReynoldsNumberTime",
            "maxReynoldsNumber",
            "maxSpeedTime",
            "maxSpeed",
            "maxTotalPressureTime",
            "maxTotalPressure",
            "tFinal",
        )
        if export_list:
            for attr in export_list:
                if not isinstance(attr, str):
                    raise TypeError("Variables in export_list must be strings.")

                # Checks if attribute is not valid
                if attr not in exportables:
                    raise ValueError(
                        "Attribute can not be exported. Check export_list."
                    )
        else:
            # No export list provided, using default list instead.
            export_list = standard_output

        return export_list

    def __export_flight_data(
        self,
        setting,
        flight,
        input_file,
        output_file,
    ):
        """Saves flight results in a .txt
        Parameters
        ----------
        setting : dict
            The flight setting used in the simulation.
        flight : Flight
            The flight object.
        input_file : str
            The name of the file containing all the inputs for the simulation.
        output_file : str
            The name of the file containing all the outputs for the simulation.
        Returns
        -------
        inputs_log : str
            The new string with the inputs of the simulation setting.
        outputs_log : str
            The new string with the outputs of the simulation setting.
        """
        # Construct the dict with the results from the flight
        results = {}
        for export_item in self.export_list:
            # if attribute is function, get source
            # TODO: check if there is a better way to do this
            if isinstance(getattr(flight, export_item), Function):
                results[export_item] = list(getattr(flight, export_item).source)
            else:
                results[export_item] = getattr(flight, export_item)

        # Write flight setting and results to file
        input_file.write(f"{setting}\n")
        output_file.write(f"{results}\n")

        return None

    # methods for importing data
    def import_outputs(self, filename=None):
        """Import dispersion results from .txt file and save it into a dictionary.

        Parameters
        ----------
        filename : str
            Name or directory path to the file to be imported. If none, Dispersion
            filename will be used

        Returns
        -------
        None
        """
        # select file to use
        filepath = filename if filename else self.filename

        try:
            with open(f"{filepath}.disp_outputs.txt", "r+", encoding="utf-8"):
                self.output_file = f"{filepath}.disp_outputs.txt"
                # Print the number of flights simulated
                print(
                    f"A total of {self.num_of_loaded_sims} simulations results were loaded from"
                    f" the following output file: {filepath}.disp_outputs.txt\n"
                )
        except FileNotFoundError:
            with open(filepath, "r+", encoding="utf-8"):
                self.output_file = filepath
                # Print the number of flights simulated
                print(
                    f"A total of {self.num_of_loaded_sims} simulations results were loaded from"
                    f" the following output file: {filepath}\n"
                )
        return None

    def import_inputs(self, filename=None):
        """Import dispersion results from .txt file and save it into a dictionary.

        Parameters
        ----------
        filename : str
            Name or directory path to the file to be imported. If none, Dispersion
            filename will be used

        Returns
        -------
        None
        """
        # select file to use
        filepath = filename if filename else self.filename

        try:
            with open(f"{filepath}.disp_inputs.txt", "r+", encoding="utf-8"):
                self.input_file = f"{filepath}.disp_inputs.txt"
                # Print the number of flights simulated
                print(
                    f"The following input file was imported: {filepath}.disp_inputs.txt\n"
                )
        except FileNotFoundError:
            with open(filepath, "r+", encoding="utf-8"):
                self.input_file = filepath
                # Print the number of flights simulated
                print(f"The following input file was imported: {filepath}\n")
        return None

    def import_errors(self, filename=None):
        """Import dispersion results from .txt file and save it into a dictionary.

        Parameters
        ----------
        filename : str
            Name or directory path to the file to be imported. If none, Dispersion
            filename will be used

        Returns
        -------
        None
        """
        # select file to use
        filepath = filename if filename else self.filename

        try:
            with open(f"{filepath}.disp_errors.txt", "r+", encoding="utf-8"):
                self.error_file = f"{filepath}.disp_errors.txt"
                # Print the number of flights simulated
                print(
                    f"The following error file was imported: {filepath}.disp_errors.txt\n"
                )
        except FileNotFoundError:
            with open(filepath, "r+", encoding="utf-8"):
                self.error_file = filepath
                # Print the number of flights simulated
                print(f"The following error file was imported: {filepath}\n")
        return None

    def import_results(self, filename=None):
        """Import dispersion results from .txt file and save it into a dictionary.

        Parameters
        ----------
        filename : str
            Name or directory path to the file to be imported. If none, Dispersion
            filename will be used

        Returns
        -------
        None
        """
        # select file to use
        filepath = filename if filename else self.filename

        self.import_outputs(filename=filepath)
        self.import_inputs(filename=filepath)
        self.import_errors(filename=filepath)

        return None

    def exportEllipsesToKML(
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
            Contains dispersion results from the Monte Carlo simulation.
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
            _,
            _,
            _,
            _,
        ) = generate_dispersion_ellipses(self.results)
        outputs = []

        if type == "all" or type == "impact":
            outputs = outputs + generate_dispersion_ellipses_coordinates(
                impact_ellipses, origin_lat, origin_lon, resolution=resolution
            )

        if type == "all" or type == "apogee":
            outputs = outputs + generate_dispersion_ellipses_coordinates(
                apogee_ellipses, origin_lat, origin_lon, resolution=resolution
            )

        # Prepare data to KML file
        kml_data = []
        for i in range(len(outputs)):
            temp = []
            for j in range(len(outputs[i])):
                temp.append((outputs[i][j][1], outputs[i][j][0]))  # lon, lat
            kml_data.append(temp)

        # Export to KML
        kml = simplekml.Kml()

        for i in range(len(outputs)):
            if (type == "all" and i < 3) or (type == "impact"):
                ellName = "Impact σ" + str(i + 1)
            elif type == "all" and i >= 3:
                ellName = "Apogee σ" + str(i - 2)
            else:
                ellName = "Apogee σ" + str(i + 1)

            mult_ell = kml.newmultigeometry(name=ellName)
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
        return None

    # methods for printing and plotting results

    def info(self):
        """Print information about the dispersion model.

        Returns
        -------
        None
        """
        self.prints.all_results()

        return None

    def allInfo(self):
        """Print and plot information about the dispersion model and the results.

        Returns
        -------
        None
        """
        self.info()
        self.plots.ellipses()
        self.plots.all_results()

        return None
