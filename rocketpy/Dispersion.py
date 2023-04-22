__author__ = "Mateus Stano Junqueira, Sofia Lopes Suesdek Rocha, Guilherme Fernandes Alves, Bruno Abdulklech Sorban"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"


import ast
import math
from random import choice
from time import process_time, time

import matplotlib.pyplot as plt
import numpy as np
import simplekml
from matplotlib.patches import Ellipse

from .Environment import Environment
from .Flight import Flight
from .Function import Function
from .Motor import SolidMotor
from .Rocket import Rocket
from .tools import get_distribution, invertedHaversine

try:
    from functools import cached_property
except ImportError:
    from .tools import cached_property

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
    Dispersion.num_of_loaded_sims : int
        Number of simulations loaded from a previous dispersion analysis.
    Dispersion.number_of_simulations : int
        Number of simulations to be performed in the run_dispersion() method.
    Dispersion.dispersion_dictionary : dict
        Dictionary containing the parameters to be used in the Monte Carlo
        simulations.
    Dispersion.export_list : list
        List of parameters to be exported from each flight in the Monte Carlo
        loop.
    Dispersion.dispersion_results : dict
        A dictionary containing all the output parameters saved from the flight
        simulations.
    Dispersion.processed_dispersion_results : dict
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
        return self._input_file

    @input_file.setter
    def input_file(self, value):
        self._input_file = value
        self.set_inputs_log()

    @property
    def output_file(self):
        return self._output_file

    @output_file.setter
    def output_file(self, value):
        self._output_file = value
        self.set_outputs_log()
        self.set_num_of_loaded_sims()
        self.set_results()
        self.set_processed_results()

    @property
    def error_file(self):
        return self._error_file

    @error_file.setter
    def error_file(self, value):
        self._error_file = value
        self.set_errors_log()

    # setters for post simulation attributes
    def set_inputs_log(self):
        """Save inputs_log from a file into an attribute for easy access"""
        # TODO: add pickle package to deal with parachute triggers and Function objects
        self.inputs_log = []
        with open(self.input_file, mode="r", encoding="utf-8") as disp_inputs:
            # Loop through each line in the file
            for line in disp_inputs:
                # swap "<" and ">" to "'"
                # this is done to interpret the trigger functions
                line = line.replace("<", "'")
                line = line.replace(">", "'")
                # # Skip comments lines
                if line[0] != "{":
                    continue
                # Try to convert the line to a dictionary
                d = ast.literal_eval(line)
                # If successful, append the dictionary to the list
                self.inputs_log.append(d)
        return None

    def set_outputs_log(self):
        """Save outputs_log from a file into an attribute for easy access"""
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
        """Save errors_log log from a file into an attribute for easy access"""
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
        processed_results: dict
            A dictionary with the mean and standard deviation of each parameter
            available in the results dictionary.
        """
        self.processed_results = {}
        for result in self.results.keys():
            mean = np.mean(self.results[result])
            stdev = np.std(self.results[result])
            self.processed_results[result] = (mean, stdev)
        return None

    # methods for running dispersion analysis
    def __yield_flight_setting(self, dispersion_dictionary, number_of_simulations):
        """Yields a flight setting for the simulation

        Parameters
        ----------
        dispersion_dictionary : dict
            The dictionary with the parameters to be analyzed. This includes the
            mean and standard deviation of the parameters.
        number_of_simulations : int
            Number of simulations desired, must be non negative.
            This is needed when running a new simulation. Default is zero.

        Yields
        ------
        setting: dict
            A dictionary with the flight setting for one simulation.
        """

        for _ in range(number_of_simulations):
            setting = {}
            for class_name, data in dispersion_dictionary.items():
                setting[class_name] = {}
                for key, value in data.items():
                    if isinstance(value, tuple):
                        setting[class_name][key] = value[-1](value[0], value[1])
                    elif isinstance(value, list):
                        setting[class_name][key] = choice(value) if value else value
                    else:
                        # else is dictionary
                        setting[class_name][key] = {}
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, tuple):
                                setting[class_name][key][sub_key] = sub_value[-1](
                                    sub_value[0], sub_value[1]
                                )
                            else:
                                # else is list
                                # setting[class_name][key][sub_key] = choice(sub_value)
                                # The choice() doesn't work when you have Functions
                                setting[class_name][key][sub_key] = (
                                    choice(sub_value) if sub_value else sub_value
                                )

            yield setting

    def __convert_field(self, value):
        """Receives a value provided by a McXxxxx and returns it in a form that
        can be used in run_dispersion. There are two possible cases:
        1. The value is a list: simply return the list
        2. The value is a tuple: return a tuple containing the mean, std and
        a np.random function. The default distribution is normal, but it can
        be changed by adding a string to the end of the tuple. For example:
        (mean, std, 'uniform') will return a tuple with the mean, std and
        np.random.uniform function.

        Parameters
        ----------
        value : tuple or list
            The value to be converted to a monte carlo tuple

        Returns
        -------
        tuple
            Mean, standard deviation and np.random function
        """
        if isinstance(value, list):
            return value
        elif isinstance(value, tuple):
            # the normal distribution is considered the default
            dist_func = (
                get_distribution(value[-1])
                if isinstance(value[-1], str)
                else np.random.normal
            )
            return (value[0], value[1], dist_func)

    def __pop_none(self, dictionary):
        """Removes all the keys that have a value of None. This is useful
        when the user wants to run a simulation with a subset of the
        available parameters.

        Parameters
        ----------
        dictionary : dict
            The dictionary to be cleaned

        Returns
        -------
        dictionary
            The cleaned dictionary
        """
        for key, value in dictionary.copy().items():
            if value is None:
                dictionary.pop(key)
            elif isinstance(value, dict):
                self.__pop_none(value)
        return dictionary

    def build_dispersion_dict(
        self,
    ):
        """Creates a dictionary to be used in the dispersion analysis. As a
        static method, it can be used without instantiating a Dispersion object.

        Parameters
        ----------
        None

        Returns
        -------
        mc_dict : dict
            A dictionary containing all the information needed to run a dispersion
            analysis. This dictionary has the following structure:
            mc_dict = {
                "environment": {
                    "railLength": (mean, std, np.random.func),
                    "date": [mean],
                },
                "rocket": {
                    'radius': (mean, std, np.random.func),
                    'mass': (mean, std, np.random.func),
                    'inertiaI': (mean, std, np.random.func)
                    },
                "motors": {},
                "flight": {},
                "nosecones": {},
                "fins": {},
                "tails": {},
                "parachutes": {
                    0: {
                        "CdS": (mean, std, np.random.func),
                        "lag": (mean, std, np.random.func),
                        ...
                    },
                    1: {...}
                },
                "buttons": {},
            }
        """

        mc_dict = {
            "environment": self.environment.dict(),
            "rocket": self.rocket.dict(),
            "flight": self.flight.dict(),
            "motors": {i: motor.dict() for i, motor in enumerate(self.motors)},
            "nosecones": {
                i: nosecone.dict() for i, nosecone in enumerate(self.nosecones)
            },
            "fins": {i: fin.dict() for i, fin in enumerate(self.fins)},
            "tails": {i: tail.dict() for i, tail in enumerate(self.tails)},
            "parachutes": {
                i: parachute.dict() for i, parachute in enumerate(self.parachutes)
            },
            "rail_buttons": {
                i: buttons.dict() for i, buttons in enumerate(self.rail_buttons)
            },
        }

        for _, data in mc_dict.items():
            # Checks if first key of dict is a number
            # for instance: parachutes : {0: McParachute.dict(), 1: McParachute}
            # therefore we need to iterate over the dicts
            if 0 in data.keys():
                for sub_data in data.values():
                    for field, value in sub_data.items():
                        sub_data[field] = self.__convert_field(value)
            else:
                for field, value in data.items():
                    data[field] = self.__convert_field(value)

        # pop any value that is None, even if it is a value of sub dictionary
        mc_dict = self.__pop_none(mc_dict)

        return mc_dict

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

        # Creates dispersion dictionary
        self.dispersion_dictionary = self.build_dispersion_dict()

        # Create data files for inputs, outputs and error logging
        open_mode = "a" if append else "w"
        input_file = open(self._input_file, open_mode, encoding="utf-8")
        output_file = open(self._output_file, open_mode, encoding="utf-8")
        error_file = open(self._error_file, open_mode, encoding="utf-8")

        # Checks export_list
        self.export_list = self.__check_export_list(export_list)

        # Initialize counter and timer
        i = self.num_of_loaded_sims
        initial_wall_time = time()
        initial_cpu_time = process_time()

        # Begin display
        print("Starting", end="\r")

        # Initialize initial environment and gets original wind data
        env_dispersion = (
            self.environment
            if isinstance(self.environment, Environment)
            else self.environment.__dict__["environment"]
        )
        # Saving original wind profiles
        windX = env_dispersion.windVelocityX
        windY = env_dispersion.windVelocityY

        # Iterate over flight settings, start the flight simulations
        for setting in self.__yield_flight_setting(
            self.dispersion_dictionary, self.number_of_simulations
        ):
            start_time = process_time()
            i += 1

            # Apply environment parameters variations on each iteration if possible
            env_dispersion.railLength = setting["environment"]["railLength"]
            env_dispersion.windVelocityX = windX * setting["environment"]["windXFactor"]
            env_dispersion.windVelocityY = windY * setting["environment"]["windYFactor"]
            if setting["environment"]["ensembleMember"]:
                env_dispersion.selectEnsembleMember(
                    setting["environment"]["ensembleMember"]
                )

            # Apply motor parameters variations on each iteration if possible
            # TODO: add hybrid and liquid motor option
            # TODO: only a single motor is supported. Future version should support more.
            motor_dispersion = SolidMotor(
                thrustSource=setting["motors"][0]["thrustSource"],
                burnOutTime=setting["motors"][0]["burnOutTime"],
                grainsCenterOfMassPosition=setting["motors"][0][
                    "grainsCenterOfMassPosition"
                ],
                grainNumber=setting["motors"][0]["grainNumber"],
                grainDensity=setting["motors"][0]["grainDensity"],
                grainOuterRadius=setting["motors"][0]["grainOuterRadius"],
                grainInitialInnerRadius=setting["motors"][0]["grainInitialInnerRadius"],
                grainInitialHeight=setting["motors"][0]["grainInitialHeight"],
                grainSeparation=setting["motors"][0]["grainSeparation"],
                nozzleRadius=setting["motors"][0]["nozzleRadius"],
                nozzlePosition=setting["motors"][0]["nozzlePosition"],
                throatRadius=setting["motors"][0]["throatRadius"],
                reshapeThrustCurve=(
                    setting["motors"][0]["burnOutTime"],
                    setting["motors"][0]["totalImpulse"],
                ),
                # interpolationMethod="linear",
                # coordinateSystemOrientation=setting["motors"][0][
                #     "coordinateSystemOrientation"
                # ],
            )

            # Apply rocket parameters variations on each iteration if possible
            rocket_dispersion = Rocket(
                radius=setting["rocket"]["radius"],
                mass=setting["rocket"]["mass"],
                inertiaI=setting["rocket"]["inertiaI"],
                inertiaZ=setting["rocket"]["inertiaZ"],
                powerOffDrag=setting["rocket"]["powerOffDrag"],
                powerOnDrag=setting["rocket"]["powerOnDrag"],
                centerOfDryMassPosition=setting["rocket"]["centerOfDryMassPosition"],
                # coordinateSystemOrientation=setting["rocket"][
                #     "coordinateSystemOrientation"
                # ],
            )

            # Edit rocket drag
            rocket_dispersion.powerOffDrag *= setting["rocket"]["powerOffDragFactor"]
            rocket_dispersion.powerOnDrag *= setting["rocket"]["powerOnDragFactor"]

            # Add Motor
            rocket_dispersion.addMotor(
                motor_dispersion, position=setting["motors"][0]["position"]
            )

            # Nose
            for nose in setting["nosecones"].keys():
                rocket_dispersion.addNose(
                    kind=setting["nosecones"][nose]["kind"],
                    length=setting["nosecones"][nose]["length"],
                    position=setting["nosecones"][nose]["position"],
                    name=nose,
                )

            # Fins
            for fin in setting["fins"].keys():
                if "sweepLength" in setting["fins"][fin].keys():
                    # means that it is trapezoidal
                    rocket_dispersion.addTrapezoidalFins(
                        n=setting["fins"][fin]["n"],
                        rootChord=setting["fins"][fin]["rootChord"],
                        tipChord=setting["fins"][fin]["tipChord"],
                        span=setting["fins"][fin]["span"],
                        position=setting["fins"][fin]["position"],
                        cantAngle=setting["fins"][fin]["cantAngle"],
                        sweepLength=setting["fins"][fin]["sweepLength"],
                        # sweepAngle=setting["fins"][fin]["sweepAngle"],
                        radius=setting["fins"][fin]["rocketRadius"],
                        airfoil=setting["fins"][fin]["airfoil"],
                        name=fin,
                    )
                else:
                    rocket_dispersion.addEllipticalFins(
                        n=setting["fins"][fin]["n"],
                        rootChord=setting["fins"][fin]["rootChord"],
                        span=setting["fins"][fin]["span"],
                        position=setting["fins"][fin]["position"],
                        cantAngle=setting["fins"][fin]["cantAngle"],
                        radius=setting["fins"][fin]["radius"],
                        airfoil=setting["fins"][fin]["airfoil"],
                        name=fin,
                    )

            # Tail
            for tail in setting["tails"].keys():
                rocket_dispersion.addTail(
                    length=setting["tails"][tail]["length"],
                    position=setting["tails"][tail]["position"],
                    topRadius=setting["tails"][tail]["topRadius"],
                    bottomRadius=setting["tails"][tail]["bottomRadius"],
                    # radius=setting["tails"][tail]["radius"],
                    # TODO: understand if we need vary this radius argument
                    name=tail,
                )

            # Rail Buttons
            for rail_buttons in setting["rail_buttons"].keys():
                rocket_dispersion.setRailButtons(
                    position=[
                        setting["rail_buttons"][rail_buttons]["upper_button_position"],
                        setting["rail_buttons"][rail_buttons]["lower_button_position"],
                    ],
                    angular_position=setting["rail_buttons"][rail_buttons][
                        "angular_position"
                    ],
                )

            # Add parachutes
            for chute in setting["parachutes"].keys():
                rocket_dispersion.addParachute(
                    name=chute,
                    CdS=setting["parachutes"][chute]["CdS"],
                    trigger=setting["parachutes"][chute]["trigger"],
                    samplingRate=setting["parachutes"][chute]["samplingRate"],
                    lag=setting["parachutes"][chute]["lag"],
                    noise=setting["parachutes"][chute]["noise"],
                )

            # Run trajectory simulation
            try:
                # TODO: Add initialSolution flight option
                dispersion_flight = Flight(
                    rocket=rocket_dispersion,
                    environment=env_dispersion,
                    inclination=setting["flight"]["inclination"],
                    heading=setting["flight"]["heading"],
                )

                # Export inputs and outputs to file
                self.__export_flight_data(
                    setting=setting,
                    flight=dispersion_flight,
                    input_file=input_file,
                    output_file=output_file,
                )
            except (TypeError, ValueError, KeyError, AttributeError) as error:
                print(f"Error on iteration {i}: {error}\n")
                error_file.write(f"{setting}\n")
            except KeyboardInterrupt:
                print("Keyboard Interrupt, file saved.")
                error_file.write(f"{setting}\n")
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

    # methods for ellipses
    def __createEllipses(self, results):
        """A function to create apogee and impact ellipses from the dispersion
        results.

        Parameters
        ----------
        results : dict
            A dictionary containing the results of the dispersion analysis.

        Returns
        -------
        apogee_ellipse : Ellipse
            An ellipse object representing the apogee ellipse.
        impact_ellipse : Ellipse
            An ellipse object representing the impact ellipse.
        apogeeX : np.array
            An array containing the x coordinates of the apogee ellipse.
        apogeeY : np.array
            An array containing the y coordinates of the apogee ellipse.
        impactX : np.array
            An array containing the x coordinates of the impact ellipse.
        impactY : np.array
            An array containing the y coordinates of the impact ellipse.
        """

        # Retrieve dispersion data por apogee and impact XY position
        try:
            apogeeX = np.array(results["apogeeX"])
            apogeeY = np.array(results["apogeeY"])
        except KeyError:
            print("No apogee data found.")
            apogeeX = np.array([])
            apogeeY = np.array([])
        try:
            impactX = np.array(results["xImpact"])
            impactY = np.array(results["yImpact"])
        except KeyError:
            print("No impact data found.")
            impactX = np.array([])
            impactY = np.array([])

        # Define function to calculate eigen values
        def eigsorted(cov):
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            return vals[order], vecs[:, order]

        # Calculate error ellipses for impact
        impactCov = np.cov(impactX, impactY)
        impactVals, impactVecs = eigsorted(impactCov)
        impactTheta = np.degrees(np.arctan2(*impactVecs[:, 0][::-1]))
        impactW, impactH = 2 * np.sqrt(impactVals)

        # Draw error ellipses for impact
        impact_ellipses = []
        for j in [1, 2, 3]:
            impactEll = Ellipse(
                xy=(np.mean(impactX), np.mean(impactY)),
                width=impactW * j,
                height=impactH * j,
                angle=impactTheta,
                color="black",
            )
            impactEll.set_facecolor((0, 0, 1, 0.2))
            impact_ellipses.append(impactEll)

        # Calculate error ellipses for apogee
        apogeeCov = np.cov(apogeeX, apogeeY)
        apogeeVals, apogeeVecs = eigsorted(apogeeCov)
        apogeeTheta = np.degrees(np.arctan2(*apogeeVecs[:, 0][::-1]))
        apogeeW, apogeeH = 2 * np.sqrt(apogeeVals)

        apogee_ellipses = []
        # Draw error ellipses for apogee
        for j in [1, 2, 3]:
            apogeeEll = Ellipse(
                xy=(np.mean(apogeeX), np.mean(apogeeY)),
                width=apogeeW * j,
                height=apogeeH * j,
                angle=apogeeTheta,
                color="black",
            )
            apogeeEll.set_facecolor((0, 1, 0, 0.2))
            apogee_ellipses.append(apogeeEll)
        return impact_ellipses, apogee_ellipses, apogeeX, apogeeY, impactX, impactY

    def plotEllipses(
        self,
        image=None,
        actual_landing_point=None,
        perimeterSize=3000,
        xlim=(-3000, 3000),
        ylim=(-3000, 3000),
        save=False,
    ):
        """A function to plot the error ellipses for the apogee and impact
        points of the rocket. The function also plots the real landing point, if
        given

        Parameters
        ----------
        image : str, optional
            The path to the image to be used as the background
        actual_landing_point : tuple, optional
            A tuple containing the actual landing point of the rocket, if known.
            Useful when comparing the dispersion results with the actual landing.
            Must be given in tuple format, such as (x, y) in meters.
            By default None.
        perimeterSize : int, optional
            The size of the perimeter to be plotted. The default is 3000.
        xlim : tuple, optional
            The limits of the x axis. The default is (-3000, 3000).
        ylim : tuple, optional
            The limits of the y axis. The default is (-3000, 3000).
        save : bool
            Whether save the output into a file or not. The default is False.
            If True, the .show() method won't be called, and the image will be
            saved with the same name as filename attribute, using a .png format.

        Returns
        -------
        None
        """
        # Import background map
        if image is not None:
            try:
                from imageio import imread

                img = imread(image)
            except ImportError:
                raise ImportError(
                    "The 'imageio' package could not be. Please install it to add background images."
                )
            except FileNotFoundError:
                raise FileNotFoundError(
                    "The image file was not found. Please check the path."
                )

        (
            impact_ellipses,
            apogee_ellipses,
            apogeeX,
            apogeeY,
            impactX,
            impactY,
        ) = self.__createEllipses(self.results)

        # Create plot figure
        plt.figure(num=None, figsize=(8, 6), dpi=150, facecolor="w", edgecolor="k")
        ax = plt.subplot(111)

        for ell in impact_ellipses:
            ax.add_artist(ell)
        for ell in apogee_ellipses:
            ax.add_artist(ell)

        # Draw launch point
        plt.scatter(0, 0, s=30, marker="*", color="black", label="Launch Point")
        # Draw apogee points
        plt.scatter(
            apogeeX, apogeeY, s=5, marker="^", color="green", label="Simulated Apogee"
        )
        # Draw impact points
        plt.scatter(
            impactX,
            impactY,
            s=5,
            marker="v",
            color="blue",
            label="Simulated Landing Point",
        )
        # Draw real landing point
        if actual_landing_point != None:
            plt.scatter(
                actual_landing_point[0],
                actual_landing_point[1],
                s=20,
                marker="X",
                color="red",
                label="Measured Landing Point",
            )

        plt.legend()

        # Add title and labels to plot
        ax.set_title(
            "1$\sigma$, 2$\sigma$ and 3$\sigma$ Dispersion Ellipses: Apogee and Lading Points"
        )
        ax.set_ylabel("North (m)")
        ax.set_xlabel("East (m)")

        # Add background image to plot
        # TODO: In the future, integrate with other libraries to plot the map (e.g. cartopy, ee, etc.)
        # You can translate the basemap by changing dx and dy (in meters)
        dx = 0
        dy = 0
        if image is not None:
            plt.imshow(
                img,
                zorder=0,
                extent=[
                    -perimeterSize - dx,
                    perimeterSize - dx,
                    -perimeterSize - dy,
                    perimeterSize - dy,
                ],
            )
        plt.axhline(0, color="black", linewidth=0.5)
        plt.axvline(0, color="black", linewidth=0.5)
        plt.xlim(*xlim)
        plt.ylim(*ylim)

        # Save plot and show result
        if save:
            plt.savefig(str(self.filename) + ".png", bbox_inches="tight", pad_inches=0)
        else:
            plt.show()
        return None

    def __prepareEllipses(self, ellipses, origin_lat, origin_lon, resolution=100):
        """Generate a list of latitude and longitude points for each ellipse in
        ellipses.

        Parameters
        ----------
        ellipses : list
            List of matplotlib.patches.Ellipse objects.
        origin_lat : float
            Latitude of the origin of the coordinate system.
        origin_lon : float
            Longitude of the origin of the coordinate system.
        resolution : int, optional
            Number of points to generate for each ellipse, by default 100

        Returns
        -------
        list
            List of lists of tuples containing the latitude and longitude of each
            point in each ellipse.
        """
        outputs = []

        for ell in ellipses:
            # Get ellipse path points
            center = ell.get_center()
            width = ell.get_width()
            height = ell.get_height()
            angle = np.deg2rad(ell.get_angle())
            points = []

            for i in range(resolution):
                x = width / 2 * math.cos(2 * np.pi * i / resolution)
                y = height / 2 * math.sin(2 * np.pi * i / resolution)
                x_rot = center[0] + x * math.cos(angle) - y * math.sin(angle)
                y_rot = center[1] + x * math.sin(angle) + y * math.cos(angle)
                points.append((x_rot, y_rot))
            points = np.array(points)

            # Convert path points to lat/lon
            lat_lon_points = []
            for point in points:
                x = point[0]
                y = point[1]

                # Convert to distance and bearing
                d = math.sqrt((x**2 + y**2))
                bearing = math.atan2(
                    x, y
                )  # math.atan2 returns the angle in the range [-pi, pi]

                lat_lon_points.append(
                    invertedHaversine(
                        origin_lat, origin_lon, d, bearing, eRadius=6.3781e6
                    )
                )

            # Export string
            outputs.append(lat_lon_points)
        return outputs

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
        ) = self.__createEllipses(self.results)
        outputs = []

        if type == "all" or type == "impact":
            outputs = outputs + self.__prepareEllipses(
                impact_ellipses, origin_lat, origin_lon, resolution=resolution
            )

        if type == "all" or type == "apogee":
            outputs = outputs + self.__prepareEllipses(
                apogee_ellipses, origin_lat, origin_lon, resolution=resolution
            )

        # Prepare data to KML file
        kml_data = []
        for i in range(len(outputs)):
            temp = []
            for j in range(len(outputs[i])):
                temp.append((outputs[i][j][1], outputs[i][j][0]))  # log, lat
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
    def print_results(self):
        """Print the mean and standard deviation of each parameter in the results
        dictionary or of the variables passed as argument.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        print("{:>25} {:>15} {:>15}".format("Parameter", "Mean", "Std. Dev."))
        print("-" * 60)
        for key, value in self.processed_results.items():
            print("{:>25} {:>15.3f} {:>15.3f}".format(key, value[0], value[1]))

        return None

    def plot_results(self):
        """Plot the results of the dispersion analysis.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        for key in self.results.keys():
            plt.figure()
            plt.hist(
                self.results[key],
            )
            plt.title("Histogram of " + key)
            plt.ylabel("Number of Occurrences")
            plt.show()

        return None

    def info(self):
        """Print information about the dispersion model.

        Returns
        -------
        None
        """

        print("Monte Carlo Simulation by RocketPy")
        print("Data Source: ", self.filename)
        print("Number of simulations: ", self.num_of_loaded_sims)
        print("Results: ")
        self.print_results()

        return None

    def allInfo(self):
        """Print and plot information about the dispersion model and the results.

        Returns
        -------
        None
        """
        self.info()
        print("Plotting results: ")
        self.plotEllipses()
        self.plot_results()

        return None
