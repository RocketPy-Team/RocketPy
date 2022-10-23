# -*- coding: utf-8 -*-

__author__ = "Mateus Stano Junqueira, Sofia Lopes Suesdek Rocha, Guilherme Fernandes Alves, Bruno Abdulklech Sorban"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"


import math
import traceback
import types
import warnings
from time import process_time, time

import matplotlib.pyplot as plt
import numpy as np
import simplekml
from imageio import imread
from IPython.display import display
from matplotlib.patches import Ellipse
from numpy.random import *

from .Environment import Environment
from .Flight import Flight
from .Function import Function
from .Motor import SolidMotor
from .Rocket import Rocket
from .supplement import invertedHaversine
from .AeroSurfaces import NoseCone, TrapezoidalFins, EllipticalFins, Tail

## Tasks from the first review:
# TODO: Save instances of the class instead of just plotting
# TODO: Document all methods
# TODO: Create a way to choose what attributes are being saved
# TODO: Allow each parameter to be varied following an specific probability distribution
# TODO: Make it more flexible so we can work with more than 1 fin set, also with different aerodynamic surfaces as well.
# TODO: Test simulations under different scenarios (with both parachutes, with only main chute, etc)
# TODO: Add unit tests
# TODO: Adjust the notebook to the new version of the code
# TODO: Optional return of matplotlib plots or abstract function to histogram plot based on stdev and mean

# TODO: Implement MRS
# TODO: Implement functions from compareDispersions notebook

# TODO: Convert the dictionary to a class attributes


class Dispersion:

    """Monte Carlo analysis to predict probability distributions of the rocket's
    landing point, apogee and other relevant information.

    Attributes
    ----------
    # TODO: Update at the end!
        Parameters:
        Dispersion.filename: string
            When running a new simulation, this attribute represents the initial
            part of the export filenames (e.g. 'filename.disp_outputs.txt').
            When analyzing the results of a previous simulation, this attribute
            shall be the filename containing the outputs of a dispersion calculation.
        Dispersion.actual_landing_point: tuple
            Rocket's experimental landing point relative to launch point.
        Dispersion.N: integer
            Number of simulations in an output file.
        Other classes:
        Dispersion.environment: Environment
            Launch environment.
            Attribute needed to run a new simulation, when Dispersion.flight remains unchanged.
        Dispersion.motor: Motor
            Rocket's motor.
            Attribute needed to run a new simulation, when Dispersion.flight remains unchanged.
        Dispersion.rocket: Rocket
            Rocket with nominal values.
            Attribute needed to run a new simulation, when Dispersion.flight remains unchanged.
    """

    def __init__(
        self,
        filename,
    ):

        """
        Parameters
        ----------
        filename: string
            When running a new simulation, this parameter represents the initial
            part of the export filenames (e.g. 'filename.disp_outputs.txt').
            When analyzing the results of a previous simulation, this parameter
            shall be the .txt filename containing the outputs of a previous ran
            dispersion analysis.

        Returns
        -------
        None
        """

        # Save  and initialize parameters
        self.filename = filename

        # Initialize variables to be used in the analysis in case of missing inputs
        self.environment_inputs = {
            "railLength": "required",
            "gravity": 9.80665,
            "date": None,
            "latitude": 0,
            "longitude": 0,
            "elevation": 0,
            "datum": "WGS84",
            "timeZone": "UTC",
        }

        self.solid_motor_inputs = {
            "thrust": "required",
            "burnOutTime": "required",
            "totalImpulse": 0,
            "grainNumber": "required",
            "grainDensity": "required",
            "grainOuterRadius": "required",
            "grainInitialInnerRadius": "required",
            "grainInitialHeight": "required",
            "grainSeparation": 0,
            "nozzleRadius": 0.0335,
            "throatRadius": 0.0114,
        }

        self.rocket_inputs = {
            "mass": "required",
            "inertiaI": "required",
            "inertiaZ": "required",
            "radius": "required",
            "distanceRocketNozzle": "required",
            "distanceRocketPropellant": "required",
            "powerOffDrag": "required",
            "powerOnDrag": "required",
        }

        self.nose_inputs = {
            "nose_name_length": "required",
            "nose_name_kind": "Von Karman",
            "nose_name_distanceToCM": "required",
            "nose_name_name": "Nose Cone",
        }

        self.fins_inputs = {
            "finSet_name_numberOfFins": "required",
            "finSet_name_rootChord": "required",
            "finSet_name_tipChord": "required",
            "finSet_name_span": "required",
            "finSet_name_distanceToCM": "required",
            "finSet_name_cantAngle": 0,
            "finSet_name_radius": None,
            "finSet_name_airfoil": None,
        }

        self.tail_inputs = {
            "tail_name_topRadius": "required",
            "tail_name_bottomRadius": "required",
            "tail_name_length": "required",
            "tail_name_distanceToCM": "required",
        }

        self.rail_buttons_inputs = {
            "positionFirstRailButton": "required",
            "positionSecondRailButton": "required",
            "railButtonAngularPosition": 45,
        }

        self.parachute_inputs = {
            "parachute_name_CdS": "required",
            "parachute_name_trigger": "required",
            "parachute_name_samplingRate": 100,
            "parachute_name_lag": 0,
            "parachute_name_noise": (0, 0, 0),
            # "parachute_name_noiseStd": 0,
            # "parachute_name_noiseCorr": 0,
        }

        self.flight_inputs = {
            "inclination": 80,
            "heading": 90,
            "initialSolution": None,
            "terminateOnApogee": False,
            "maxTime": 600,
            "maxTimeStep": np.inf,
            "minTimeStep": 0,
            "rtol": 1e-6,
            "atol": 6 * [1e-3] + 4 * [1e-6] + 3 * [1e-3],
            "timeOvershoot": True,
            "verbose": False,
        }

        # Initialize variables so they can be accessed by MATLAB
        self.dispersion_results = {}
        self.mean_out_of_rail_time = 0
        self.std_out_of_rail_time = 0

    def __set_distribution_function(self, distribution_type):
        """_summary_

        Parameters
        ----------
        distribution_type : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        if distribution_type == "normal" or distribution_type == None:
            return normal
        elif distribution_type == "beta":
            return beta
        elif distribution_type == "binomial":
            return binomial
        elif distribution_type == "chisquare":
            return chisquare
        elif distribution_type == "dirichlet":
            return dirichlet
        elif distribution_type == "exponential":
            return exponential
        elif distribution_type == "f":
            return f
        elif distribution_type == "gamma":
            return gamma
        elif distribution_type == "geometric":
            return geometric
        elif distribution_type == "gumbel":
            return gumbel
        elif distribution_type == "hypergeometric":
            return hypergeometric
        elif distribution_type == "laplace":
            return laplace
        elif distribution_type == "logistic":
            return logistic
        elif distribution_type == "lognormal":
            return lognormal
        elif distribution_type == "logseries":
            return logseries
        elif distribution_type == "multinomial":
            return multinomial
        elif distribution_type == "multivariate_normal":
            return multivariate_normal
        elif distribution_type == "negative_binomial":
            return negative_binomial
        elif distribution_type == "noncentral_chisquare":
            return noncentral_chisquare
        elif distribution_type == "noncentral_f":
            return noncentral_f
        elif distribution_type == "pareto":
            return pareto
        elif distribution_type == "poisson":
            return poisson
        elif distribution_type == "power":
            return power
        elif distribution_type == "rayleigh":
            return rayleigh
        elif distribution_type == "standard_cauchy":
            return standard_cauchy
        elif distribution_type == "standard_exponential":
            return standard_exponential
        elif distribution_type == "standard_gamma":
            return standard_gamma
        elif distribution_type == "standard_normal":
            return standard_normal
        elif distribution_type == "standard_t":
            return standard_t
        elif distribution_type == "triangular":
            return triangular
        elif distribution_type == "uneliform":
            return uniform
        elif distribution_type == "vonmises":
            return vonmises
        elif distribution_type == "wald":
            return wald
        elif distribution_type == "weibull":
            return weibull
        elif distribution_type == "zipf":
            return zipf
        else:
            warnings.warn("Distribution type not supported")

    def __process_dispersion_dict(self, dictionary):
        """Read the inputted dispersion dictionary from the run_dispersion method
        and return a dictionary with the processed parameters, being ready to be
        used in the dispersion simulation.

        Parameters
        ----------
        dictionary : dict
            Dictionary containing the parameters to be varied in the dispersion
            simulation. The keys of the dictionary are the names of the parameters
            to be varied, and the values can be either tuple or list. If the value
            is a single value, the corresponding class of the parameter need to
            be passed on the run_dispersion method.

        Returns
        -------
        dictionary: dict
            The modified dictionary with the processed parameters.
        """
        # First we need to check if the dictionary is empty
        if not dictionary:
            warnings.warn(
                "The dispersion dictionary is empty, no dispersion will be performed"
            )
            return dictionary

        # Now we prepare all the parachute data
        dictionary = self.__process_parachute_from_dict(dictionary)

        # Check remaining class inputs
        # Environment
        dictionary = self.__process_environment_from_dict(dictionary)

        # Motor
        dictionary = self.__process_motor_from_dict(dictionary)

        # Rocket
        dictionary = self.__process_rocket_from_dict(dictionary)

        # Rail button
        dictionary = self.__process_rail_buttons_from_dict(dictionary)

        # Aerodynamic Surfaces
        dictionary = self.__process_aerodynamic_surfaces_from_dict(dictionary)

        # Flight
        dictionary = self.__process_flight_from_dict(dictionary)

        # Finally check the inputted data
        self.__check_inputted_values_from_dict(dictionary)

        return dictionary

    def __process_flight_from_dict(self, dictionary):
        """Check if all the relevant inputs for the Flight class are present in
        the dispersion dictionary, input the missing ones and return the modified
        dictionary.

        Parameters
        ----------
        dictionary : dict
            Dictionary containing the parameters to be varied in the dispersion
            simulation. The keys of the dictionary are the names of the parameters
            to be varied, and the values can be either tuple or list. If the value
            is a single value, the corresponding class of the parameter need to
            be passed on the run_dispersion method.

        Returns
        -------
        dictionary: dict
            Modified dictionary with the processed flight parameters.
        """
        if not all(
            flight_input in dictionary for flight_input in self.flight_inputs.keys()
        ):
            # Iterate through missing inputs
            for missing_input in set(self.flight_inputs.keys()) - dictionary.keys():
                missing_input = str(missing_input)
                # Add to the dict
                try:
                    # First try to catch value from the Flight object if passed
                    dictionary[missing_input] = [getattr(self.flight, missing_input)]
                except:
                    # class was not inputted
                    # check if missing parameter is required
                    if self.flight_inputs[missing_input] == "required":
                        warnings.warn(f'Missing "{missing_input}" in dictionary')
                    else:  # if not, uses default value
                        dictionary[missing_input] = [self.flight_inputs[missing_input]]

        return dictionary

    def __process_rocket_from_dict(self, dictionary):
        """Check if all the relevant inputs for the Rocket class are present in
        the dispersion dictionary, input the missing ones and return the modified
        dictionary.

        Parameters
        ----------
        dictionary : dict
            Dictionary containing the parameters to be varied in the dispersion
            simulation. The keys of the dictionary are the names of the parameters
            to be varied, and the values can be either tuple or list. If the value
            is a single value, the corresponding class of the parameter need to
            be passed on the run_dispersion method.

        Returns
        -------
        dictionary: dict
            Modified dictionary with the processed rocket parameters.
        """
        if not all(
            rocket_input in dictionary for rocket_input in self.rocket_inputs.keys()
        ):
            # Iterate through missing inputs
            for missing_input in set(self.rocket_inputs.keys()) - dictionary.keys():
                missing_input = str(missing_input)
                # Add to the dict
                try:
                    dictionary[missing_input] = [getattr(self.rocket, missing_input)]
                except:
                    # class was not inputted
                    # checks if missing parameter is required
                    if self.rocket_inputs[missing_input] == "required":
                        warnings.warn(f'Missing "{missing_input}" in dictionary')
                    else:  # if not, uses default value
                        dictionary[missing_input] = [self.rocket_inputs[missing_input]]

        return dictionary

    def __process_motor_from_dict(self, dictionary):
        """Check if all the relevant inputs for the Motor class are present in
        the dispersion dictionary, input the missing ones and return the modified
        dictionary.

        Parameters
        ----------
        dictionary : dict
            Dictionary containing the parameters to be varied in the dispersion
            simulation. The keys of the dictionary are the names of the parameters
            to be varied, and the values can be either tuple or list. If the value
            is a single value, the corresponding class of the parameter need to
            be passed on the run_dispersion method.

        Returns
        -------
        dictionary: dict
            Modified dictionary with the processed rocket parameters.
        """
        # TODO: Add mor options of motor (i.e. Liquid and Hybrids)

        if not all(
            motor_input in dictionary for motor_input in self.solid_motor_inputs.keys()
        ):
            # Iterate through missing inputs
            for missing_input in (
                set(self.solid_motor_inputs.keys()) - dictionary.keys()
            ):
                missing_input = str(missing_input)
                # Add to the dict
                try:
                    dictionary[missing_input] = [getattr(self.motor, missing_input)]
                except:
                    # class was not inputted
                    # checks if missing parameter is required
                    if self.solid_motor_inputs[missing_input] == "required":
                        warnings.warn(f'Missing "{missing_input}" in d')
                    else:  # if not uses default value
                        dictionary[missing_input] = [
                            self.solid_motor_inputs[missing_input]
                        ]
        return dictionary

    def __process_environment_from_dict(self, dictionary):
        """Check if all the relevant inputs for the Environment class are present in
        the dispersion dictionary, input the missing ones and return the modified
        dictionary.

        Parameters
        ----------
        dictionary :  dict
            Dictionary containing the parameters to be varied in the dispersion
            simulation. The keys of the dictionary are the names of the parameters
            to be varied, and the values can be either tuple or list. If the value
            is a single value, the corresponding class of the parameter need to
            be passed on the run_dispersion method.

        Returns
        -------
        dictionary: dict
            Modified dictionary with the processed environment parameters.
        """
        if not all(
            environment_input in dictionary
            for environment_input in self.environment_inputs.keys()
        ):
            # Iterate through missing inputs
            for missing_input in (
                set(self.environment_inputs.keys()) - dictionary.keys()
            ):
                missing_input = str(missing_input)
                # Add to the dict
                try:
                    dictionary[missing_input] = [
                        getattr(self.environment, missing_input)
                    ]
                except:
                    # class was not inputted
                    # checks if missing parameter is required
                    if self.environment_inputs[missing_input] == "required":
                        warnings.warn("Missing {} in dictionary".format(missing_input))
                    else:  # if not, use default value
                        dictionary[missing_input] = [
                            self.environment_inputs[missing_input]
                        ]
        return dictionary

    def __process_parachute_from_dict(self, dictionary):
        """Check if all the relevant inputs for the Parachute class are present in
        the dispersion dictionary, input the missing ones and return the modified
        dictionary.

        Parameters
        ----------
        dictionary : dict
            Dictionary containing the parameters to be varied in the dispersion
            simulation. The keys of the dictionary are the names of the parameters
            to be varied, and the values can be either tuple or list. If the value
            is a single value, the corresponding class of the parameter need to
            be passed on the run_dispersion method.

        Returns
        -------
        dictionary: dict
            Modified dictionary with the processed parachute parameters.
        """
        # Get parachutes names
        if "parachuteNames" in dictionary:  # TODO: use only dictionary
            for i, name in enumerate(dictionary["parachuteNames"]):
                if "CdS" in dictionary:
                    dictionary["parachute_" + name + "_CdS"] = dictionary["CdS"][i]
                if "trigger" in dictionary:
                    dictionary["parachute_" + name + "_trigger"] = dictionary[
                        "trigger"
                    ][i]
                if "samplingRate" in dictionary:
                    dictionary["parachute_" + name + "_samplingRate"] = dictionary[
                        "samplingRate"
                    ][i]
                if "lag" in dictionary:
                    dictionary["parachute_" + name + "_lag"] = dictionary["lag"][i]
                if "noise_mean" in dictionary:
                    dictionary["parachute_" + name + "_noise_mean"] = dictionary[
                        "noise_mean"
                    ][i]
                if "noise_sd" in dictionary:
                    dictionary["parachute_" + name + "_noise_std"] = dictionary[
                        "noise_sd"
                    ][i]
                if "noise_corr" in dictionary:
                    dictionary["parachute_" + name + "_noise_corr"] = dictionary[
                        "noise_corr"
                    ][i]
            # Remove already used keys from dictionary to avoid confusion
            dictionary.pop("CdS", None)
            dictionary.pop("trigger", None)
            dictionary.pop("samplingRate", None)
            dictionary.pop("lag", None)
            dictionary.pop("noise_mean", None)
            dictionary.pop("noise_sd", None)
            dictionary.pop("noise_corr", None)
            self.parachute_names = dictionary.pop("parachuteNames", None)

        return dictionary

    def __check_inputted_values_from_dict(self, dictionary):
        """Check if the inputted values are valid. If not, raise an error.

        Parameters
        ----------
        dictionary : dict
            Dictionary containing the parameters to be varied in the dispersion
            simulation. The keys of the dictionary are the names of the parameters
            to be varied, and the values can be either tuple or list. If the value
            is a single value, the corresponding class of the parameter need to
            be passed on the run_dispersion method.

        Returns
        -------
        dictionary: dict
            The modified dictionary with the processed parameters.
        """
        for parameter_key, parameter_value in dictionary.items():
            if isinstance(parameter_value, (tuple, list)):
                # Everything is right with the data, we have mean and stdev
                continue

            # In this case the parameter_value is only the std. dev.
            ## First solve the parachute values
            if "parachute" in parameter_key:
                _, parachute_name, parameter = parameter_key.split("_")
                dictionary[parameter_key] = (
                    getattr(
                        self.rocket.parachutes[
                            self.parachute_names.index(parachute_name)
                        ],
                        parameter,
                    ),
                    parameter_value,
                )

            ## Second corrections - Environment
            if parameter_key in self.environment_inputs.keys():
                try:
                    dictionary[parameter_key] = (
                        getattr(self.environment, parameter_key),
                        parameter_value,
                    )
                except Exception as E:
                    print("Error:")
                    print(
                        "Please check if the parameter was inputted correctly in dispersion_dictionary."
                        + " Dictionary values must be either tuple or lists."
                        + " If single value, the corresponding Class must "
                        + " be inputted in Dispersion.run_dispersion method.\n"
                    )
                    print(traceback.format_exc())

            ## Third corrections - SolidMotor
            elif parameter_key in self.solid_motor_inputs.keys():
                try:
                    dictionary[parameter_key] = (
                        getattr(self.motor, parameter_key),
                        parameter_value,
                    )
                except Exception as E:
                    print("Error:")
                    print(
                        "Please check if the parameter was inputted correctly in dispersion_dictionary."
                        + " Dictionary values must be either tuple or lists."
                        + " If single value, the corresponding Class must "
                        + "must be inputted in Dispersion.run_dispersion method.\n"
                    )
                    print(traceback.format_exc())

            # Fourth correction - Rocket
            elif parameter_key in self.rocket_inputs.keys():
                try:
                    dictionary[parameter_key] = (
                        getattr(self.rocket, parameter_key),
                        parameter_value,
                    )
                except Exception as E:
                    print("Error:")
                    print(
                        "Please check if the parameter was inputted correctly in dispersion_dictionary."
                        + " Dictionary values must be either tuple or lists."
                        + " If single value, the corresponding Class must "
                        + "must be inputted in Dispersion.run_dispersion method.\n"
                    )
                    print(traceback.format_exc())

            # Fifth correction - Flight
            elif parameter_key in self.flight_inputs.keys():
                try:
                    dictionary[parameter_key] = (
                        getattr(self.flight, parameter_key),
                        parameter_value,
                    )
                except Exception as E:
                    print("Error:")
                    print(
                        "Please check if the parameter was inputted correctly in dispersion_dictionary."
                        + " Dictionary values must be either tuple or lists."
                        + " If single value, the corresponding Class must "
                        + "must be inputted in Dispersion.run_dispersion method.\n"
                    )
                    print(traceback.format_exc())

        # The analysis parameter dictionary must be corrected now!

        return dictionary

    def __yield_flight_setting(
        self, distribution_func, analysis_parameters, number_of_simulations
    ):
        """Yields a flight setting for the simulation

        Parameters
        ----------
        distribution_func : _type_
            _description_
        analysis_parameters : dict
            _description_
        number_of_simulations : int
            Number of simulations desired, must be non negative.
            This is needed when running a new simulation. Default is zero.

        Yields
        ------
        flight_setting

        """

        i = 0
        while i < number_of_simulations:
            # Generate a flight setting
            flight_setting = {}
            for parameter_key, parameter_value in analysis_parameters.items():
                if type(parameter_value) is tuple:
                    flight_setting[parameter_key] = distribution_func(*parameter_value)
                elif isinstance(parameter_value, Function):
                    flight_setting[parameter_key] = distribution_func(*parameter_value)
                else:
                    # shuffles list and gets first item
                    shuffle(parameter_value)
                    flight_setting[parameter_key] = parameter_value[0]

            # Update counter
            i += 1
            # Yield a flight setting
            yield flight_setting

    # TODO: allow user to chose what is going to be exported
    def __export_flight_data(
        self,
        flight_setting,
        flight,
        exec_time,
        dispersion_input_file,
        dispersion_output_file,
    ):
        """Saves flight results in a .txt

        Parameters
        ----------
        flight_setting : _type_
            _description_
        flight : Flight
            _description_
        exec_time : _type_
            _description_
        dispersion_input_file : _type_
            _description_
        dispersion_output_file : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """

        # Generate flight results
        flight_result = {
            "outOfRailTime": flight.outOfRailTime,
            "outOfRailVelocity": flight.outOfRailVelocity,
            "apogeeTime": flight.apogeeTime,
            "apogeeAltitude": flight.apogee - flight.env.elevation,
            "apogeeX": flight.apogeeX,
            "apogeeY": flight.apogeeY,
            "impactTime": flight.tFinal,
            "impactX": flight.xImpact,
            "impactY": flight.yImpact,
            "impactVelocity": flight.impactVelocity,
            "initialStaticMargin": flight.rocket.staticMargin(0),
            "outOfRailStaticMargin": flight.rocket.staticMargin(flight.outOfRailTime),
            "finalStaticMargin": flight.rocket.staticMargin(
                flight.rocket.motor.burnOutTime
            ),
            "numberOfEvents": len(flight.parachuteEvents),
            "drogueTriggerTime": [],
            "drogueInflatedTime": [],
            "drogueInflatedVelocity": [],
            "executionTime": exec_time,
            "lateralWind": flight.lateralSurfaceWind,
            "frontalWind": flight.frontalSurfaceWind,
        }

        # # Calculate maximum reached velocity
        # sol = np.array(flight.solution)
        # flight.vx = Function(
        #     sol[:, [0, 4]],
        #     "Time (s)",
        #     "Vx (m/s)",
        #     "linear",
        #     extrapolation="natural",
        # )
        # flight.vy = Function(
        #     sol[:, [0, 5]],
        #     "Time (s)",
        #     "Vy (m/s)",
        #     "linear",
        #     extrapolation="natural",
        # )
        # flight.vz = Function(
        #     sol[:, [0, 6]],
        #     "Time (s)",
        #     "Vz (m/s)",
        #     "linear",
        #     extrapolation="natural",
        # )
        # flight.speed = (flight.vx**2 + flight.vy**2 + flight.vz**2) ** 0.5
        # flight.maxVel = np.amax(flight.speed.source[:, 1])
        # flight_result["maxVelocity"] = flight.maxVel
        flight_result["maxVelocity"] = flight.maxSpeed

        # Take care of parachute results
        for trigger_time, parachute in flight.parachuteEvents:
            flight_result[parachute.name + "_triggerTime"] = trigger_time
            flight_result[parachute.name + "_inflatedTime"] = (
                trigger_time + parachute.lag
            )
            flight_result[parachute.name + "_inflatedVelocity"] = flight.speed(
                trigger_time + parachute.lag
            )
        else:
            flight_result["parachuteInfo"] = "No Parachute Events"

        # Write flight setting and results to file
        flight_setting.pop("thrust", None)
        dispersion_input_file.write(str(flight_setting) + "\n")
        dispersion_output_file.write(str(flight_result) + "\n")

        return None

    def __export_flight_data(self, flight_setting, dispersion_error_file):

        """Saves flight error in a .txt"""

        dispersion_error_file.write(str(flight_setting) + "\n")

        return None

    def run_dispersion(
        self,
        number_of_simulations,
        dispersion_dictionary,
        environment=None,
        flight=None,
        motor=None,
        rocket=None,
        bg_image=None,
        actual_landing_point=None,
    ):
        """Runs the given number of simulations and saves the data

        Parameters
        ----------
        number_of_simulations : int
            Number of simulations desired, must be non negative.
            This is needed when running a new simulation. Default is zero.
        dispersion_dictionary : _type_
            _description_
        environment : _type_
            _description_
        flight : Flight, optional
            Original rocket's flight with nominal values. Parameter needed to run
            a new flight simulation when environment, motor and rocket remain
            unchanged. By default None.
        motor : _type_, optional
            _description_, by default None
        rocket : _type_, optional
            _description_, by default None
        distribution_type : str, optional
            _description_, by default "normal"
        bg_image : str, optional
            The path to the image to be used as the background
        actual_landing_point : tuple, optional
            A tuple containing the actual landing point of the rocket, if known.
            Useful when comparing the dispersion results with the actual landing.
            Must be given in tuple format, such as (lat, lon). By default None.
            # TODO: Check the order of these coordinates

        Returns
        -------
        None
        """

        self.number_of_simulations = number_of_simulations
        self.dispersion_dictionary = dispersion_dictionary
        self.environment = None
        self.motor = None
        self.rocket = None
        if flight:  # In case a flight object is passed
            self.environment = flight.env
            self.motor = flight.rocket.motor
            self.rocket = flight.rocket
        self.environment = environment if environment else self.environment
        self.motor = motor if motor else self.motor
        self.rocket = rocket if rocket else self.rocket
        self.flight = flight
        self.distribution_type = "normal"  # TODO: Must be parametrized
        self.image = bg_image
        self.actual_landing_point = actual_landing_point  # (lat, lon)

        # Obs.: The flight object is not prioritized, which is a good thing, but need to be documented

        # Check if there's enough object to start a flight:
        ## Raise an error in case of any troubles
        self.__check_initial_objects()

        # Creates copy of dispersion_dictionary that will be altered
        modified_dispersion_dict = {i: j for i, j in dispersion_dictionary.items()}

        analysis_parameters = self.__process_dispersion_dict(modified_dispersion_dict)

        # TODO: This should be more flexible, allow different distributions for different parameters
        self.distributionFunc = self.__set_distribution_function(self.distribution_type)

        # Create data files for inputs, outputs and error logging
        dispersion_error_file = open(str(self.filename) + ".disp_errors.txt", "w")
        dispersion_input_file = open(str(self.filename) + ".disp_inputs.txt", "w")
        dispersion_output_file = open(str(self.filename) + ".disp_outputs.txt", "w")

        # Initialize counter and timer
        i = 0

        initial_wall_time = time()
        initial_cpu_time = process_time()

        # Iterate over flight settings, start the flight simulations
        out = display("Starting", display_id=True)
        for setting in self.__yield_flight_setting(
            self.distributionFunc, analysis_parameters, self.number_of_simulations
        ):
            self.start_time = process_time()
            i += 1

            # Creates a copy of the environment
            env_dispersion = self.environment

            # Apply environment parameters variations on each iteration if possible
            env_dispersion.railLength = setting["railLength"]
            env_dispersion.gravity = setting["gravity"]
            env_dispersion.date = setting["date"]
            env_dispersion.latitude = setting["latitude"]
            env_dispersion.longitude = setting["longitude"]
            env_dispersion.elevation = setting["elevation"]
            if env_dispersion.atmosphericModelType in ["Ensemble", "Reanalysis"]:
                env_dispersion.selectEnsembleMember(setting["ensembleMember"])

            # Creates copy of motor
            motor_dispersion = self.motor

            # Apply motor parameters variations on each iteration if possible
            # TODO: add hybrid motor option
            motor_dispersion = SolidMotor(
                thrustSource=setting["thrust"],
                burnOut=setting["burnOutTime"],
                grainNumber=setting["grainNumber"],
                grainDensity=setting["grainDensity"],
                grainOuterRadius=setting["grainOuterRadius"],
                grainInitialInnerRadius=setting["grainInitialInnerRadius"],
                grainInitialHeight=setting["grainInitialHeight"],
                grainSeparation=setting["grainSeparation"],
                nozzleRadius=setting["nozzleRadius"],
                throatRadius=setting["throatRadius"],
                reshapeThrustCurve=(setting["burnOutTime"], setting["totalImpulse"]),
            )

            # Creates copy of rocket
            rocket_dispersion = self.rocket

            # Apply rocket parameters variations on each iteration if possible
            rocket_dispersion = Rocket(
                motor=motor_dispersion,
                mass=setting["mass"],
                inertiaI=setting["inertiaI"],
                inertiaZ=setting["inertiaZ"],
                radius=setting["radius"],
                distanceRocketNozzle=setting["distanceRocketNozzle"],
                distanceRocketPropellant=setting["distanceRocketPropellant"],
                powerOffDrag=setting["powerOffDrag"],
                powerOnDrag=setting["powerOnDrag"],
            )

            # Add rocket nose, fins and tail
            rocket_dispersion.addNose(
                length=setting["noseLength"],
                kind=setting["noseKind"],
                distanceToCM=setting["noseDistanceToCM"],
            )
            rocket_dispersion.addFins(
                n=setting["numberOfFins"],
                rootChord=setting["finRootChord"],
                tipChord=setting["finTipChord"],
                span=setting["finSpan"],
                distanceToCM=setting["finDistanceToCM"],
                radius=setting["radius"],
                airfoil=setting["airfoil"],
            )
            if not "noTail" in setting:
                rocket_dispersion.addTail(
                    topRadius=setting["topRadius"],
                    bottomRadius=setting["bottomRadius"],
                    length=setting["length"],
                    distanceToCM=setting["distanceToCM"],
                )

            # Add parachutes
            for num, name in enumerate(self.parachute_names):
                rocket_dispersion.addParachute(
                    name=name,
                    CdS=setting["parachute_" + name + "_CdS"],
                    trigger=setting["parachute_" + name + "_trigger"],
                    samplingRate=setting["parachute_" + name + "_samplingRate"],
                    lag=setting["parachute_" + name + "_lag"],
                    noise=(
                        setting["parachute_" + name + "_noise_mean"],
                        setting["parachute_" + name + "_noise_std"],
                        setting["parachute_" + name + "_noise_corr"],
                    ),
                )

            rocket_dispersion.setRailButtons(
                distanceToCM=[
                    setting["positionFirstRailButton"],
                    setting["positionSecondRailButton"],
                ],
                angularPosition=setting["railButtonAngularPosition"],
            )

            # Run trajectory simulation
            try:
                # TODO: Add initialSolution flight option
                TestFlight = Flight(
                    rocket=rocket_dispersion,
                    environment=env_dispersion,
                    inclination=setting["inclination"],
                    heading=setting["heading"],
                    terminateOnApogee=setting["terminateOnApogee"],
                    maxTime=setting["maxTime"],
                    maxTimeStep=setting["maxTimeStep"],
                    minTimeStep=setting["minTimeStep"],
                    rtol=setting["rtol"],
                    atol=setting["atol"],
                    timeOvershoot=setting["timeOvershoot"],
                    verbose=setting["verbose"],
                )

                self.__export_flight_data(
                    flight_setting=setting,
                    flight_data=TestFlight,
                    exec_time=process_time() - self.start_time,
                    dispersion_input_file=dispersion_input_file,
                    dispersion_output_file=dispersion_output_file,
                )
            except Exception as E:
                print(E)
                print(traceback.format_exc())
                self.__export_flight_data(setting, dispersion_error_file)

            # Register time
            out.update(
                f"Current iteration: {i:06d} | Average Time per Iteration: {(process_time() - initial_cpu_time)/i:2.6f} s | Estimated time left: {int((number_of_simulations - i)*((process_time() - initial_cpu_time)/i))} s"
            )

        # Clean the house once all the simulations were already done

        ## Print and save total time
        final_string = f"Completed {i} iterations successfully. Total CPU time: {process_time() - initial_cpu_time} s. Total wall time: {time() - initial_wall_time} s"
        out.update(final_string)
        dispersion_input_file.write(final_string + "\n")
        dispersion_output_file.write(final_string + "\n")
        dispersion_error_file.write(final_string + "\n")

        ## Close files
        dispersion_input_file.close()
        dispersion_output_file.close()
        dispersion_error_file.close()

        return None

    def __check_initial_objects(self):
        """Create rocketpy objects (Environment, Motor, Rocket, Flight) in case
        that

        Returns
        -------
        _type_
            _description_
        """
        if self.environment is None:
            self.environment = Environment(
                railLength=self.dispersion_dictionary["railLength"][0]
            )
        if self.motor is None:
            self.motor = SolidMotor(
                thrustSource=self.dispersion_dictionary["thrustSource"][0],
                burnOut=self.dispersion_dictionary["burnOutTime"][0],
                grainNumber=self.dispersion_dictionary["grainNumber"][0],
                grainDensity=self.dispersion_dictionary["grainDensity"][0],
                grainOuterRadius=self.dispersion_dictionary["grainOuterRadius"][0],
                grainInitialInnerRadius=self.dispersion_dictionary[
                    "grainInitialInnerRadius"
                ][0],
                grainInitialHeight=self.dispersion_dictionary["grainInitialHeight"][0],
            )
        if self.rocket is None:
            self.rocket = Rocket(
                motor=self.motor,
                mass=self.dispersion_dictionary["mass"][0],
                radius=self.dispersion_dictionary["radius"][0],
                inertiaI=self.dispersion_dictionary["inertiaI"][
                    0
                ],  # TODO: remove hardcode
                inertiaZ=self.dispersion_dictionary["inertiaZ"][
                    0
                ],  # TODO: remove hardcode
                distanceRocketPropellant=self.dispersion_dictionary[
                    "distanceRocketPropellant"
                ][0],
                distanceRocketNozzle=self.dispersion_dictionary["distanceRocketNozzle"][
                    0
                ],
                powerOffDrag=0.6,  # TODO: Remove this hardcoded
                powerOnDrag=0.6,  # TODO: Remove this hardcoded
            )
            self.rocket.setRailButtons(distanceToCM=[0.2, -0.5])
        if self.flight is None:
            self.flight = Flight(
                rocket=self.rocket,
                environment=self.environment,
                inclination=self.dispersion_dictionary["inclination"][0],
                heading=self.dispersion_dictionary["heading"][0],
            )
        return None

    def import_results(self, dispersion_output_file):
        """Import dispersion results from .txt file

        Parameters
        ----------
        dispersion_output_file : str
            Path to the dispersion output file. This file will not be overwritten,
            modified or deleted by this function.

        Returns
        -------
        None
        """
        # Initialize variable to store all results
        dispersion_general_results = []

        # TODO: Add more flexible way to define dispersion_results
        dispersion_results = {
            "outOfRailTime": [],
            "outOfRailVelocity": [],
            "apogeeTime": [],
            "apogeeAltitude": [],
            "apogeeX": [],
            "apogeeY": [],
            "impactTime": [],
            "impactX": [],
            "impactY": [],
            "impactVelocity": [],
            "initialStaticMargin": [],
            "outOfRailStaticMargin": [],
            "finalStaticMargin": [],
            "numberOfEvents": [],
            "maxVelocity": [],
            "drogueTriggerTime": [],
            "drogueInflatedTime": [],
            "drogueInflatedVelocity": [],
            "executionTime": [],
            "railDepartureAngleOfAttack": [],
            "lateralWind": [],
            "frontalWind": [],
        }

        # Get all dispersion results
        # Open the file
        file = open(dispersion_output_file, "r+")

        # Read each line of the file and convert to dict
        for line in file:
            # Skip comments lines
            if line[0] != "{":
                continue
            # Evaluate results and store them
            flight_result = eval(line)
            dispersion_general_results.append(flight_result)
            for parameter_key, parameter_value in flight_result.items():
                dispersion_results[parameter_key].append(parameter_value)

        # Close data file
        file.close()

        # Calculate the number of flights simulated
        self.num_of_loaded_sims = len(dispersion_general_results)

        # Print the number of flights simulated
        print(
            f"A total of {self.num_of_loaded_sims} simulations were loaded from the following file: {dispersion_output_file}"
        )

        # Save the results as an attribute of the class
        self.dispersion_results = dispersion_results

        return None

    # Start the processing analysis

    def outOfRailTime(self):
        """Calculate the time of the rocket's departure from the rail, in seconds.

        Returns
        -------
        _type_
            _description_
        """
        self.mean_out_of_rail_time = (
            np.mean(self.dispersion_results["outOfRailTime"])
            if self.dispersion_results["outOfRailTime"]
            else None
        )
        self.std_out_of_rail_time = (
            np.std(self.dispersion_results["outOfRailTime"])
            if self.dispersion_results["outOfRailTime"]
            else None
        )
        return None

    def printMeanOutOfRailTime(self):
        """Prints out the mean and std. dev. of the "outOfRailTime" parameter.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.outOfRailTime()
        print(f"Out of Rail Time -Mean Value: {self.mean_out_of_rail_time:0.3f} s")
        print(f"Out of Rail Time - Std. Dev.: {self.std_out_of_rail_time:0.3f} s")

        return None

    def plotOutOfRailTime(self):
        """Plot the out of rail time distribution

        Parameters
        ----------
        dispersion_results : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        self.outOfRailTime()

        plt.figure()
        plt.hist(
            self.dispersion_results["outOfRailTime"],
            bins=int(self.num_of_loaded_sims**0.5),
        )
        plt.title("Out of Rail Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Number of Occurrences")
        plt.show()

        return None

    def meanOutOfRailVelocity(self, dispersion_results):
        """_summary_

        Parameters
        ----------
        dispersion_results : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        print(
            f'Out of Rail Velocity -Mean Value: {np.mean(dispersion_results["outOfRailVelocity"]):0.3f} m/s'
        )
        print(
            f'Out of Rail Velocity - Std. Dev.: {np.std(dispersion_results["outOfRailVelocity"]):0.3f} m/s'
        )

        return None

    def plotOutOfRailVelocity(self, dispersion_results):
        """_summary_

        Parameters
        ----------
        dispersion_results : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        self.meanOutOfRailVelocity(dispersion_results)

        plt.figure()
        plt.hist(
            dispersion_results["outOfRailVelocity"],
            bins=int(self.num_of_loaded_sims**0.5),
        )
        plt.title("Out of Rail Velocity")
        plt.xlabel("Velocity (m/s)")
        plt.ylabel("Number of Occurrences")
        plt.show()

        return None

    def meanApogeeTime(self, dispersion_results):
        """_summary_

        Parameters
        ----------
        dispersion_results : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        print(
            f'Impact Time -Mean Value: {np.mean(dispersion_results["impactTime"]):0.3f} s'
        )
        print(
            f'Impact Time - Std. Dev.: {np.std(dispersion_results["impactTime"]):0.3f} s'
        )

        return None

    def plotApogeeTime(self, dispersion_results):
        """_summary_

        Parameters
        ----------
        dispersion_results : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        self.meanApogeeTime(dispersion_results)

        plt.figure()
        plt.hist(
            dispersion_results["impactTime"],
            bins=int(self.num_of_loaded_sims**0.5),
        )
        plt.title("Impact Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Number of Occurrences")
        plt.show()

        return None

    def meanApogeeAltitude(self, dispersion_results):
        """_summary_

        Parameters
        ----------
        dispersion_results : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        print(
            f'Apogee Altitude -Mean Value: {np.mean(dispersion_results["apogeeAltitude"]):0.3f} m'
        )
        print(
            f'Apogee Altitude - Std. Dev.: {np.std(dispersion_results["apogeeAltitude"]):0.3f} m'
        )

        return None

    def plotApogeeAltitude(self, dispersion_results):
        """_summary_

        Parameters
        ----------
        dispersion_results : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        self.meanApogeeAltitude(dispersion_results)

        plt.figure()
        plt.hist(
            dispersion_results["apogeeAltitude"],
            bins=int(self.num_of_loaded_sims**0.5),
        )
        plt.title("Apogee Altitude")
        plt.xlabel("Altitude (m)")
        plt.ylabel("Number of Occurrences")
        plt.show()

        return None

    def meanApogeeXPosition(self, dispersion_results):
        """_summary_

        Parameters
        ----------
        dispersion_results : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        print(
            f'Apogee X Position -Mean Value: {np.mean(dispersion_results["apogeeX"]):0.3f} m'
        )
        print(
            f'Apogee X Position - Std. Dev.: {np.std(dispersion_results["apogeeX"]):0.3f} m'
        )

        return None

    def plotApogeeXPosition(self, dispersion_results):
        """_summary_

        Parameters
        ----------
        dispersion_results : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        self.meanApogeeAltitude(dispersion_results)

        plt.figure()
        plt.hist(
            dispersion_results["apogeeX"],
            bins=int(self.num_of_loaded_sims**0.5),
        )
        plt.title("Apogee X Position")
        plt.xlabel("Apogee X Position (m)")
        plt.ylabel("Number of Occurrences")
        plt.show()

        return None

    def meanApogeeYPosition(self, dispersion_results):
        """_summary_

        Parameters
        ----------
        dispersion_results : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        print(
            f'Apogee Y Position -Mean Value: {np.mean(dispersion_results["apogeeY"]):0.3f} m'
        )
        print(
            f'Apogee Y Position - Std. Dev.: {np.std(dispersion_results["apogeeY"]):0.3f} m'
        )

        return None

    def plotApogeeYPosition(self, dispersion_results):
        """_summary_

        Parameters
        ----------
        dispersion_results : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        self.meanApogeeAltitude(dispersion_results)

        plt.figure()
        plt.hist(
            dispersion_results["apogeeY"],
            bins=int(self.num_of_loaded_sims**0.5),
        )
        plt.title("Apogee Y Position")
        plt.xlabel("Apogee Y Position (m)")
        plt.ylabel("Number of Occurrences")
        plt.show()

        return None

    def meanImpactTime(self, dispersion_results):
        """_summary_

        Parameters
        ----------
        dispersion_results : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        print(
            f'Impact Time -Mean Value: {np.mean(dispersion_results["impactTime"]):0.3f} s'
        )
        print(
            f'Impact Time - Std. Dev.: {np.std(dispersion_results["impactTime"]):0.3f} s'
        )

        return None

    def plotImpactTime(self, dispersion_results):
        """_summary_

        Parameters
        ----------
        dispersion_results : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        self.meanImpactTime(dispersion_results)

        plt.figure()
        plt.hist(
            dispersion_results["impactTime"],
            bins=int(self.num_of_loaded_sims**0.5),
        )
        plt.title("Impact Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Number of Occurrences")
        plt.show()

        return None

    def meanImpactXPosition(self, dispersion_results):
        """_summary_

        Parameters
        ----------
        dispersion_results : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        print(
            f'Impact X Position -Mean Value: {np.mean(dispersion_results["impactX"]):0.3f} m'
        )
        print(
            f'Impact X Position - Std. Dev.: {np.std(dispersion_results["impactX"]):0.3f} m'
        )

        return None

    def plotImpactXPosition(self, dispersion_results):
        """_summary_

        Parameters
        ----------
        dispersion_results : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        self.meanImpactXPosition(dispersion_results)

        plt.figure()
        plt.hist(
            dispersion_results["impactX"],
            bins=int(self.num_of_loaded_sims**0.5),
        )
        plt.title("Impact X Position")
        plt.xlabel("Impact X Position (m)")
        plt.ylabel("Number of Occurrences")
        plt.show()

        return None

    def meanImpactYPosition(self, dispersion_results):
        """_summary_

        Parameters
        ----------
        dispersion_results : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        print(
            f'Impact Y Position -Mean Value: {np.mean(dispersion_results["impactY"]):0.3f} m'
        )
        print(
            f'Impact Y Position - Std. Dev.: {np.std(dispersion_results["impactY"]):0.3f} m'
        )

        return None

    def plotImpactYPosition(self, dispersion_results):
        """_summary_

        Parameters
        ----------
        dispersion_results : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        self.meanImpactYPosition(dispersion_results)

        plt.figure()
        plt.hist(
            dispersion_results["impactY"],
            bins=int(self.num_of_loaded_sims**0.5),
        )
        plt.title("Impact Y Position")
        plt.xlabel("Impact Y Position (m)")
        plt.ylabel("Number of Occurrences")
        plt.show()

        return None

    def meanImpactVelocity(self, dispersion_results):
        """_summary_

        Parameters
        ----------
        dispersion_results : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        print(
            f'Impact Velocity -Mean Value: {np.mean(dispersion_results["impactVelocity"]):0.3f} m/s'
        )
        print(
            f'Impact Velocity - Std. Dev.: {np.std(dispersion_results["impactVelocity"]):0.3f} m/s'
        )

        return None

    def plotImpactVelocity(self, dispersion_results):
        """_summary_

        Parameters
        ----------
        dispersion_results : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        self.meanImpactVelocity(dispersion_results)

        plt.figure()
        plt.hist(
            dispersion_results["impactVelocity"],
            bins=int(self.num_of_loaded_sims**0.5),
        )
        plt.title("Impact Velocity")
        plt.xlim(-35, 0)
        plt.xlabel("Velocity (m/s)")
        plt.ylabel("Number of Occurrences")
        plt.show()

        return None

    def meanStaticMargin(self, dispersion_results):
        """_summary_

        Parameters
        ----------
        dispersion_results : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        print(
            f'Initial Static Margin -    Mean Value: {np.mean(dispersion_results["initialStaticMargin"]):0.3f} c'
        )
        print(
            f'Initial Static Margin -     Std. Dev.: {np.std(dispersion_results["initialStaticMargin"]):0.3f} c'
        )

        print(
            f'Out of Rail Static Margin -Mean Value: {np.mean(dispersion_results["outOfRailStaticMargin"]):0.3f} c'
        )
        print(
            f'Out of Rail Static Margin - Std. Dev.: {np.std(dispersion_results["outOfRailStaticMargin"]):0.3f} c'
        )

        print(
            f'Final Static Margin -      Mean Value: {np.mean(dispersion_results["finalStaticMargin"]):0.3f} c'
        )
        print(
            f'Final Static Margin -       Std. Dev.: {np.std(dispersion_results["finalStaticMargin"]):0.3f} c'
        )

        return None

    def plotStaticMargin(self, dispersion_results):
        """_summary_

        Parameters
        ----------
        dispersion_results : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        self.meanStaticMargin(dispersion_results)

        plt.figure()
        plt.hist(
            dispersion_results["initialStaticMargin"],
            label="Initial",
            bins=int(self.num_of_loaded_sims**0.5),
        )
        plt.hist(
            dispersion_results["outOfRailStaticMargin"],
            label="Out of Rail",
            bins=int(self.num_of_loaded_sims**0.5),
        )
        plt.hist(
            dispersion_results["finalStaticMargin"],
            label="Final",
            bins=int(self.num_of_loaded_sims**0.5),
        )
        plt.legend()
        plt.title("Static Margin")
        plt.xlabel("Static Margin (c)")
        plt.ylabel("Number of Occurrences")
        plt.show()

        return None

    def meanMaximumVelocity(self, dispersion_results):
        """_summary_

        Parameters
        ----------
        dispersion_results : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        print(
            f'Maximum Velocity -Mean Value: {np.mean(dispersion_results["maxVelocity"]):0.3f} m/s'
        )
        print(
            f'Maximum Velocity - Std. Dev.: {np.std(dispersion_results["maxVelocity"]):0.3f} m/s'
        )

        return None

    def plotMaximumVelocity(self, dispersion_results):
        """_summary_

        Parameters
        ----------
        dispersion_results : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        self.meanMaximumVelocity(dispersion_results)

        plt.figure()
        plt.hist(
            dispersion_results["maxVelocity"],
            bins=int(self.num_of_loaded_sims**0.5),
        )
        plt.title("Maximum Velocity")
        plt.xlabel("Velocity (m/s)")
        plt.ylabel("Number of Occurrences")
        plt.show()

        return None

    def meanNumberOfParachuteEvents(self, dispersion_results):
        """_summary_

        Parameters
        ----------
        dispersion_results : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        print(
            f'Number of Parachute Events -Mean Value: {np.mean(dispersion_results["numberOfEvents"]):0.3f} s'
        )
        print(
            f'Number of Parachute Events - Std. Dev.: {np.std(dispersion_results["numberOfEvents"]):0.3f} s'
        )

        return None

    def plotNumberOfParachuteEvents(self, dispersion_results):
        """_summary_

        Parameters
        ----------
        dispersion_results : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        self.meanNumberOfParachuteEvents(dispersion_results)

        plt.figure()
        plt.hist(dispersion_results["numberOfEvents"])
        plt.title("Parachute Events")
        plt.xlabel("Number of Parachute Events")
        plt.ylabel("Number of Occurrences")
        plt.show()

        return None

    def meanDrogueTriggerTime(self, dispersion_results):
        """_summary_

        Parameters
        ----------
        dispersion_results : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        print(
            f'Drogue Trigger Time -Mean Value: {np.mean(dispersion_results["drogueTriggerTime"]):0.3f} s'
        )
        print(
            f'Drogue Trigger Time - Std. Dev.: {np.std(dispersion_results["drogueTriggerTime"]):0.3f} s'
        )

        return None

    def plotDrogueTriggerTime(self, dispersion_results):
        """_summary_

        Parameters
        ----------
        dispersion_results : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        self.meanDrogueTriggerTime(dispersion_results)

        plt.figure()
        plt.hist(
            dispersion_results["drogueTriggerTime"],
            bins=int(self.num_of_loaded_sims**0.5),
        )
        plt.title("Drogue Trigger Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Number of Occurrences")
        plt.show()

        return None

    def meanDrogueFullyInflatedTime(self, dispersion_results):
        """_summary_

        Parameters
        ----------
        dispersion_results : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        print(
            f'Drogue Fully Inflated Time -Mean Value: {np.mean(dispersion_results["drogueInflatedTime"]):0.3f} s'
        )
        print(
            f'Drogue Fully Inflated Time - Std. Dev.: {np.std(dispersion_results["drogueInflatedTime"]):0.3f} s'
        )

        return None

    def plotDrogueFullyInflatedTime(self, dispersion_results):
        """_summary_

        Parameters
        ----------
        dispersion_results : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        self.meanDrogueFullyInflatedTime(dispersion_results)

        plt.figure()
        plt.hist(
            dispersion_results["drogueInflatedTime"],
            bins=int(self.num_of_loaded_sims**0.5),
        )
        plt.title("Drogue Fully Inflated Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Number of Occurrences")
        plt.show()

        return None

    def meanDrogueFullyVelocity(self, dispersion_results):
        """_summary_

        Parameters
        ----------
        dispersion_results : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        print(
            f'Drogue Parachute Fully Inflated Velocity -Mean Value: {np.mean(dispersion_results["drogueInflatedVelocity"]):0.3f} m/s'
        )
        print(
            f'Drogue Parachute Fully Inflated Velocity - Std. Dev.: {np.std(dispersion_results["drogueInflatedVelocity"]):0.3f} m/s'
        )

        return None

    def plotDrogueFullyVelocity(self, dispersion_results):
        """_summary_

        Parameters
        ----------
        dispersion_results : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        self.meanDrogueFullyVelocity(dispersion_results)

        plt.figure()
        plt.hist(
            dispersion_results["drogueInflatedVelocity"],
            bins=int(self.num_of_loaded_sims**0.5),
        )
        plt.title("Drogue Parachute Fully Inflated Velocity")
        plt.xlabel("Velocity m/s)")
        plt.ylabel("Number of Occurrences")
        plt.show()

        return None

    def createEllipses(self, dispersion_results):
        """_summary_

        Parameters
        ----------
        dispersion_results : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        """A function to create apogee and impact ellipses from the dispersion
        results.

        Parameters
        ----------
        dispersion_results : dict
            A dictionary containing the results of the dispersion analysis.
        """

        # Retrieve dispersion data por apogee and impact XY position
        apogeeX = np.array(dispersion_results["apogeeX"])
        apogeeY = np.array(dispersion_results["apogeeY"])
        impactX = np.array(dispersion_results["impactX"])
        impactY = np.array(dispersion_results["impactY"])

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
        return impact_ellipses, apogee_ellipses

    def plotEllipses(
        self,
        dispersion_results,
        image=None,
        actual_landing_point=None,
        perimeterSize=3000,
        xlim=(-3000, 3000),
        ylim=(-3000, 3000),
    ):
        """A function to plot the error ellipses for the apogee and impact
        points of the rocket. The function also plots the real landing point, if
        given

        Parameters
        ----------
        dispersion_results : dict
            A dictionary containing the results of the dispersion analysis
        image : str, optional
            The path to the image to be used as the background
        actual_landing_point : tuple, optional
            A tuple containing the actual landing point of the rocket, if known.
            Useful when comparing the dispersion results with the actual landing.
            Must be given in tuple format, such as (lat, lon). By default None. # TODO: Check the order
        """
        # Import background map
        if image is not None:
            img = imread(image)

        # Retrieve dispersion data por apogee and impact XY position
        apogeeX = np.array(dispersion_results["apogeeX"])
        apogeeY = np.array(dispersion_results["apogeeY"])
        impactX = np.array(dispersion_results["impactX"])
        impactY = np.array(dispersion_results["impactY"])

        impact_ellipses, apogee_ellipses = self.createEllipses(dispersion_results)

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
        plt.savefig(str(self.filename) + ".pdf", bbox_inches="tight", pad_inches=0)
        plt.savefig(str(self.filename) + ".svg", bbox_inches="tight", pad_inches=0)
        plt.show()
        return None

    def prepareEllipses(self, ellipses, origin_lat, origin_lon, resolution=100):
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
        dispersion_results,
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
        dispersion_results : dict
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
        """

        impact_ellipses, apogee_ellipses = self.createEllipses(dispersion_results)
        outputs = []

        if type == "all" or type == "impact":
            outputs = outputs + self.prepareEllipses(
                impact_ellipses, origin_lat, origin_lon, resolution=resolution
            )

        if type == "all" or type == "apogee":
            outputs = outputs + self.prepareEllipses(
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
            # mult_ell.innerboundaryis = kml_data
            # mult_ell.outerboundaryis = kml_data
            mult_ell.style.linestyle.color = color
            mult_ell.style.linestyle.width = 3
            mult_ell.style.polystyle.color = simplekml.Color.changealphaint(
                100, simplekml.Color.blue
            )

        kml.save(filename)
        return None

    def meanLateralWindSpeed(self, dispersion_results):
        """_summary_

        Parameters
        ----------
        dispersion_results : _type_
            _description_
        """
        print(
            f'Lateral Surface Wind Speed -Mean Value: {np.mean(dispersion_results["lateralWind"]):0.3f} m/s'
        )
        print(
            f'Lateral Surface Wind Speed - Std. Dev.: {np.std(dispersion_results["lateralWind"]):0.3f} m/s'
        )

    def plotLateralWindSpeed(self, dispersion_results):
        """_summary_

        Parameters
        ----------
        dispersion_results : _type_
            _description_
        """
        self.meanLateralWindSpeed(dispersion_results)

        plt.figure()
        plt.hist(
            dispersion_results["lateralWind"],
            bins=int(self.num_of_loaded_sims**0.5),
        )
        plt.title("Lateral Surface Wind Speed")
        plt.xlabel("Velocity (m/s)")
        plt.ylabel("Number of Occurrences")
        plt.show()

    def meanFrontalWindSpeed(self, dispersion_results):
        """_summary_

        Parameters
        ----------
        dispersion_results : _type_
            _description_
        """
        print(
            f'Frontal Surface Wind Speed -Mean Value: {np.mean(dispersion_results["frontalWind"]):0.3f} m/s'
        )
        print(
            f'Frontal Surface Wind Speed - Std. Dev.: {np.std(dispersion_results["frontalWind"]):0.3f} m/s'
        )

    def plotFrontalWindSpeed(self, dispersion_results):
        """_summary_

        Parameters
        ----------
        dispersion_results : _type_
            _description_
        """
        self.meanFrontalWindSpeed(dispersion_results)

        plt.figure()
        plt.hist(
            dispersion_results["frontalWind"],
            bins=int(self.num_of_loaded_sims**0.5),
        )
        plt.title("Frontal Surface Wind Speed")
        plt.xlabel("Velocity (m/s)")
        plt.ylabel("Number of Occurrences")
        plt.show()

    def info(self):
        """_summary_

        Returns
        -------
        None
        """

        dispersion_results = self.dispersion_results

        self.meanApogeeAltitude(dispersion_results)

        self.meanOutOfRailVelocity(dispersion_results)

        self.meanStaticMargin(dispersion_results)

        self.meanLateralWindSpeed(dispersion_results)

        self.meanFrontalWindSpeed(dispersion_results)

        self.printMeanOutOfRailTime(dispersion_results)

        self.meanApogeeTime(dispersion_results)

        self.meanApogeeXPosition(dispersion_results)

        self.meanApogeeYPosition(dispersion_results)

        self.meanImpactTime(dispersion_results)

        self.meanImpactVelocity(dispersion_results)

        self.meanImpactXPosition(dispersion_results)

        self.meanImpactYPosition(dispersion_results)

        self.meanMaximumVelocity(dispersion_results)

        self.meanNumberOfParachuteEvents(dispersion_results)

        self.meanDrogueFullyInflatedTime(dispersion_results)

        self.meanDrogueFullyVelocity(dispersion_results)

        self.meanDrogueTriggerTime(dispersion_results)

        return None

    def allInfo(self):
        dispersion_results = self.dispersion_results

        self.plotEllipses(dispersion_results, self.image, self.actual_landing_point)

        self.plotApogeeAltitude(dispersion_results)

        self.plotOutOfRailVelocity(dispersion_results)

        self.plotStaticMargin(dispersion_results)

        self.plotLateralWindSpeed(dispersion_results)

        self.plotFrontalWindSpeed(dispersion_results)

        self.plotOutOfRailTime(dispersion_results)

        self.plotApogeeTime(dispersion_results)

        self.plotApogeeXPosition(dispersion_results)

        self.plotApogeeYPosition(dispersion_results)

        self.plotImpactTime(dispersion_results)

        self.plotImpactVelocity(dispersion_results)

        self.plotImpactXPosition(dispersion_results)

        self.plotImpactYPosition(dispersion_results)

        self.plotMaximumVelocity(dispersion_results)

        self.plotNumberOfParachuteEvents(dispersion_results)

        self.plotDrogueFullyInflatedTime(dispersion_results)

        self.plotDrogueFullyVelocity(dispersion_results)

        self.plotDrogueTriggerTime(dispersion_results)

        return None
