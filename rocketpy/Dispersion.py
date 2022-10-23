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

## Tasks :
# TODO: Allow each parameter to be varied following an specific probability distribution
# TODO: Test simulations under different scenarios (with both parachutes, with only main chute, etc)
# TODO: Add unit tests
# TODO: Adjust the notebook to the new version of the code
# TODO: Implement MRS method
# TODO: Implement functions from compareDispersions notebook


class Dispersion:

    """Monte Carlo analysis to predict probability distributions of the rocket's
    landing point, apogee and other relevant information.

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
        """Sets the distribution function to be used in the analysis.

        Parameters
        ----------
        distribution_type : string
            The type of distribution to be used in the analysis. It can be
            'uniform', 'normal', 'lognormal', etc.

        Returns
        -------
        np.random distribution function
            The distribution function to be used in the analysis.
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
                    # Flight class was not inputted
                    # check if missing parameter is required
                    if self.flight_inputs[missing_input] == "required":
                        warnings.warn(f'Missing "{missing_input}" in dictionary')
                    else:  # if not required, uses default value
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

    def __process_rail_buttons_from_dict(self, dictionary):
        """Check if all the relevant inputs for the RailButtons class are present
        in the dispersion dictionary, input the missing ones and return the
        modified dictionary.

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
            Modified dictionary with the processed rail buttons parameters.
        """

        if not all(
            rail_buttons_input in dictionary
            for rail_buttons_input in self.rail_buttons_inputs.keys()
        ):
            # Iterate through missing inputs
            for missing_input in (
                set(self.rail_buttons_inputs.keys()) - dictionary.keys()
            ):
                missing_input = str(missing_input)
                # Add to the dict
                try:
                    dictionary[missing_input] = [
                        getattr(self.rocket, missing_input)
                    ]
                except:
                    # class was not inputted
                    # checks if missing parameter is required
                    if self.rail_buttons_inputs[missing_input] == "required":
                        warnings.warn(f'Missing "{missing_input}" in dictionary')
                    else:
                        # if not, uses default value
                        dictionary[missing_input] = [
                            self.rail_buttons_inputs[missing_input]
                        ]

        return dictionary

    def __process_aerodynamic_surfaces_from_dict(self, dictionary):
        """Still not implemented.
        Must check if all the relevant inputs for the AerodynamicSurfaces class
        are present in the dispersion dictionary, input the missing ones and
        return the modified dictionary.
        Something similar to the __process_parachute_from_dict method can be
        used here, since aerodynamic surfaces are optional for the simulation.

        Parameters
        ----------
        dictionary : _type_
            _description_
        """

        # Check the number of fin sets, noses, and tails
        self.nose_names = []
        self.finSet_names = []
        self.tail_names = []
        # Get names from the input dictionary
        for var in dictionary.keys():
            if "nose" in var:
                self.nose_names.append(var).split("_")[1]
            elif "finSet" in var:
                self.finSet_names.append(var).split("_")[1]
            elif "tail" in var:
                self.tail_names.append(var).split("_")[1]
        # Get names from the rocket object
        for surface in self.rocket.aerodynamicSurfaces:
            if isinstance(surface, NoseCone):
                self.nose_names.append(surface.name)
            elif isinstance(surface, (TrapezoidalFins, EllipticalFins)):
                self.finSet_names.append(surface.name)
            elif isinstance(surface, Tail):
                self.tail_names.append(surface.name)
        # Remove duplicates
        self.nose_names = list(set(self.nose_names))
        self.finSet_names = list(set(self.finSet_names))
        self.tail_names = list(set(self.tail_names))

        # Check if there are enough arguments for each kind of aero surface

        # Iterate through nose names
        for name in self.nose_names:
            # Iterate through aerodynamic surface available at rocket object
            for surface in self.rocket.aerodynamicSurfaces:
                if surface.name == name and isinstance(surface, NoseCone):
                    # in case we find the corresponding nose, check if all the
                    # inputs are present in the dictionary
                    for input in self.nose_inputs.keys():
                        _, _, parameter = input.split("_")
                        if f"nose_{name}_{parameter}" not in dictionary:
                            # Try to get the value from the rocket object
                            try:
                                dictionary[f"nose_{name}_{parameter}"] = [
                                    getattr(surface, parameter)
                                ]
                            except:
                                # If not possible, check if the parameter is required
                                if self.nose_inputs[input] == "required":
                                    warnings.warn(f'Missing "{input}" in dictionary')
                                else:
                                    # If not required, use default value
                                    dictionary[f"nose_{name}_{parameter}"] = [
                                        self.nose_inputs[input]
                                    ]

        # Iterate through fin sets names
        for name in self.finSet_names:
            # Iterate through aerodynamic surface available at rocket object
            for surface in self.rocket.aerodynamicSurfaces:
                if surface.name == name and isinstance(
                    surface, (TrapezoidalFins, EllipticalFins)
                ):
                    # in case we find the corresponding fin set, check if all the
                    # inputs are present in the dictionary
                    for input in self.fins_inputs.keys():
                        _, _, parameter = input.split("_")
                        if f"finSet_{name}_{parameter}" not in dictionary:
                            # Try to get the value from the rocket object
                            try:
                                dictionary[f"finSet_{name}_{parameter}"] = [
                                    getattr(surface, parameter)
                                ]
                            except:
                                # If not possible, check if the parameter is required
                                if self.fins_inputs[input] == "required":
                                    warnings.warn(f'Missing "{input}" in dictionary')
                                else:
                                    # If not required, use default value
                                    dictionary[f"finSet_{name}_{parameter}"] = [
                                        self.fins_inputs[input]
                                    ]

        # Iterate through tail names
        for name in self.tail_names:
            # Iterate through aerodynamic surface available at rocket object
            for surface in self.rocket.aerodynamicSurfaces:
                if surface.name == name and isinstance(surface, Tail):
                    # in case we find the corresponding tail, check if all the
                    # inputs are present in the dictionary
                    for input in self.tail_inputs.keys():
                        _, _, parameter = input.split("_")
                        if f"tail_{name}_{parameter}" not in dictionary:
                            # Try to get the value from the rocket object
                            try:
                                dictionary[f"tail_{name}_{parameter}"] = [
                                    getattr(surface, parameter)
                                ]
                            except:
                                # If not possible, check if the parameter is required
                                if self.tail_inputs[input] == "required":
                                    warnings.warn(f'Missing "{input}" in dictionary')
                                else:
                                    # If not required, use default value
                                    dictionary[f"tail_{name}_{parameter}"] = [
                                        self.tail_inputs[input]
                                    ]

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
        # TODO: Add more options of motor (i.e. Liquid and Hybrids)

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
        # Check if there is any missing input for the environment
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
                    # First try to catch value from the Environment object if passed
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
        # Get the number and names of parachutes
        self.parachute_names = []
        for key in dictionary.keys():
            if "parachute_" in key:
                self.parachute_names.append(key.split("_")[1])
        # Remove duplicates
        self.parachute_names = list(set(self.parachute_names))

        # Check if there is enough arguments for defining each parachute
        for name in self.parachute_names:
            for parachute_input in self.parachute_inputs.keys():
                _, _, parameter = parachute_input.split("_")
                if "parachute_{}_{}".format(name, parameter) not in dictionary.keys():
                    try:  # Try to get the value from the Parachute object
                        for chute in self.rocket.parachutes:
                            if getattr(chute, "name") == name:
                                dictionary[
                                    "parachute_{}_{}".format(name, parameter)
                                ] = [getattr(chute, parameter)]
                    except:  # Class not passed
                        if self.parachute_inputs[parachute_input] == "required":
                            warnings.warn(
                                "Missing {} for parachute {} in dictionary, which is required to run a simulation".format(
                                    parachute_input.split("_")[2], name
                                )
                            )
                        else:
                            dictionary["parachute_{}_{}".format(name, parameter)] = [
                                self.parachute_inputs[parachute_input],
                            ]

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
                except:
                    raise AttributeError(
                        f"Please check if the parameter {parameter_key} was inputted"
                        "correctly in dispersion_dictionary."
                        " Dictionary values must be either tuple or lists."
                        " If single value, the corresponding Class must"
                        " be inputted in the run_dispersion method."
                    )

            ## Third corrections - SolidMotor
            elif parameter_key in self.solid_motor_inputs.keys():
                try:
                    dictionary[parameter_key] = (
                        getattr(self.motor, parameter_key),
                        parameter_value,
                    )
                except:
                    raise AttributeError(
                        f"Please check if the parameter {parameter_key} was inputted"
                        "correctly in dispersion_dictionary."
                        " Dictionary values must be either tuple or lists."
                        " If single value, the corresponding Class must"
                        " be inputted in the run_dispersion method."
                    )

            # Fourth correction - Rocket
            elif parameter_key in self.rocket_inputs.keys():
                try:
                    dictionary[parameter_key] = (
                        getattr(self.rocket, parameter_key),
                        parameter_value,
                    )
                except:
                    raise AttributeError(
                        f"Please check if the parameter {parameter_key} was inputted"
                        "correctly in dispersion_dictionary."
                        " Dictionary values must be either tuple or lists."
                        " If single value, the corresponding Class must"
                        " be inputted in the run_dispersion method."
                    )

            # Fifth correction - Flight
            elif parameter_key in self.flight_inputs.keys():
                try:
                    dictionary[parameter_key] = (
                        getattr(self.flight, parameter_key),
                        parameter_value,
                    )
                except:
                    raise AttributeError(
                        f"Please check if the parameter {parameter_key} was inputted"
                        "correctly in dispersion_dictionary."
                        " Dictionary values must be either tuple or lists."
                        " If single value, the corresponding Class must"
                        " be inputted in the run_dispersion method."
                    )
                    print(traceback.format_exc())

        # The analysis parameter dictionary is ready! Now we have mean and stdev
        # for all parameters

        return dictionary

    def __check_initial_objects(self):
        """Create rocketpy objects (Environment, Motor, Rocket, Flight) in case
        that they were not created yet.

        Returns
        -------
        None
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
                inertiaI=self.dispersion_dictionary["inertiaI"][0],
                inertiaZ=self.dispersion_dictionary["inertiaZ"][0],
                distanceRocketPropellant=self.dispersion_dictionary[
                    "distanceRocketPropellant"
                ][0],
                distanceRocketNozzle=self.dispersion_dictionary["distanceRocketNozzle"][
                    0
                ],
                powerOffDrag=self.dispersion_dictionary["powerOffDrag"][0],
                powerOnDrag=self.dispersion_dictionary["dispersion_dictionary"][0],
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

    def __export_flight_data(
        self,
        flight_setting,
        flight,
        exec_time,
        dispersion_input_file,
        dispersion_output_file,
        variables=None,
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

        # In case not variables are passed, export default variables
        if not isinstance(variables, list):
            variables = [
                "apogee",
                "apogeeTime",
                "apogeeX",
                "apogeeY",
                "executionTime",
                "finalStaticMargin",
                "frontalSurfaceWind",
                "impactVelocity",
                "initialStaticMargin",
                "lateralSurfaceWind",
                "maxAcceleration",
                "maxAccelerationTime",
                "maxSpeed",
                "maxSpeedTime",
                "numberOfEvents",
                "outOfRailStaticMargin",
                "outOfRailTime",
                "outOfRailVelocity",
                "tFinal",
                "xImpact",
                "yImpact",
            ]
        else:  # Check if variables are valid and raise error if not
            if not all([isinstance(var, str) for var in variables]):
                raise TypeError("Variables must be strings.")

        # First, capture the flight data that are saved in the flight object
        attributes_list = list(set(dir(flight)).intersection(variables))
        flight_result = {}
        for var in attributes_list:
            flight_result[str(var)] = getattr(flight, var)

        # Second, capture data that needs to be calculated
        for var in list(set(variables) - set(attributes_list)):
            if var == "executionTime":
                flight_result[str(var)] = exec_time
            elif var == "numberOfEvents":
                flight_result[str(var)] = len(flight.parachuteEvents)
            else:
                raise ValueError(f"Variable {var} could not be found.")

        # Take care of parachute results
        for trigger_time, parachute in flight.parachuteEvents:
            flight_result[parachute.name + "_triggerTime"] = trigger_time
            flight_result[parachute.name + "_inflatedTime"] = (
                trigger_time + parachute.lag
            )
            flight_result[parachute.name + "_inflatedVelocity"] = flight.speed(
                trigger_time + parachute.lag
            )

        # Write flight setting and results to file
        flight_setting.pop("thrust", None)
        dispersion_input_file.write(str(flight_setting) + "\n")
        dispersion_output_file.write(str(flight_result) + "\n")

        return None

    def __export_flight_data_error(setting, flight_setting, dispersion_error_file):
        """Saves flight error in a .txt

        Parameters
        ----------
        setting : _type_
            _description_
        dispersion_error_file : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """

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

            # Clean up aerodynamic surfaces
            rocket_dispersion.aerodynamicSurfaces = []  # Remove all surfaces

            # Add rocket nose, fins and tail
            # Nose
            for nose in self.nose_names:
                rocket_dispersion.addNose(
                    length=setting[f"nose_{nose}_length"],
                    kind=setting[f"nose_{nose}_kind"],
                    distanceToCM=setting[f"nose_{nose}_distanceToCM"],
                    name=nose,
                )

            # Fins
            for finSet in self.finSet_names:
                # TODO: Allow elliptical fins as well
                rocket_dispersion.addTrapezoidalFins(
                    n=setting[f"finSet_{finSet}_numberOfFins"],
                    rootChord=setting[f"finSet_{finSet}_rootChord"],
                    tipChord=setting[f"finSet_{finSet}_tipChord"],
                    span=setting[f"finSet_{finSet}_span"],
                    distanceToCM=setting[f"finSet_{finSet}_distanceToCM"],
                    radius=setting[f"finSet_{finSet}_radius"],
                    airfoil=setting[f"finSet_{finSet}_airfoil"],
                    name=finSet,
                )

            # Tail
            for tail in self.tail_names:
                rocket_dispersion.addTail(
                    topRadius=setting[f"tail_{tail}_topRadius"],
                    bottomRadius=setting[f"tail_{tail}_bottomRadius"],
                    length=setting[f"tail_{tail}_length"],
                    distanceToCM=setting[f"tail_{tail}_distanceToCM"],
                    radius=None,
                    name="Tail",
                )

            # Add parachutes
            rocket_dispersion.parachutes = []  # Remove existing parachutes
            for name in self.parachute_names:
                rocket_dispersion.addParachute(
                    name=name,
                    CdS=setting["parachute_" + name + "_CdS"],
                    trigger=setting["parachute_" + name + "_trigger"],
                    samplingRate=setting["parachute_" + name + "_samplingRate"],
                    lag=setting["parachute_" + name + "_lag"],
                    noise=setting["parachute_" + name + "_noise"],
                )

            # TODO: Remove hard-coded rail buttons definition
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

    def import_results(self):
        """Import dispersion results from .txt file and save it into a dictionary.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Initialize variable to store all results
        dispersion_results = {}

        # Get all dispersion results
        # Open the file
        file = open(self.filename.split(".")[0] + ".disp_outputs.txt", "r+")

        # Read each line of the file and convert to dict
        for line in file:
            # Skip comments lines
            if line[0] != "{":
                continue
            # Evaluate results and store them
            flight_result = eval(line)
            # Append to the list
            for parameter_key, parameter_value in flight_result.items():
                if parameter_key not in dispersion_results.keys():
                    # Create a new list to store the parameter
                    dispersion_results[parameter_key] = [parameter_value]
                else:
                    # Append the parameter value to the list
                    dispersion_results[parameter_key].append(parameter_value)

        # Close data file
        file.close()

        # Calculate the number of flights simulated
        len_dict = {key: len(value) for key, value in dispersion_results.items()}
        if min(len_dict.values()) - max(len_dict.values()) > 1:
            print(
                "Warning: The number of simulations imported from the file is not "
                "the same for all parameters. The number of simulations will be "
                "set to the minimum number of simulations found."
            )
        self.num_of_loaded_sims = min(len_dict.values())

        # Print the number of flights simulated
        print(
            f"A total of {self.num_of_loaded_sims} simulations were loaded from"
            f" the following file: {self.filename.split('.')[0] + '.disp_outputs.txt'}"
        )

        # Save the results as an attribute of the class
        self.dispersion_results = dispersion_results

        return None

    # Start the processing analysis

    def process_results(self, variables=None):
        """Save the mean and standard deviation of each parameter in the results
        dictionary. Create class attributes for each parameter.

        Parameters
        ----------
        variables : list, optional
            List of variables to be processed. If None, all variables will be
            processed. The default is None. Example: ['outOfRailTime', 'apogeeTime']

        Returns
        -------
        None
        """
        if isinstance(variables, list):
            for result in variables:
                mean = np.mean(self.dispersion_results[result])
                stdev = np.std(self.dispersion_results[result])
                setattr(self, str(result), (mean, stdev))
        else:
            for result in self.dispersion_results.keys():
                mean = np.mean(self.dispersion_results[result])
                stdev = np.std(self.dispersion_results[result])
                setattr(self, str(result), (mean, stdev))
        return None

    # TODO: print as a table instead of prints
    def print_results(self, variables=None):
        """Print the mean and standard deviation of each parameter in the results
        dictionary or of the variables passed as argument.

        Parameters
        ----------
        variables : list, optional
            List of variables to be processed. If None, all variables will be
            processed. The default is None. Example: ['outOfRailTime', 'apogee']

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If the variable passed as argument is not a string.
        """
        # Check if the variables argument is a list, if not, use all variables
        if not isinstance(variables, list):
            variables = self.dispersion_results.keys()

        # Check if the variables are strings
        if not all(isinstance(var, str) for var in variables):
            raise TypeError("The list of variables must be a list of strings.")

        for var in variables:
            tp = getattr(self, var)  # Get the tuple with the mean and stdev
            print("{}: \u03BC = {:.3f}, \u03C3 = {:.3f}".format(var, tp[0], tp[1]))

        return None

    def plot_results(self, variables=None):
        """_summary_

        Parameters
        ----------
        variables : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        TypeError
            _description_
        """
        # Check if the variables argument is a list, if not, use all variables
        if not isinstance(variables, list):
            variables = self.dispersion_results.keys()

        # Check if the variables are strings
        if not all(isinstance(var, str) for var in variables):
            raise TypeError("The list of variables must be a list of strings.")

        for var in variables:
            plt.figure()
            plt.hist(
                self.dispersion_results[var],
            )
            plt.title("Histogram of " + var)
            # plt.xlabel("Time (s)")
            plt.ylabel("Number of Occurrences")
            plt.show()

        return None

    # TODO: Create evolution plots to analyze convergence

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
