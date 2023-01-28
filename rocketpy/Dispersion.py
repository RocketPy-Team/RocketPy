# -*- coding: utf-8 -*-

__author__ = "Mateus Stano Junqueira, Sofia Lopes Suesdek Rocha, Guilherme Fernandes Alves, Bruno Abdulklech Sorban"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"


import math
import traceback
import types
from time import process_time, time

import matplotlib.pyplot as plt
import numpy as np
import simplekml
from imageio import imread
from IPython.display import display
from matplotlib.patches import Ellipse
from numpy.random import *

from rocketpy.utilities import invertedHaversine

from .AeroSurfaces import EllipticalFins, NoseCone, Tail, TrapezoidalFins
from .Environment import Environment
from .Flight import Flight
from .Function import Function
from .Motor import SolidMotor
from .Rocket import Rocket


class Dispersion:

    """Monte Carlo analysis to predict probability distributions of the rocket's
    landing point, apogee and other relevant information.

    Parameters
    ----------
    filename : string
        The name of the file containing the data to be used in the analysis.

    Attributes
    ---------- # TODO: finish documentation
        Dispersion.filename : string
            Directory and name of dispersion files. When running a new simulation,
            this parameter represents the initial part of the export filenames
            (e.g. 'filename.disp_outputs.txt'). When analyzing the results of a
            previous simulation, this parameter shall be the .txt filename containing
            the outputs of a previous ran dispersion analysis.
        Dispersion.inputs_dict : dict
            Contains information regarding the input arguments of the
            classes. Its keys refers to each of the classes that must be defined during
            the simulation. Its values are dictionaries where the keys are the input
            arguments of each class and the values are either the string "required"
            (meaning it is not an optional argument) or the default value if that argument
            is optional.
        Dispersion.dispersion_results : dict
            Holds dispersion results.
        Dispersion.dispersion_dictionary : dict
            Contains inputs to run dispersion
        Dispersion.nose_names = []
        Dispersion.finSet_names = []
        Dispersion.tail_names = []
        Dispersion.parachute_names = []
        Dispersion.distributionFunc = None
        Dispersion.distribution_type = None
        Dispersion.environment = None
        Dispersion.flight = None
        Dispersion.motor = None
        Dispersion.rocket = None
        Dispersion.rocket_dispersion = None
        Dispersion.number_of_simulations = 0
        Dispersion.num_of_loaded_sims = 0
        Dispersion.start_time = 0

        Dispersion.num_of_loaded_sims : int
            The number of simulations loaded from the file.
        Dispersion.num_of_sims : int
            The number of simulations to be performed.
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
        self.inputs_dict = {
            "environment": {
                "railLength": "required",
                "gravity": 9.80665,
                "date": None,
                "latitude": 0,
                "longitude": 0,
                "elevation": 0,
                "datum": "WGS84",
                "timeZone": "UTC",
            },
            "solidmotor": {
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
                "grainsCenterOfMassPosition": "required",
                "nozzlePosition": 0,
                "coordinateSystemOrientation": "nozzleToCombustionChamber",
            },
            "rocket": {
                "mass": "required",
                "radius": "required",
                "inertiaI": "required",
                "inertiaZ": "required",
                "powerOffDrag": "required",
                "powerOnDrag": "required",
                "centerOfDryMassPosition": 0,
                "coordinateSystemOrientation": "tailToNose",
                "powerOffDragFactor": 1,
                "powerOnDragFactor": 1,
                "motorPosition": "required",
            },
            "nose": {
                "length": "required",
                "kind": "Von Karman",
                "position": "required",
                "name": "NoseCone",
            },
            "fins": {
                "n": "required",
                "rootChord": "required",
                "tipChord": "required",
                "span": "required",
                "position": "required",
                "cantAngle": 0,
                "radius": None,
                "airfoil": None,
            },
            "tail": {
                "topRadius": "required",
                "bottomRadius": "required",
                "length": "required",
                "position": "required",
            },
            "railbuttons": {
                "positionFirstRailButton": "required",
                "positionSecondRailButton": "required",
                "railButtonAngularPosition": 45,
            },
            "parachute": {
                "CdS": "required",
                "trigger": "required",
                "samplingRate": "required",
                "lag": "required",
                "noise": (0, 0, 0),
            },
            "flight": {
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
            },
        }
        # TODO: add all exportable attributes to this list
        self.exportable_list = [
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
        # Initialize variables so they can be accessed by MATLAB
        self.dispersion_results = {}
        self.dispersion_dictionary = {}
        self.nose_names = []
        self.finSet_names = []
        self.tail_names = []
        self.parachute_names = []
        self.distributionFunc = None
        self.distribution_type = None
        self.environment = None
        self.flight = None
        self.motor = None
        self.rocket = None
        self.rocket_dispersion = None
        self.number_of_simulations = 0
        self.num_of_loaded_sims = 0
        self.start_time = 0

        return None

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
            raise ValueError(
                "Distribution type not recognized. Please use a valid distribution type."
            )

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
        # First check if all the inputs for the flight class are present in the
        # dictionary, if not, input the missing ones
        if not all(
            flight_input in dictionary
            for flight_input in self.inputs_dict["flight"].keys()
        ):
            # Iterate through missing inputs
            for missing_input in (
                set(self.inputs_dict["flight"].keys()) - dictionary.keys()
            ):
                missing_input = str(missing_input)
                # Add to the dict
                try:
                    # First try to catch value from the Flight object if passed
                    dictionary[missing_input] = [getattr(self.flight, missing_input)]
                except AttributeError:
                    # Flight class was not inputted
                    # check if missing parameter is required
                    if self.inputs_dict["flight"][missing_input] == "required":
                        raise ValueError(
                            "The input {} is required for the Flight class.".format(
                                missing_input
                            )
                        )
                    else:  # if not required, uses default value
                        dictionary[missing_input] = [
                            self.inputs_dict["flight"][missing_input]
                        ]

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

        # Checks if there are any missing rocket inputs in dictionary
        if not all(
            rocket_input in dictionary
            for rocket_input in self.inputs_dict["rocket"].keys()
        ):
            # Iterate through missing inputs
            for missing_input in (
                set(self.inputs_dict["rocket"].keys()) - dictionary.keys()
            ):
                missing_input = str(missing_input)
                # Add to the dict
                try:
                    dictionary[missing_input] = [getattr(self.rocket, missing_input)]
                except AttributeError:
                    # class was not inputted
                    # checks if missing parameter is required
                    if self.inputs_dict["rocket"][missing_input] == "required":
                        raise ValueError(
                            "The input {} is required for the Rocket class.".format(
                                missing_input
                            )
                        )
                    else:  # if not, uses default value
                        dictionary[missing_input] = [
                            self.inputs_dict["rocket"][missing_input]
                        ]

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

        # if there is any missing input
        if not all(
            rail_buttons_input in dictionary
            for rail_buttons_input in self.inputs_dict["railbuttons"].keys()
        ):
            # Iterate through missing inputs
            for missing_input in (
                set(self.inputs_dict["railbuttons"].keys()) - dictionary.keys()
            ):
                missing_input = str(missing_input)
                # Add to the dict
                try:
                    dictionary[missing_input] = [getattr(self.rocket, missing_input)]
                except AttributeError:
                    # class was not inputted
                    # checks if missing parameter is required
                    if self.inputs_dict["railbuttons"][missing_input] == "required":
                        raise ValueError(
                            "The input {} is required for the RailButtons class.".format(
                                missing_input
                            )
                        )
                    else:
                        # if not, uses default value
                        dictionary[missing_input] = [
                            self.inputs_dict["railbuttons"][missing_input]
                        ]
        # else, all inputs are there
        else:
            # Iterate through possible inputs
            for rail_buttons_input in self.inputs_dict["railbuttons"].keys():
                # if value of input is not list or tuple
                if not isinstance(dictionary[rail_buttons_input], (tuple, list)):
                    dictionary[rail_buttons_input] = (
                        getattr(self.rocket, rail_buttons_input),
                        dictionary[rail_buttons_input],
                    )
        return dictionary

    def __process_aerodynamic_surfaces_from_dict(self, dictionary):
        """Check if all the relevant inputs for the AerodynamicSurfaces class
        are present in the dispersion dictionary, input the missing ones and
        return the modified dictionary.

        Parameters
        ----------
        dictionary : dict
            Dictionary containing the parameters to be varied in the dispersion
            simulation. The keys of the dictionary are the names of the parameters
            to be varied, and the values can be either tuple or list. If the value
            is a single value, the corresponding class of the parameter need to
            be passed on the run_dispersion method.
        """

        # Check the number of fin sets, noses, and tails
        self.nose_names, self.finSet_names, self.tail_names = [], [], []

        # Get names from the input dictionary
        for var in dictionary.keys():
            if "nose" in var:
                self.nose_names.append(var.split("_")[1])
            elif "finSet" in var:
                self.finSet_names.append(var.split("_")[1])
            elif "tail" in var:
                self.tail_names.append(var.split("_")[1])
        # Get names from the rocket object
        for surface, position in self.rocket.aerodynamicSurfaces:
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
            for surface, position in self.rocket.aerodynamicSurfaces:
                if surface.name == name and isinstance(surface, NoseCone):
                    # in case we find the corresponding nose, check if all the
                    # inputs are present in the dictionary
                    for input in self.inputs_dict["nose"].keys():
                        if f"nose_{name}_{input}" not in dictionary:
                            # Try to get the value from the rocket object
                            try:
                                # special check for position
                                if input == "position":
                                    dictionary[f"nose_{name}_{input}"] = [position]
                                else:
                                    dictionary[f"nose_{name}_{input}"] = [
                                        getattr(surface, input)
                                    ]
                            except AttributeError:
                                # If not possible, check if the input is required
                                if self.inputs_dict["nose"][input] == "required":
                                    raise ValueError(
                                        "The input {} is required for the NoseCone class.".format(
                                            input
                                        )
                                    )

                                else:
                                    # If not required, use default value
                                    dictionary[f"nose_{name}_{input}"] = [
                                        self.inputs_dict["nose"][input]
                                    ]
                        # if the input is in the dictionary, then check format
                        else:
                            # if value of input is not list or tuple
                            if not isinstance(
                                dictionary[f"nose_{name}_{input}"], (list, tuple)
                            ):
                                # Try to get value from class to construct tuple
                                if len(self.rocket.aerodynamicSurfaces) > 0:
                                    for (
                                        surface,
                                        position,
                                    ) in self.rocket.aerodynamicSurfaces:
                                        if isinstance(surface, NoseCone):
                                            try:
                                                # special check for position
                                                if input == "position":
                                                    dictionary[
                                                        f"nose_{name}_{input}"
                                                    ] = (
                                                        position,
                                                        dictionary[
                                                            f"nose_{name}_{input}"
                                                        ],
                                                    )
                                                else:
                                                    dictionary[
                                                        f"nose_{name}_{input}"
                                                    ] = (
                                                        getattr(surface, input),
                                                        dictionary[
                                                            f"nose_{name}_{input}"
                                                        ],
                                                    )

                                            except AttributeError:
                                                raise AttributeError(
                                                    f"Please check if the parameter "
                                                    + f"nose_{name}_{input}"
                                                    + " was inputted"
                                                    " correctly in dispersion_dictionary."
                                                    " Dictionary values must be either tuple or lists."
                                                    " If single value, a Rocket object with at least one"
                                                    " NoseCone must be inputted in the run_dispersion method"
                                                    " and the NoseCone must have {input} defined"
                                                )

                                else:
                                    raise AttributeError(
                                        f"Please check if the parameter "
                                        + f"nose_{name}_{input}"
                                        + " was inputted"
                                        " correctly in dispersion_dictionary."
                                        " Dictionary values must be either tuple or lists."
                                        " If single value, a Rocket object with at least one"
                                        " NoseCone must be inputted in the run_dispersion method."
                                    )

        # Iterate through fin sets names
        for name in self.finSet_names:
            # Iterate through aerodynamic surface available at rocket object
            for surface, position in self.rocket.aerodynamicSurfaces:
                if surface.name == name and isinstance(
                    surface, (TrapezoidalFins, EllipticalFins)
                ):
                    # in case we find the corresponding fin set, check if all the
                    # inputs are present in the dictionary
                    for input in self.inputs_dict["fins"].keys():
                        if f"finSet_{name}_{input}" not in dictionary:
                            # Try to get the value from the rocket object
                            try:
                                # special check for position
                                if input == "position":
                                    dictionary[f"finSet_{name}_{input}"] = [position]
                                else:
                                    dictionary[f"finSet_{name}_{input}"] = [
                                        getattr(surface, input)
                                    ]
                            except AttributeError:
                                # If not possible, check if the input is required
                                if self.inputs_dict["fins"][input] == "required":
                                    raise ValueError(
                                        "The input {} is required for the Fins class.".format(
                                            input
                                        )
                                    )
                                else:
                                    # If not required, use default value
                                    dictionary[f"finSet_{name}_{input}"] = [
                                        self.inputs_dict["fins"][input]
                                    ]
                        # if the input is in the dictionary, then check format
                        else:
                            # if value of input is not list or tuple
                            if not isinstance(
                                dictionary[f"finSet_{name}_{input}"], (list, tuple)
                            ):
                                # Try to get value from class to construct tuple
                                if len(self.rocket.aerodynamicSurfaces) > 0:
                                    for (
                                        surface,
                                        position,
                                    ) in self.rocket.aerodynamicSurfaces:
                                        if isinstance(
                                            surface, (TrapezoidalFins, EllipticalFins)
                                        ):
                                            try:
                                                # special check for position
                                                if input == "position":
                                                    dictionary[
                                                        f"finSet_{name}_{input}"
                                                    ] = (
                                                        position,
                                                        dictionary[
                                                            f"finSet_{name}_{input}"
                                                        ],
                                                    )
                                                else:
                                                    dictionary[
                                                        f"finSet_{name}_{input}"
                                                    ] = (
                                                        getattr(surface, input),
                                                        dictionary[
                                                            f"finSet_{name}_{input}"
                                                        ],
                                                    )

                                            except AttributeError:
                                                raise AttributeError(
                                                    f"Please check if the parameter "
                                                    + f"finSet_{name}_{input}"
                                                    + " was inputted"
                                                    " correctly in dispersion_dictionary."
                                                    " Dictionary values must be either tuple or lists."
                                                    " If single value, a Rocket object with at least one"
                                                    " FinSet must be inputted in the run_dispersion method"
                                                    " and the FinSet must have {input} defined"
                                                )

                                else:
                                    raise AttributeError(
                                        f"Please check if the parameter "
                                        + f"finSet_{name}_{input}"
                                        + " was inputted"
                                        " correctly in dispersion_dictionary."
                                        " Dictionary values must be either tuple or lists."
                                        " If single value, a Rocket object with at least one"
                                        " FinSet must be inputted in the run_dispersion method."
                                    )

        # Iterate through tail names
        for name in self.tail_names:
            # Iterate through aerodynamic surface available at rocket object
            for surface, position in self.rocket.aerodynamicSurfaces:
                if surface.name == name and isinstance(surface, Tail):
                    # in case we find the corresponding tail, check if all the
                    # inputs are present in the dictionary
                    for input in self.inputs_dict["tail"].keys():
                        if f"tail_{name}_{input}" not in dictionary:
                            # Try to get the value from the rocket object
                            try:
                                # special check for position
                                if input == "position":
                                    dictionary[f"tail_{name}_{input}"] = [position]
                                else:
                                    dictionary[f"tail_{name}_{input}"] = [
                                        getattr(surface, input)
                                    ]
                            except AttributeError:
                                # If not possible, check if the input is required
                                if self.inputs_dict["tail"][input] == "required":
                                    raise ValueError(
                                        "The input {} is required for the Tail class.".format(
                                            input
                                        )
                                    )
                                else:
                                    # If not required, use default value
                                    dictionary[f"tail_{name}_{input}"] = [
                                        self.inputs_dict["tail"][input]
                                    ]
                        # if the input is in the dictionary, then check format
                        else:
                            # if value of input is not list or tuple
                            if not isinstance(
                                dictionary[f"tail_{name}_{input}"], (list, tuple)
                            ):
                                # Try to get value from class to construct tuple
                                if len(self.rocket.aerodynamicSurfaces) > 0:
                                    for (
                                        surface,
                                        position,
                                    ) in self.rocket.aerodynamicSurfaces:
                                        if isinstance(surface, Tail):
                                            try:
                                                # special check for position
                                                if input == "position":
                                                    dictionary[
                                                        f"tail_{name}_{input}"
                                                    ] = (
                                                        position,
                                                        dictionary[
                                                            f"tail_{name}_{input}"
                                                        ],
                                                    )
                                                else:
                                                    dictionary[
                                                        f"tail_{name}_{input}"
                                                    ] = (
                                                        getattr(surface, input),
                                                        dictionary[
                                                            f"tail_{name}_{input}"
                                                        ],
                                                    )

                                            except AttributeError:
                                                raise AttributeError(
                                                    f"Please check if the parameter "
                                                    + f"tail_{name}_{input}"
                                                    + " was inputted"
                                                    " correctly in dispersion_dictionary."
                                                    " Dictionary values must be either tuple or lists."
                                                    " If single value, a Rocket object with at least one"
                                                    " Tail must be inputted in the run_dispersion method"
                                                    " and the Tail must have {input} defined"
                                                )

                                else:
                                    raise AttributeError(
                                        f"Please check if the parameter "
                                        + f"tail_{name}_{input}"
                                        + " was inputted"
                                        " correctly in dispersion_dictionary."
                                        " Dictionary values must be either tuple or lists."
                                        " If single value, a Rocket object with at least one"
                                        " Tail must be inputted in the run_dispersion method."
                                    )

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
            motor_input in dictionary
            for motor_input in self.inputs_dict["solidmotor"].keys()
        ):
            # Iterate through missing inputs
            for missing_input in (
                set(self.inputs_dict["solidmotor"].keys()) - dictionary.keys()
            ):
                missing_input = str(missing_input)
                # Add to the dict
                try:
                    dictionary[missing_input] = [getattr(self.motor, missing_input)]
                except AttributeError:
                    # class was not inputted
                    # checks if missing parameter is required
                    if self.inputs_dict["solidmotor"][missing_input] == "required":
                        # if motor position in relation to rocket
                        if missing_input == "motor_position":
                            raise ValueError(
                                'Missing motor\'s "position" argument in the input dictionary.'
                                " This refers to the position of the motor's nozzle in relation"
                                " to the rocket's coordinate system, used in the Rocket.addMotor method."
                            )
                        else:
                            raise ValueError(
                                "The input {} is required for the SolidMotor class.".format(
                                    missing_input
                                )
                            )
                    else:  # if not uses default value
                        dictionary[missing_input] = [
                            self.inputs_dict["solidmotor"][missing_input]
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
            for environment_input in self.inputs_dict["environment"].keys()
        ):
            # Iterate through missing inputs
            for missing_input in (
                set(self.inputs_dict["environment"].keys()) - dictionary.keys()
            ):
                missing_input = str(missing_input)
                # Add to the dict
                try:
                    # First try to catch value from the Environment object if passed
                    dictionary[missing_input] = [
                        getattr(self.environment, missing_input)
                    ]
                except AttributeError:
                    # class was not inputted
                    # checks if missing parameter is required
                    if self.inputs_dict["environment"][missing_input] == "required":
                        raise ValueError(
                            "The input {} is required for the Environment class.".format(
                                missing_input
                            )
                        )
                    else:  # if not required, use default value
                        dictionary[missing_input] = [
                            self.inputs_dict["environment"][missing_input]
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
        # Get names from the rocket object
        for chute in self.rocket.parachutes:
            self.parachute_names.append(chute.name)
        # Remove duplicates
        self.parachute_names = list(set(self.parachute_names))

        # Check if there is enough arguments for defining each parachute
        for name in self.parachute_names:
            for parachute_input in self.inputs_dict["parachute"].keys():
                if (
                    "parachute_{}_{}".format(name, parachute_input)
                    not in dictionary.keys()
                ):
                    try:  # Try to get the value from the Parachute object
                        if len(self.rocket.parachutes) > 0:
                            for chute in self.rocket.parachutes:
                                if getattr(chute, "name") == name:
                                    dictionary[
                                        "parachute_{}_{}".format(name, parachute_input)
                                    ] = [getattr(chute, parachute_input)]
                        else:
                            raise Exception
                    except Exception:  # Class not passed
                        if self.inputs_dict["parachute"][parachute_input] == "required":
                            raise ValueError(
                                "The input {} is required for the Parachute class.".format(
                                    parachute_input
                                )
                            )
                        else:
                            dictionary[
                                "parachute_{}_{}".format(name, parachute_input)
                            ] = [
                                self.inputs_dict["parachute"][parachute_input],
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
                # try:
                if isinstance(parameter_value, types.FunctionType):
                    # Deal with trigger functions
                    dictionary[parameter_key] = parameter_value
                else:
                    dictionary[parameter_key] = (
                        getattr(
                            self.rocket.get_parachute_by_name(parachute_name), parameter
                        ),
                        parameter_value,
                    )

            ## Second corrections - Environment
            elif parameter_key in self.inputs_dict["environment"].keys():
                try:
                    dictionary[parameter_key] = (
                        getattr(self.environment, parameter_key),
                        parameter_value,
                    )
                except AttributeError:
                    raise AttributeError(
                        f"Please check if the parameter {parameter_key} was inputted"
                        "correctly in dispersion_dictionary."
                        " Dictionary values must be either tuple or lists."
                        " If single value, the corresponding Class must"
                        " be inputted in the run_dispersion method."
                    )

            ## Third corrections - SolidMotor
            elif parameter_key in self.inputs_dict["solidmotor"].keys():
                try:
                    dictionary[parameter_key] = (
                        getattr(self.motor, parameter_key),
                        parameter_value,
                    )
                except AttributeError:
                    raise AttributeError(
                        f"Please check if the parameter {parameter_key} was inputted"
                        " correctly in dispersion_dictionary."
                        " Dictionary values must be either tuple or lists."
                        " If single value, the corresponding Class must"
                        " be inputted in the run_dispersion method."
                    )

            # Fourth correction - Rocket
            elif parameter_key in self.inputs_dict["rocket"].keys():
                try:
                    dictionary[parameter_key] = (
                        getattr(self.rocket, parameter_key),
                        parameter_value,
                    )
                except AttributeError:
                    raise AttributeError(
                        f"Please check if the parameter {parameter_key} was inputted"
                        "correctly in dispersion_dictionary."
                        " Dictionary values must be either tuple or lists."
                        " If single value, the corresponding Class must"
                        " be inputted in the run_dispersion method."
                    )

            # Fifth correction - Flight
            elif parameter_key in self.inputs_dict["flight"].keys():
                try:
                    dictionary[parameter_key] = (
                        getattr(self.flight, parameter_key),
                        parameter_value,
                    )
                except AttributeError:
                    raise AttributeError(
                        f"Please check if the parameter {parameter_key} was inputted"
                        "correctly in dispersion_dictionary."
                        " Dictionary values must be either tuple or lists."
                        " If single value, the corresponding Class must"
                        " be inputted in the run_dispersion method."
                    )

        # The analysis parameter dictionary is ready! Now we have mean and stdev
        # for all parameters

        return dictionary

    def __create_initial_objects(self):
        """Create rocketpy objects (Environment, Motor, Rocket, Flight) in case
        that they were not created yet.

        Returns
        -------
        None
        """
        if self.environment is None:
            try:
                self.environment = Environment(
                    railLength=self.dispersion_dictionary["railLength"][0]
                )
            except:
                raise TypeError(
                    "Cannot define basic Environment. Missing railLength value in dictionary"
                )
        if self.motor is None:
            try:
                self.motor = SolidMotor(
                    thrustSource=self.dispersion_dictionary["thrust"][0],
                    burnOut=self.dispersion_dictionary["burnOutTime"][0],
                    grainNumber=self.dispersion_dictionary["grainNumber"][0],
                    grainDensity=self.dispersion_dictionary["grainDensity"][0],
                    grainOuterRadius=self.dispersion_dictionary["grainOuterRadius"][0],
                    grainInitialInnerRadius=self.dispersion_dictionary[
                        "grainInitialInnerRadius"
                    ][0],
                    grainInitialHeight=self.dispersion_dictionary["grainInitialHeight"][
                        0
                    ],
                    grainsCenterOfMassPosition=self.dispersion_dictionary[
                        "grainsCenterOfMassPosition"
                    ][0],
                )
            except:
                raise TypeError(
                    "Cannot define basic SolidMotor. Missing required parameters in dictionary"
                )
        if self.rocket is None:
            try:
                self.rocket = Rocket(
                    mass=self.dispersion_dictionary["mass"][0],
                    radius=self.dispersion_dictionary["radius"][0],
                    inertiaI=self.dispersion_dictionary["inertiaI"][0],
                    inertiaZ=self.dispersion_dictionary["inertiaZ"][0],
                    powerOffDrag=self.dispersion_dictionary["powerOffDrag"][0],
                    powerOnDrag=self.dispersion_dictionary["powerOnDrag"][0],
                )
                self.rocket.setRailButtons(
                    position=[0.2, -0.5]
                )  # TODO: needs to check if required
            except:
                raise TypeError(
                    "Cannot define basic Rocket and add rail buttons. Missing required parameters in dictionary"
                )
        if self.flight is None:
            try:
                self.flight = Flight(
                    rocket=self.rocket,
                    environment=self.environment,
                    inclination=self.dispersion_dictionary["inclination"][0],
                    heading=self.dispersion_dictionary["heading"][0],
                )
            except:
                raise TypeError(
                    "Cannot define basic Flight. Missing required parameters in dictionary"
                )
        return None

    def __yield_flight_setting(
        self, distribution_func, analysis_parameters, number_of_simulations
    ):
        """Yields a flight setting for the simulation

        Parameters
        ----------
        distribution_func : np.random distribution function
            The function that will be used to generate the random values.
        analysis_parameters : dict
            The dictionary with the parameters to be analyzed. This includes the
            mean and standard deviation of the parameters.
        number_of_simulations : int
            Number of simulations desired, must be non negative.
            This is needed when running a new simulation. Default is zero.

        Yields
        ------
        flight_setting: dict
            A dictionary with the flight setting for one simulation.

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
                elif isinstance(parameter_value, types.FunctionType):
                    # Deal with parachute triggers functions
                    flight_setting[parameter_key] = parameter_value
                elif isinstance(parameter_value, list):
                    # shuffles list and gets first item
                    shuffle(parameter_value)
                    flight_setting[parameter_key] = parameter_value[0]
                else:
                    # parameter_value is defined wrong
                    raise AttributeError(
                        "Something went wrong when analyzing the dispersion_dictionary."
                        " Please check if everything was inputed correctly."
                    )

            # Update counter
            i += 1
            # Yield a flight setting
            yield flight_setting

    def __check_export_list(self, export_list):
        """Check if export list is valid or if it is None. In case it is
        None, export all possible attributes.

        Parameters
        ----------
        export_list : list
            List of strings with the names of the attributes to be exported

        Returns
        -------
        export_list
        """

        if export_list:
            for attr in export_list:
                if not isinstance(attr, str):
                    raise TypeError("Variables must be strings.")

                # Checks if attribute is not valid
                if attr not in self.export_list:
                    raise ValueError(
                        "Attribute can not be exported. Check export_list."
                    )
        else:
            export_list = self.exportable_list

        return export_list

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
        flight_setting : dict
            The flight setting used in the simulation.
        flight : Flight
            The flight object.
        exec_time : float
            The execution time of the simulation.
        dispersion_input_file : str
            The name of the file containing all the inputs for the simulation.
        dispersion_output_file : str
            The name of the file containing all the outputs for the simulation.

        Returns
        -------
        None
        """
        # TODO: This method is called at every loop of the dispersion
        # so all the for loops are slowing down de dispersion
        # find a more efficient way to save attributes

        # TODO: Add a way to save more than just flight attributes, i.e. rocket, motor...

        # Get list of selected flight attributes
        attributes_list = list(set(dir(flight)).intersection(self.export_list))
        flight_result = {}
        for var in self.export_list:
            # First, capture the flight data that are saved in the flight object
            if var in attributes_list:
                # Check if Function. If so, get source
                if isinstance(getattr(flight, var), Function):
                    flight_result[str(var)] = getattr(flight, var).getSource()
                else:
                    flight_result[str(var)] = getattr(flight, var)

            # Second, capture data that needs to be calculated
            elif var == "executionTime":
                flight_result[str(var)] = exec_time
            elif var == "numberOfEvents":
                flight_result[str(var)] = len(flight.parachuteEvents)
            else:
                raise ValueError(f"Variable {var} could not be found.")

        # Take care of parachute results
        # TODO: this always gets saved... must be parametrized
        for trigger_time, parachute in flight.parachuteEvents:
            flight_result[parachute.name + "_triggerTime"] = trigger_time
            flight_result[parachute.name + "_inflatedTime"] = (
                trigger_time + parachute.lag
            )
            flight_result[parachute.name + "_inflatedVelocity"] = flight.speed(
                trigger_time + parachute.lag
            )

        # TODO: maybe we should not have any Function object in flight_setting,
        #       only have their source and make this check before the dispersion loop
        # Check if attr is Function. If so, get source
        for key, value in flight_setting.items():
            if isinstance(value, Function):
                flight_setting[key] = value.getSource()

        dispersion_input_file.write(str(flight_setting) + "\n")
        dispersion_output_file.write(str(flight_result) + "\n")

        return None

    def __export_flight_data_error(self, flight_setting, dispersion_error_file):
        """Saves flight error in a .txt

        Parameters
        ----------
        flight_setting : dict
            The flight setting used in the simulation.
        dispersion_error_file : str
            The name of the file containing all the errors for the simulation.

        Returns
        -------
        None
        """
        # Check if attr is Function. If so, get source
        for key, value in flight_setting.items():
            if isinstance(value, Function):
                flight_setting[key] = value.getSource()

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
        distribution_type="normal",
        export_list=None,
        append=False,
    ):
        """Runs the dispersion simulation and saves all data. For the simulation to be run
        all classes must be defined. This can happen either trough the dispersion_dictionary
        or by inputing objects

        Parameters
        ----------
        number_of_simulations : int
            Number of simulations to be run, must be non negative.
        dispersion_dictionary : dict
            The dictionary with the parameters to be analyzed. The keys must be the
            names of the attributes that will be used in the dispersion simulation.
            The values can either be a tuple, containing the nominal values of that
            parameter and its standard deviation, a list, containing the possible
            values to be randomly chosen in each simulation, or a single value (int
            or float), being the standard deviation of that parameter. See example
            for further explanations.
        environment : Environment, optional
            Environment object that will be used in the simulations. Default is None.
            If none, environment must be defined via passing its attributes in the
            dispersion_dictionary. Arguments related to environment will only vary
            according to the distribution method if the standard deviation for the
            desired attributes are on the dispersion_dictionary.
        flight : Flight, optional
            Flight object that will be used in the simulations. Default is None.
            If none, Flight must be defined via passing its attributes in the
            dispersion_dictionary. Arguments related to Flight will only vary
            according to the distribution method if the standard deviation for the
            desired attributes are on the dispersion_dictionary.
        motor : Motor, optional
            Motor object that will be used in the simulations. Default is None.
            If none, Motor must be defined via passing its attributes in the
            dispersion_dictionary. Arguments related to Motor will only vary
            according to the distribution method if the standard deviation for the
            desired attributes are on the dispersion_dictionary.
        rocket : Rocket, optional
            Rocket object that will be used in the simulations. Default is None.
            If none, Rocket must be defined via passing its attributes in the
            dispersion_dictionary. Arguments related to Rocket will only vary
            according to the distribution method if the standard deviation for the
            desired attributes are on the dispersion_dictionary.
        distribution_type : str, optional
            The probability distribution function to be used in the analysis,
            by default "normal". Options are any numpy.ramdom distributions
        export_list : list, optional
            A list containing the name of the attributes to be saved on the dispersion
            outputs file. See Examples for all possible attribues
        append : bool, optional
            If True, the results will be appended to the existing files. If False,
            the files will be overwritten. By default False.

        Returns
        -------
        None

        Examples
        --------

        TODO: add list of all possible attributes in dispersion_dictionaries and
              all possible attributes in export_list

        """

        # Saving the arguments as attributes
        self.number_of_simulations = number_of_simulations
        self.dispersion_dictionary = dispersion_dictionary
        if flight:  # In case a flight object is passed
            self.environment = flight.env
            self.motor = flight.rocket.motor
            self.rocket = flight.rocket
            self.flight = flight
        if rocket:
            self.rocket = rocket
            self.motor = rocket.motor
        if motor:
            self.motor = motor
        if environment:
            self.environment = environment

        self.distribution_type = distribution_type

        # Check if there's enough object to start a flight:
        ## Raise an error in case of any troubles
        self.__create_initial_objects()

        # Creates copy of dispersion_dictionary that will be altered
        modified_dispersion_dict = {i: j for i, j in dispersion_dictionary.items()}

        analysis_parameters = self.__process_dispersion_dict(modified_dispersion_dict)

        # TODO: This should be more flexible, allow different distributions for different parameters
        self.distributionFunc = self.__set_distribution_function(self.distribution_type)

        # Create data files for inputs, outputs and error logging
        open_mode = "a" if append else "w"
        dispersion_error_file = open(f"{self.filename}.disp_errors.txt", open_mode)
        dispersion_input_file = open(f"{self.filename}.disp_inputs.txt", open_mode)
        dispersion_output_file = open(f"{self.filename}.disp_outputs.txt", open_mode)

        # Checks export_list
        self.export_list = self.__check_export_list(export_list)

        # Creates a copy of the environment
        env_dispersion = self.environment

        # Creates copy of motor
        motor_dispersion = self.motor

        # Creates copy of rocket
        rocket_dispersion = self.rocket

        # Initialize counter and timer
        i = 0
        initial_wall_time = time()
        initial_cpu_time = process_time()

        # Begin display when running in notebook
        out = display("Starting", display_id=True)

        # Iterate over flight settings, start the flight simulations
        for setting in self.__yield_flight_setting(
            self.distributionFunc, analysis_parameters, self.number_of_simulations
        ):
            self.start_time = process_time()
            i += 1

            # TODO: resolve environment definitions here. Currently does not work
            #       because atmospheric model is not recalculated

            # Apply environment parameters variations on each iteration if possible
            env_dispersion.railLength = setting["railLength"]
            env_dispersion.gravity = setting["gravity"]
            env_dispersion.date = setting["date"]
            env_dispersion.latitude = setting["latitude"]
            env_dispersion.longitude = setting["longitude"]
            env_dispersion.elevation = setting["elevation"]
            if env_dispersion.atmosphericModelType in ["Ensemble", "Reanalysis"]:
                env_dispersion.selectEnsembleMember(setting["ensembleMember"])

            # Apply motor parameters variations on each iteration if possible
            # TODO: add hybrid and liquid motor option
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
                grainsCenterOfMassPosition=setting["grainsCenterOfMassPosition"],
                nozzlePosition=setting["nozzlePosition"],
            )

            # Apply rocket parameters variations on each iteration if possible
            rocket_dispersion = Rocket(
                mass=setting["mass"],
                inertiaI=setting["inertiaI"],
                inertiaZ=setting["inertiaZ"],
                radius=setting["radius"],
                powerOffDrag=setting["powerOffDrag"],
                powerOnDrag=setting["powerOnDrag"],
                centerOfDryMassPosition=setting["centerOfDryMassPosition"],
            )

            # Edit rocket drag
            rocket_dispersion.powerOffDrag *= setting["powerOffDragFactor"]
            rocket_dispersion.powerOnDrag *= setting["powerOnDragFactor"]

            # Add Motor
            rocket_dispersion.addMotor(
                motor_dispersion, position=setting["motorPosition"]
            )

            # Add rocket nose, fins and tail
            # Nose
            for nose in self.nose_names:
                rocket_dispersion.addNose(
                    length=setting[f"nose_{nose}_length"],
                    kind=setting[f"nose_{nose}_kind"],
                    position=setting[f"nose_{nose}_position"],
                    name=nose,
                )

            # Fins
            for finSet in self.finSet_names:
                # TODO: Allow elliptical fins
                rocket_dispersion.addTrapezoidalFins(
                    n=setting[f"finSet_{finSet}_n"],
                    rootChord=setting[f"finSet_{finSet}_rootChord"],
                    tipChord=setting[f"finSet_{finSet}_tipChord"],
                    span=setting[f"finSet_{finSet}_span"],
                    position=setting[f"finSet_{finSet}_position"],
                    airfoil=setting[f"finSet_{finSet}_airfoil"],
                    name=finSet,
                )

            # Tail
            for tail in self.tail_names:
                rocket_dispersion.addTail(
                    topRadius=setting[f"tail_{tail}_topRadius"],
                    bottomRadius=setting[f"tail_{tail}_bottomRadius"],
                    length=setting[f"tail_{tail}_length"],
                    position=setting[f"tail_{tail}_position"],
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

            rocket_dispersion.setRailButtons(
                position=[
                    setting["positionFirstRailButton"],
                    setting["positionSecondRailButton"],
                ],
                angularPosition=setting["railButtonAngularPosition"],
            )

            # Run trajectory simulation
            try:
                # TODO: Add initialSolution flight option
                dispersion_flight = Flight(
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
                    flight=dispersion_flight,
                    exec_time=process_time() - self.start_time,
                    dispersion_input_file=dispersion_input_file,
                    dispersion_output_file=dispersion_output_file,
                )
            except Exception as E:
                print(E)
                print(traceback.format_exc())
                self.__export_flight_data_error(setting, dispersion_error_file)

            # Register time
            # checks if out was defined
            # out only gets defined when using a jupyter notebook
            if out:
                out.update(
                    f"Current iteration: {i:06d} | Average Time per Iteration: "
                    f"{(process_time() - initial_cpu_time)/i:2.6f} s | Estimated time"
                    f" left: {int((number_of_simulations - i)*((process_time() - initial_cpu_time)/i))} s"
                )

        # Clean the house once all the simulations were already done

        ## Print and save total time
        final_string = (
            f"Completed {i} iterations successfully. Total CPU time: "
            f"{process_time() - initial_cpu_time} s. Total wall time: "
            f"{time() - initial_wall_time} s"
        )
        if out:
            out.update(final_string)
        dispersion_input_file.write(final_string + "\n")
        dispersion_output_file.write(final_string + "\n")
        dispersion_error_file.write(final_string + "\n")

        ## Close files
        dispersion_input_file.close()
        dispersion_output_file.close()
        dispersion_error_file.close()

        return None

    def import_results(self, variables=None):
        """Import dispersion results from .txt file and save it into a dictionary.

        Parameters
        ----------
        variables : list of str, optional
            List of variables to be imported. If None, all variables will be imported.

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

        # Process the results and save them as attributes of the class
        self.__process_results(variables=variables)

        return None

    # Start the processing analysis

    def __process_results(self, variables=None):
        """Save the mean and standard deviation of each parameter available
        in the results dictionary. Create class attributes for each parameter.

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
        """Plot the results of the dispersion analysis.

        Parameters
        ----------
        variables : list, optional
            List of variables to be plotted. If None, all variables will be
            plotted. The default is None. Example: ['outOfRailTime', 'apogee']

        Returns
        -------
        None
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

    def __createEllipses(self, dispersion_results):
        """A function to create apogee and impact ellipses from the dispersion
        results.

        Parameters
        ----------
        dispersion_results : dict
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
            apogeeX = np.array(dispersion_results["apogeeX"])
            apogeeY = np.array(dispersion_results["apogeeY"])
        except KeyError:
            print("No apogee data found.")
            apogeeX = np.array([])
            apogeeY = np.array([])
        try:
            impactX = np.array(dispersion_results["xImpact"])
            impactY = np.array(dispersion_results["yImpact"])
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
        perimeterSize : int, optional
            The size of the perimeter to be plotted. The default is 3000.
        xlim : tuple, optional
            The limits of the x axis. The default is (-3000, 3000).
        ylim : tuple, optional
            The limits of the y axis. The default is (-3000, 3000).

        Returns
        -------
        None
        """
        # Import background map
        if image is not None:
            img = imread(image)

        (
            impact_ellipses,
            apogee_ellipses,
            apogeeX,
            apogeeY,
            impactX,
            impactY,
        ) = self.__createEllipses(dispersion_results)

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
        plt.savefig(str(self.filename) + ".pdf", bbox_inches="tight", pad_inches=0)
        plt.savefig(str(self.filename) + ".svg", bbox_inches="tight", pad_inches=0)
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
        ) = self.__createEllipses(self.dispersion_results)
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
            # mult_ell.innerboundaryis = kml_data
            # mult_ell.outerboundaryis = kml_data
            mult_ell.style.linestyle.color = color
            mult_ell.style.linestyle.width = 3
            mult_ell.style.polystyle.color = simplekml.Color.changealphaint(
                100, simplekml.Color.blue
            )

        kml.save(filename)
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
        dispersion_results = self.dispersion_results

        print("Monte Carlo Simulation by RocketPy")
        print("Data Source: ", self.filename)
        print("Number of simulations: ", self.num_of_loaded_sims)
        print("Results: ")
        self.print_results()
        print("Plotting results: ")
        self.plotEllipses(dispersion_results=dispersion_results)
        self.plot_results()

        return None
