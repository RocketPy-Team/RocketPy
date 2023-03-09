__author__ = "Mateus Stano Junqueira, Sofia Lopes Suesdek Rocha, Guilherme Fernandes Alves, Bruno Abdulklech Sorban"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"


import math
import traceback
from time import process_time, time

import matplotlib.pyplot as plt
import numpy as np
import simplekml
from IPython.display import display
from matplotlib.patches import Ellipse
from numpy.random import *

from .AeroSurfaces import EllipticalFins, NoseCone, Tail, TrapezoidalFins
from .Environment import Environment
from .Flight import Flight
from .Function import Function
from .Motor import SolidMotor
from .Rocket import Rocket
from .tools import _get_distribution, invertedHaversine


class Dispersion:

    """Monte Carlo analysis to predict probability distributions of the rocket's
    landing point, apogee and other relevant information.

    Parameters
    ----------
    filename : string
        The name of the file containing the data to be used in the analysis.
    environment : Environment or McEnvironment
        The environment in which the rocket will be launched.
    rocket : Rocket
        The rocket to be launched.
    flight : Flight
        The flight conditions of the rocket.

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
        environment: Environment or McEnvironment
            The environment in which the rocket will be launched.
        rocket: Rocket
            The rocket to be launched.
        flight: Flight
            The flight conditions of the rocket.

        Returns
        -------
        None
        """

        # Save and initialize parameters
        self.filename = filename.split(".")[0]

        self.environment = environment
        self.rocket = rocket
        self.flight = flight
        self.motors = rocket.motors
        self.parachutes = rocket.parachutes
        self.nosecones = rocket.nosecones
        self.fins = rocket.fins
        self.tails = rocket.tails

        # TODO: Initialize variables so they can be accessed by MATLAB

        return None

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
                    if isinstance(value, dict):
                        setting[class_name][key] = {}
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, tuple):
                                try:
                                    setting[class_name][key][sub_key] = sub_value[-1](
                                        sub_value[0], sub_value[1]
                                    )
                                except TypeError:
                                    # Got the sweepLength or sweepAngle
                                    setting[class_name][key][sub_key] = sub_value[0]
                            else:
                                # else is list
                                # setting[class_name][key][sub_key] = choice(sub_value)
                                # The choice() doesn't work when you have Functions
                                setting[class_name][key][sub_key] = sub_value[
                                    randint(0, len(sub_value) - 1)
                                    if len(sub_value) > 1
                                    else 0
                                ]
                    elif isinstance(value, tuple):
                        setting[class_name][key] = value[-1](value[0], value[1])
                    else:
                        # else is list
                        setting[class_name][key] = value[
                            randint(0, len(value) - 1) if len(value) > 1 else 0
                        ]

            yield setting

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
        setting,
        flight,
        exec_time,
        inputs_log,
        outputs_log,
        export_list,
        save_parachute_data=False,
    ):
        """Saves flight results in a .txt
        Parameters
        ----------
        setting : dict
            The flight setting used in the simulation.
        flight : Flight
            The flight object.
        exec_time : float
            The execution time of the simulation.
        inputs_log : str
            The name of the file containing all the inputs for the simulation.
        outputs_log : str
            The name of the file containing all the outputs for the simulation.
        save_parachute_data : bool, optional
            If True, saves the parachute data, by default False
        export_list : list or tuple, optional
            List of variables to be saved, by default None. If None, use a
            default list of variables.
        Returns
        -------
        inputs_log : str
            The new string with the inputs of the simulation setting.
        outputs_log : str
            The new string with the outputs of the simulation setting.
        """
        m = map(getattr, [flight] * len(export_list), export_list)
        results = dict(zip(export_list, m))
        results["executionTime"] = exec_time

        # Sometimes we want to skip the parachute data to save time
        if save_parachute_data:
            for trigger_time, parachute in flight.parachuteEvents:
                # TODO: These should be better implemented in Flight events, avoiding
                # making any calculations here
                results[parachute.name + "_triggerTime"] = trigger_time
                results[parachute.name + "_inflatedTime"] = trigger_time + parachute.lag
                results[parachute.name + "_inflatedVelocity"] = flight.speed(
                    trigger_time + parachute.lag
                )
                results[parachute.name + "_inflatedAltitude"] = (
                    flight.z(trigger_time + parachute.lag) - flight.env.elevation
                )

        # Remove the powerOffDrag item from setting
        setting.pop("powerOffDrag", None)
        setting.pop("powerOnDrag", None)
        setting.pop("date", None)
        setting.pop("thrust", None)
        # TODO: Find a way to pop the parachute trigger functions

        inputs_log += str(setting) + "\n"
        outputs_log += str(results) + "\n"

        return inputs_log, outputs_log

    def __export_flight_data_error(self, setting, errors_log):
        """Saves flight error in a .txt
        Parameters
        ----------
        setting : dict
            The flight setting used in the simulation.
        errors_log : str
            The name of the file containing all the errors for the simulation.
        Returns
        -------
        errors_log : str
            The new string with the flight setting error saved.
        """

        errors_log += str(setting) + "\n"

        return errors_log

    @staticmethod
    def build_dispersion_dict(
        env,
        rocket,
        motors,
        flight,
        nosecones=None,
        fins=None,
        tails=None,
        parachutes=None,
        buttons=None,
    ):
        """Creates a dictionary to be used in the dispersion analysis. As a
        static method, it can be used without instantiating a Dispersion object.

        Parameters
        ----------
        env : rocketpy.monte_carlo.McEnvironment or rocketpy.Environment
            An environment object to be used in the simulations. Only a single
            object is supported.
            # TODO: verify what happens if we use Environment instead of McEnvironment
        rocket : rocketpy.monte_carlo.McRocket or rocketpy.Rocket
            A rocket object to be used in the simulations. Only a single object
            is supported. The rocket object doesn't need to have a motor, aerodynamic
            surfaces or parachutes attached. Indeed, all those components will be
            ignored when creating the dictionary, since they can be defined in
            the other arguments of this method.
        motors : Motor, McMotor or list of McMotor
            A motor object to be used in the simulations. In case of having more
            than one motor, the user must pass a list of McMotor objects.
        flight : Flight or McFlight
            A flight object to be used in the simulations. Only a single object
            is supported.
        nosecones : McNoseCone or list of McNoseCone, optional
            A nosecone object to be used in the simulations. In case of having more
            than one nosecone, the user must pass a list of McNoseCone objects.
        fins : McTrapezoidalFins, McEllipticalFins or list of them, optional
            A fin object to be used in the simulations. In case of having more
            than one fin, the user must pass a list of McFin objects.
        tails : Tail, McTail or list of McTail, optional
            A tail object to be used in the simulations. In case of having more
            than one tail, the user must pass a list of McTail objects.
        parachutes : Parachute, McParachute or list of McParachute, optional
            A parachute object to be used in the simulations. In case of having more
            than one parachute, the user must pass a list of McParachute objects.
        buttons : McButton, optional
            A button object to be used in the simulations. Only a single object
            is supported.

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
                    "0": {
                        "CdS": (mean, std, np.random.func),
                        "lag": (mean, std, np.random.func),
                        ...
                    },
                    "1": {...}
                },
                "buttons": {},
            }

        Examples
        --------
        >>> mc_env = McEnvironment(...)
        >>> mc_motor = McSolidMotor(...)
        >>> mc_rocket = McRocket(...)
        >>> mc_nose = McNoseCone(...)
        >>> mc_fins1 = McTrapezoidalFins(...)
        >>> mc_fins2 = McTrapezoidalFins(...)
        >>> mc_flight = McFlight(...)
        >>> mc_dict = Dispersion.build_dispersion_dict(
        >>>     mc_env, mc_rocket, mc_motor, mc_flight, mc_nose, [mc_fins1, mc_fins2]
        >>> )

        """

        mc_dict = {}
        structure = {
            "environment": env,
            "rocket": rocket,
            "motors": motors,
            "flight": flight,
            "nosecones": nosecones,
            "fins": fins,
            "tails": tails,
            "parachutes": parachutes,
            "buttons": buttons,
        }

        for name, data in structure.items():
            if data is None:
                continue
            mc_dict[name] = {}
            # Lists are special, for instance: parachutes = [McParachute(), McParachute()]
            # therefore we need to iterate over the list
            if isinstance(data, list):
                for i, item in enumerate(data):
                    mc_dict[name][i] = {}
                    for field in item.__annotations__.keys():
                        value = getattr(item, field)
                        # Avoid undesired fields (private and name)
                        if field[0] == "_":
                            continue
                        mc_dict[name][i][field] = Dispersion._create_mc_tuple(value)
            else:
                for field in data.__annotations__.keys():
                    value = getattr(data, field)
                    if field[0] == "_":
                        continue
                    mc_dict[name][field] = Dispersion._create_mc_tuple(value)

        # pop any value that is None, even if it is a value of sub dictionary
        mc_dict = Dispersion._pop_none(mc_dict)

        return mc_dict

    @staticmethod
    def _create_mc_tuple(value):
        """Receives a value provided by a McXxxxx object and returns a tuple
        containing the mean, standard deviation and a np.random function to be
        used in the simulation. There are two possible cases:
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
                _get_distribution(value[-1])
                if isinstance(value[-1], str)
                else np.random.normal
            )
            return (value[0], value[1], dist_func)

    @staticmethod
    def _pop_none(dictionary):
        """Removes all the keys that have a value of None. This is useful
        when the user wants to run a simulation with a subset of the
        available parameters.

        Parameters
        ----------
        dictionary : dict
            The dictionary to be cleaned

        Returns
        -------
        dict
            The cleaned dictionary
        """
        for key, value in dictionary.copy().items():
            if value is None:
                dictionary.pop(key)
            elif isinstance(value, dict):
                Dispersion._pop_none(value)
        return dictionary

    def run_dispersion(
        self,
        number_of_simulations,
        export_list=None,
        append=False,
        save_parachute_data=False,
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
        self.dispersion_dictionary = self.build_dispersion_dict(
            env=self.environment,
            rocket=self.rocket,
            motors=self.motors,
            flight=self.flight,
            nosecones=self.nosecones,
            fins=self.fins,
            tails=self.tails,
            parachutes=self.parachutes,
            buttons=None,
        )

        # Create data strings for inputs, outputs and error logging
        open_mode = "a" if append else "w"
        if open_mode == "a":
            with open(
                f"{self.filename}.disp_errors.txt", open_mode, encoding="utf-8"
            ) as f:
                errors_log = f
            with open(
                f"{self.filename}.disp_inputs.txt", open_mode, encoding="utf-8"
            ) as f:
                inputs_log = f
            with open(
                f"{self.filename}.disp_outputs.txt", open_mode, encoding="utf-8"
            ) as f:
                outputs_log = f
        else:
            errors_log = inputs_log = outputs_log = str()

        # Use a default export list if none is provided
        if export_list is None:
            print("No export list provided, using default list instead.")
            export_list = self.standard_output
        else:
            # Checks export_list
            self.export_list = self.__check_export_list(export_list)

        # Initialize counter and timer
        i = 0
        initial_wall_time = time()
        initial_cpu_time = process_time()

        # Begin display when running in notebook
        out = display("Starting", display_id=True)

        env_dispersion = (
            self.environment
            if isinstance(self.environment, Environment)
            else self.environment.__dict__["environment"]
        )

        # Iterate over flight settings, start the flight simulations
        for setting in self.__yield_flight_setting(
            self.dispersion_dictionary, self.number_of_simulations
        ):
            start_time = process_time()
            i += 1

            # Apply environment parameters variations on each iteration if possible
            env_dispersion.railLength = setting["environment"]["railLength"]
            # if env_dispersion.atmosphericModelType in ["Ensemble", "Reanalysis"]:
            #     env_dispersion.selectEnsembleMember(setting["environment"]["ensembleMember"])
            # TODO: allow varying ensembleMember, or use windXFactor and windYFactor

            # Apply motor parameters variations on each iteration if possible
            # TODO: add hybrid and liquid motor option
            # TODO: only a single motor is supported. Future version should support more.
            # TODO: the burnOutTime should no longer be an arg, see issue #309
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
                if "sweepAngle" in setting["fins"][fin].keys():
                    # means that it is trapezoidal
                    rocket_dispersion.addTrapezoidalFins(
                        n=setting["fins"][fin]["n"],
                        rootChord=setting["fins"][fin]["rootChord"],
                        tipChord=setting["fins"][fin]["tipChord"],
                        span=setting["fins"][fin]["span"],
                        position=setting["fins"][fin]["position"],
                        cantAngle=setting["fins"][fin]["cantAngle"],
                        sweepLength=setting["fins"][fin]["sweepLength"],
                        sweepAngle=setting["fins"][fin]["sweepAngle"],
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

            # Add parachutes
            rocket_dispersion.parachutes = []  # Remove existing parachutes
            for chute in setting["parachutes"].keys():
                rocket_dispersion.addParachute(
                    name=chute,
                    CdS=setting["parachutes"][chute]["CdS"],
                    trigger=setting["parachutes"][chute]["trigger"],
                    samplingRate=setting["parachutes"][chute]["samplingRate"],
                    lag=setting["parachutes"][chute]["lag"],
                    noise=setting["parachutes"][chute]["noise"],
                )

            # TODO: Fix rail buttons definition
            # rocket_dispersion.setRailButtons()

            # Run trajectory simulation
            try:
                # TODO: Add initialSolution flight option
                dispersion_flight = Flight(
                    rocket=rocket_dispersion,
                    environment=env_dispersion,
                    inclination=setting["flight"]["inclination"],
                    heading=setting["flight"]["heading"],
                    # terminateOnApogee=setting["flight"]["terminateOnApogee"],
                    # maxTime=setting["flight"]["maxTime"],
                    # maxTimeStep=setting["flight"]["maxTimeStep"],
                    # minTimeStep=setting["flight"]["minTimeStep"],
                    # rtol=setting["flight"]["rtol"],
                    # atol=setting["flight"]["atol"],
                    # timeOvershoot=setting["flight"]["timeOvershoot"],
                    # verbose=setting["flight"]["verbose"],
                )

                inputs_log, outputs_log = self.__export_flight_data(
                    setting=setting,
                    flight=dispersion_flight,
                    exec_time=process_time() - start_time,
                    inputs_log=inputs_log,
                    outputs_log=outputs_log,
                    save_parachute_data=save_parachute_data,
                    export_list=export_list,
                )
            except (TypeError, ValueError, KeyError, AttributeError) as error:
                print(f"Error on iteration {i}: {error}")
                errors_log = self.__export_flight_data_error(setting, errors_log)
            except KeyboardInterrupt:
                print("Keyboard Interrupt, file saved.")
                errors_log = self.__export_flight_data_error(setting, errors_log)
                self.__save_logs(inputs_log, outputs_log, errors_log)
                break

            # Update progress bar. Only works on jupyter notebook
            if out:
                out.update(
                    f"Current iteration: {i:06d} | Average Time per Iteration: "
                    f"{(process_time() - initial_cpu_time)/i:2.6f} s | Estimated time"
                    f" left: {int((number_of_simulations - i)*((process_time() - initial_cpu_time)/i))} s"
                )

        # Clean the house once all the simulations were already done

        ## Print and save total time
        final_string = (
            f"Completed {i} iterations. Total CPU time: "
            f"{process_time() - initial_cpu_time:.1f} s. Total wall time: "
            f"{time() - initial_wall_time:.1f} s"
        )
        if out:
            out.update(final_string)
        inputs_log = inputs_log + final_string + "\n"
        outputs_log = outputs_log + final_string + "\n"
        errors_log = errors_log + final_string + "\n"
        self.__save_logs(inputs_log, outputs_log, errors_log)

        return None

    def __save_logs(self, inputs_log, outputs_log, errors_log):
        """Save logs to files.
        Parameters
        ----------
        inputs_log : str
            String containing all inputs.
        outputs_log : str
            String containing all outputs.
        errors_log : str
            String containing all errors.
        Returns
        -------
        None
        """
        # Save inputs
        with open(self.filename + ".disp_inputs.txt", "w", encoding="utf-8") as file:
            file.write(inputs_log)

        # Save outputs
        with open(self.filename + ".disp_outputs.txt", "w", encoding="utf-8") as file:
            file.write(outputs_log)

        # Save errors
        with open(self.filename + ".disp_errors.txt", "w", encoding="utf-8") as file:
            file.write(errors_log)

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
            for key, value in flight_result.items():
                if key not in dispersion_results.keys():
                    # Create a new list to store the parameter
                    dispersion_results[key] = [value]
                else:
                    # Append the parameter value to the list
                    dispersion_results[key].append(value)

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
            try:
                from imageio import imread

                img = imread(image)
            except ImportError:
                print(
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
