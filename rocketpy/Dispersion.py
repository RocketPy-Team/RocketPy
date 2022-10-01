# -*- coding: utf-8 -*-

__author__ = "Mateus Stano Junqueira, Sofia Lopes Suesdek Rocha"
__copyright__ = "Copyright 20XX, Projeto Jupiter"
__license__ = "MIT"


import traceback
from time import process_time, time
import warnings

import matplotlib.pyplot as plt
import numpy as np
from imageio import imread
from IPython.display import display
from matplotlib.patches import Ellipse
from numpy.random import *
from datetime import datetime

from .Environment import Environment
from .Flight import Flight
from .Function import Function
from .Motor import HybridMotor, SolidMotor
from .Rocket import Rocket


class Dispersion:

    """Monte Carlo analysis to predict probability distributions of the rocket's
    landing point, apogee and other relevant information.

    Attributes
    ----------
        Parameters:
        Dispersion.filename: string
            When running a new simulation, this attribute represents the initial
            part of the export filenames (e.g. 'filename.disp_outputs.txt').
            When analyzing the results of a previous simulation, this attribute
            shall be the filename containing the outputs of a dispersion calculation.
        Dispersion.image: string
            Launch site PNG file to be plotted along with the dispersion ellipses.
            Attribute needed to run a new simulation.
        Dispersion.realLandingPoint: tuple
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
            shall be the .txt filename containing the outputs of a dispersion calculation.
        number_of_simulations: integer, needed when running a new simulation
            Number of simulations desired, must be greater than zero.
            Default is zero.
        flight: Flight
            Original rocket's flight with nominal values.
            Parameter needed to run a new simulation, when environment,
            motor and rocket remain unchanged.
            Default is None.
        image: string, needed when running a new simulation
            Launch site PNG file to be plotted along with the dispersion ellipses.
        dispersionDict: dictionary, optional
            Contains the information of which environment, motor, rocket and flight variables
            will vary according to its standard deviation.
            Format {'parameter0': (nominal value, standard deviation), 'parameter1':
            (nominal value, standard deviation), ...}
            (e.g. {'rocketMass':(20, 0.2),
            'burnOut': (3.9, 0.3), 'railLength': (5.2, 0.05)})
            Default is {}.
        environment: Environment
            Launch environment.
            Parameter needed to run a new simulation, when Dispersion.flight remains unchanged.
            Default is None.
        motor: Motor, optional
            Rocket's motor.
            Parameter needed to run a new simulation, when Dispersion.flight remains unchanged.
            Default is None.
        rocket: Rocket, optional
            Rocket with nominal values.
            Parameter needed to run a new simulation, when Dispersion.flight remains unchanged.
            Default is None.
        distributionType: string, optional
            Determines which type of distribution will be applied to variable parameters and
            its respective standard deviation.
            Default is 'normal'
        realLandingPoint: tuple, optional
            Rocket's experimental landing point relative to launch point.
            Format (horizontal distance, vertical distance)
        Returns
        -------
        None
        """

        # Save  and initialize parameters
        self.filename = filename
        self.number_of_simulations = 0
        self.flight = None
        self.dispersionDict = {}
        self.environment = None
        self.motor = None
        self.rocket = None
        self.distributionType = "normal"
        self.image = None
        self.realLandingPoint = None
        self.parachuteTriggers = []

    def classCheck(self):
        rocketAttributes = []
        rocketInputs = []

    def setDistributionFunc(self, distributionType):
        if distributionType == "normal" or distributionType == None:
            return normal
        elif distributionType == "beta":
            return beta
        elif distributionType == "binomial":
            return binomial
        elif distributionType == "chisquare":
            return chisquare
        elif distributionType == "dirichlet":
            return dirichlet
        elif distributionType == "exponential":
            return exponential
        elif distributionType == "f":
            return f
        elif distributionType == "gamma":
            return gamma
        elif distributionType == "geometric":
            return geometric
        elif distributionType == "gumbel":
            return gumbel
        elif distributionType == "hypergeometric":
            return hypergeometric
        elif distributionType == "laplace":
            return laplace
        elif distributionType == "logistic":
            return logistic
        elif distributionType == "lognormal":
            return lognormal
        elif distributionType == "logseries":
            return logseries
        elif distributionType == "multinomial":
            return multinomial
        elif distributionType == "multivariate_normal":
            return multivariate_normal
        elif distributionType == "negative_binomial":
            return negative_binomial
        elif distributionType == "noncentral_chisquare":
            return noncentral_chisquare
        elif distributionType == "noncentral_f":
            return noncentral_f
        elif distributionType == "pareto":
            return pareto
        elif distributionType == "poisson":
            return poisson
        elif distributionType == "power":
            return power
        elif distributionType == "rayleigh":
            return rayleigh
        elif distributionType == "standard_cauchy":
            return standard_cauchy
        elif distributionType == "standard_exponential":
            return standard_exponential
        elif distributionType == "standard_gamma":
            return standard_gamma
        elif distributionType == "standard_normal":
            return standard_normal
        elif distributionType == "standard_t":
            return standard_t
        elif distributionType == "triangular":
            return triangular
        elif distributionType == "uneliform":
            return uniform
        elif distributionType == "vonmises":
            return vonmises
        elif distributionType == "wald":
            return wald
        elif distributionType == "weibull":
            return weibull
        elif distributionType == "zipf":
            return zipf
        else:
            warnings.warn("Distribution type not supported")

    def processDispersionDict(self, dispersionDict):
        # Get parachutes names
        analysis_parameters = {}
        if "parachuteNames" in dispersionDict:  # TODO: use only dispersionDict
            for i, name in enumerate(dispersionDict["parachuteNames"]):
                if "CdS" in dispersionDict:
                    analysis_parameters["parachute_" + name + "_CdS"] = dispersionDict[
                        "CdS"
                    ][i]
                if "trigger" in dispersionDict:
                    analysis_parameters[
                        "parachute_" + name + "_trigger"
                    ] = dispersionDict["trigger"][i]
                if "samplingRate" in dispersionDict:
                    analysis_parameters[
                        "parachute_" + name + "_samplingRate"
                    ] = dispersionDict["samplingRate"][i]
                if "lag" in dispersionDict:
                    analysis_parameters["parachute_" + name + "_lag"] = dispersionDict[
                        "lag"
                    ][i]
                if "noise_mean" in dispersionDict:
                    analysis_parameters[
                        "parachute_" + name + "_noise_mean"
                    ] = dispersionDict["noise_mean"][i]
                if "noise_sd" in dispersionDict:
                    analysis_parameters[
                        "parachute_" + name + "_noise_std"
                    ] = dispersionDict["noise_sd"][i]
                if "noise_corr" in dispersionDict:
                    analysis_parameters[
                        "parachute_" + name + "_noise_corr"
                    ] = dispersionDict["noise_corr"][i]
            dispersionDict.pop("CdS", None)
            dispersionDict.pop("trigger", None)
            dispersionDict.pop("samplingRate", None)
            dispersionDict.pop("lag", None)
            dispersionDict.pop("noise_mean", None)
            dispersionDict.pop("noise_sd", None)
            dispersionDict.pop("noise_corr", None)
            self.parachute_names = dispersionDict.pop("parachuteNames", None)

        for key, value in dispersionDict.items():
            analysis_parameters[key] = value

        return analysis_parameters

    def yield_flight_setting(
        self, distributionFunc, analysis_parameters, number_of_simulations
    ):

        """Yields a flight setting for the simulation"""

        i = 0
        while i < number_of_simulations:
            # Generate a flight setting
            flight_setting = {}
            for parameter_key, parameter_value in analysis_parameters.items():
                if type(parameter_value) is tuple:
                    flight_setting[parameter_key] = distributionFunc(*parameter_value)
                elif type(parameter_value) is list:
                    flight_setting[parameter_key] = choice(parameter_value)
                else:  # if parameter_value is only the standard deviation
                    if "parachute" in parameter_key:
                        _, parachute_name, parameter = parameter_key.split("_")
                        flight_setting[parameter_key] = distributionFunc(
                            getattr(
                                self.rocket.parachutes[
                                    self.parachute_names.index(parachute_name)
                                ],
                                parameter,
                            ),
                            parameter_value,
                        )
                    else:
                        flight_setting[parameter_key] = distributionFunc(
                            getattr(self.rocket, parameter_key), parameter_value
                        )  # TODO: BUG: accept more then just rocket class
                        # create a list with the strings of possible inputs and
                        # check in that string to find to which class it belongs

            # Update counter
            i += 1
            # Yield a flight setting
            yield flight_setting

    # TODO: Rework post process Flight method making it possible (and optmized) to
    # chose what is going to be exported
    def export_flight_data(
        self,
        flight_setting,
        flight_data,
        exec_time,
        dispersion_input_file,
        dispersion_output_file,
    ):

        """Saves flight results in a .txt"""

        # Generate flight results
        flight_result = {
            "outOfRailTime": flight_data.outOfRailTime,
            "outOfRailVelocity": flight_data.outOfRailVelocity,
            "apogeeTime": flight_data.apogeeTime,
            "apogeeAltitude": flight_data.apogee - flight_data.env.elevation,
            "apogeeX": flight_data.apogeeX,
            "apogeeY": flight_data.apogeeY,
            "impactTime": flight_data.tFinal,
            "impactX": flight_data.xImpact,
            "impactY": flight_data.yImpact,
            "impactVelocity": flight_data.impactVelocity,
            "initialStaticMargin": flight_data.rocket.staticMargin(0),
            "outOfRailStaticMargin": flight_data.rocket.staticMargin(
                flight_data.outOfRailTime
            ),
            "finalStaticMargin": flight_data.rocket.staticMargin(
                flight_data.rocket.motor.burnOutTime
            ),
            "numberOfEvents": len(flight_data.parachuteEvents),
            "drogueTriggerTime": [],
            "drogueInflatedTime": [],
            "drogueInflatedVelocity": [],
            "executionTime": exec_time,
            "lateralWind": flight_data.lateralSurfaceWind,
            "frontalWind": flight_data.frontalSurfaceWind,
        }

        # Calculate maximum reached velocity
        sol = np.array(flight_data.solution)
        flight_data.vx = Function(
            sol[:, [0, 4]],
            "Time (s)",
            "Vx (m/s)",
            "linear",
            extrapolation="natural",
        )
        flight_data.vy = Function(
            sol[:, [0, 5]],
            "Time (s)",
            "Vy (m/s)",
            "linear",
            extrapolation="natural",
        )
        flight_data.vz = Function(
            sol[:, [0, 6]],
            "Time (s)",
            "Vz (m/s)",
            "linear",
            extrapolation="natural",
        )
        flight_data.v = (
            flight_data.vx**2 + flight_data.vy**2 + flight_data.vz**2
        ) ** 0.5
        flight_data.maxVel = np.amax(flight_data.v.source[:, 1])
        flight_result["maxVelocity"] = flight_data.maxVel

        # Take care of parachute results
        for trigger_time, parachute in flight_data.parachuteEvents:
            flight_result[parachute.name + "_triggerTime"] = trigger_time
            flight_result[parachute.name + "_inflatedTime"] = (
                trigger_time + parachute.lag
            )
            flight_result[parachute.name + "_inflatedVelocity"] = flight_data.v(
                trigger_time + parachute.lag
            )
        else:
            flight_result["parachuteInfo"] = "No Parachute Events"

        # Write flight setting and results to file
        dispersion_input_file.write(str(flight_setting) + "\n")
        dispersion_output_file.write(str(flight_result) + "\n")

        return None

    def export_flight_error(self, flight_setting, dispersion_error_file):

        """Saves flight error in a .txt"""

        dispersion_error_file.write(str(flight_setting) + "\n")

        return None

    def runDispersion(
        self,
        number_of_simulations,
        dispersionDict,
        environment,
        flight=None,
        motor=None,
        rocket=None,
        distributionType="normal",
        image=None,
        realLandingPoint=None,
    ):

        """Runs the given number of simulations and saves the data"""

        self.number_of_simulations = number_of_simulations
        self.dispersionDict = dispersionDict
        self.environment = environment
        self.flight = flight
        if flight:
            self.motor = flight.rocket.motor if not motor else motor
            self.rocket = flight.rocket if not rocket else rocket
        self.motor = motor if motor else self.motor
        self.rocket = rocket if rocket else self.rocket
        self.distributionType = distributionType
        self.image = image
        self.realLandingPoint = realLandingPoint

        # Creates copy of dispersionDict that will be altered
        modified_dispersion_dict = {i: j for i, j in dispersionDict.items()}
        analysis_parameters = self.processDispersionDict(modified_dispersion_dict)

        self.distribuitionFunc = self.setDistributionFunc(distributionType)
        # Basic analysis info

        # Create data files for inputs, outputs and error logging
        dispersion_error_file = open(str(self.filename) + ".disp_errors.txt", "w")
        dispersion_input_file = open(str(self.filename) + ".disp_inputs.txt", "w")
        dispersion_output_file = open(str(self.filename) + ".disp_outputs.txt", "w")

        # # Initialize Environment
        # customAtmosphere = False
        # if not self.environment:
        #     self.environment = Environment(
        #         railLength=0,
        #     )
        #     if "envAtmosphericType" in dispersionDict:
        #         if dispersionDict["envAtmosphericType"] == "CustomAtmosphere":
        #             customAtmosphere = True
        #         self.environment.setDate(datetime(*dispersionDict["date"][0]))
        #         self.environment.setAtmosphericModel(
        #             type=dispersionDict["envAtmosphericType"],
        #             file=dispersionDict["envAtmosphericFile"]
        #             if "envAtmosphericFile" in dispersionDict
        #             else None,
        #             dictionary=dispersionDict["envAtmosphericDictionary"]
        #             if "envAtmosphericDictionary" in dispersionDict
        #             else None,
        #         )

        # Initialize counter and timer
        i = 0

        initial_wall_time = time()
        initial_cpu_time = process_time()

        # Iterate over flight settings
        out = display("Starting", display_id=True)
        for setting in self.yield_flight_setting(
            self.distribuitionFunc, analysis_parameters, self.number_of_simulations
        ):
            start_time = process_time()
            i += 1

            # Creates an of environment
            envDispersion = self.environment

            # Apply environment parameters variations on each iteration if possible
            envDispersion.railLength = (
                setting["railLength"] if "railLength" in setting else envDispersion.rL
            )
            envDispersion.gravity = (
                setting["gravity"] if "gravity" in setting else envDispersion.g
            )
            envDispersion.date = (
                setting["date"] if "date" in setting else envDispersion.date
            )
            envDispersion.latitude = (
                setting["latitude"] if "latitude" in setting else envDispersion.lat
            )
            envDispersion.longitude = (
                setting["longitude"] if "longitude" in setting else envDispersion.lon
            )
            envDispersion.elevation = (
                setting["elevation"]
                if "elevation" in setting
                else envDispersion.elevation
            )
            envDispersion.datum = (
                setting["datum"] if "datum" in setting else envDispersion.datum
            )
            envDispersion.timeZone = (
                dispersionDict["timeZone"] if "timeZone" in dispersionDict else "UTC"
            )
            if "ensembleMember" in setting:
                envDispersion.selectEnsembleMember(setting["ensembleMember"])

            # Creates copy of motor
            motorDispersion = self.motor

            # Apply motor parameters variations on each iteration if possible
            # TODO: add hybrid motor option
            motorDispersion = SolidMotor(
                thrustSource=setting["thrustSource"]
                if "thrustSource" in setting
                else self.motor.thrust,
                burnOut=setting["burnOut"]
                if "burnOut" in setting
                else self.motor.burnOutTime,
                grainNumber=setting["grainNumber"]
                if "grainNumber" in setting
                else self.motor.grainNumber,
                grainDensity=setting["grainDensity"]
                if "grainDensity" in setting
                else self.motor.grainDensity,
                grainOuterRadius=setting["grainOuterRadius"]
                if "grainOuterRadius" in setting
                else self.motor.grainOuterRadius,
                grainInitialInnerRadius=setting["grainInitialInnerRadius"]
                if "grainInitialInnerRadius" in setting
                else self.motor.grainInitialInnerRadius,
                grainInitialHeight=setting["grainInitialHeight"]
                if "grainInitialHeight" in setting
                else self.motor.grainInitialHeight,
                grainSeparation=setting["grainSeparation"]
                if "grainSeparation" in setting
                else self.motor.grainSeparation,
                nozzleRadius=setting["nozzleRadius"]
                if "nozzleRadius" in setting
                else self.motor.nozzleRadius,
                throatRadius=setting["throatRadius"]
                if "throatRadius" in setting
                else self.motor.throatRadius,
                reshapeThrustCurve=(setting["burnOut"], setting["impulse"])
                if "burnOut" and "impulse" in setting
                else False,
                interpolationMethod=setting["interpolationMethod"]
                if "interpolationMethod" in setting
                else self.motor.interpolate,
            )

            # Creates copy of rocket
            rocketDispersion = self.rocket

            # Apply rocket parameters variations on each iteration if possible
            rocketDispersion = Rocket(
                motor=motorDispersion,
                mass=setting["rocketMass"]
                if "rocketMass" in setting
                else self.rocket.mass,
                inertiaI=setting["inertiaI"]
                if "inertiaI" in setting
                else self.rocket.inertiaI,
                inertiaZ=setting["inertiaZ"]
                if "inertiaZ" in setting
                else self.rocket.inertiaZ,
                radius=setting["radius"] if "radius" in setting else self.rocket.radius,
                distanceRocketNozzle=setting["distanceRocketNozzle"]
                if "distanceRocketNozzle" in setting
                else self.rocket.distanceRocketNozzle,
                distanceRocketPropellant=setting["distanceRocketPropellant"]
                if "distanceRocketPropellant" in setting
                else self.rocket.distanceRocketPropellant,
                powerOffDrag=setting["powerOffDrag"]
                if "powerOffDrag" in setting
                else self.rocket.powerOffDrag,
                powerOnDrag=setting["powerOnDrag"]
                if "powerOnDrag" in setting
                else self.rocket.powerOnDrag,
            )

            # Add rocket nose, fins and tail
            rocketDispersion.addNose(
                length=setting["noseLength"]
                if "noseLength" in setting
                else self.rocket.noseLength,
                kind=setting["noseKind"]
                if "noseKind" in setting
                else self.rocket.noseKind,
                distanceToCM=setting["noseDistanceToCM"]
                if "noseDistanceToCM" in setting
                else self.rocket.noseDistanceToCM,
            )
            rocketDispersion.addFins(
                n=setting["numberOfFins"]
                if "numberOfFins" in setting
                else self.rocket.numberOfFins,
                rootChord=setting["rootChord"]
                if "rootChord" in setting
                else self.rocket.rootChord,
                tipChord=setting["tipChord"]
                if "tipChord" in setting
                else self.rocket.tipChord,
                span=setting["span"] if "span" in setting else self.rocket.span,
                distanceToCM=setting["distanceToCM"]
                if "distanceToCM" in setting
                else self.rocket.distanceRocketFins,
                radius=setting["radius"]
                if "radius" in setting
                else self.rocket.finRadius,
                airfoil=setting["airfoil"]
                if "airfoil" in setting
                else self.rocket.finAirfoil,
            )
            if not "noTail" in setting:
                rocketDispersion.addTail(
                    topRadius=setting["topRadius"]
                    if "topRadius" in setting
                    else self.rocket.tailTopRadius,
                    bottomRadius=setting["bottomRadius"]
                    if "bottomRadius" in setting
                    else self.rocket.tailBottomRadius,
                    length=setting["length"]
                    if "length" in setting
                    else self.rocket.tailLength,
                    distanceToCM=setting["distanceToCM"]
                    if "distanceToCM" in setting
                    else self.rocket.tailDistanceToCM,
                )

            # Add parachutes
            for num, name in enumerate(self.parachute_names):
                rocketDispersion.addParachute(
                    name=name,
                    CdS=setting["parachute_" + name + "_CdS"]
                    if "parachute_" + name + "_CdS" in setting
                    else self.rocket.parachutes[num].CdS,
                    trigger=setting["parachute_" + name + "_trigger"]
                    if "parachute_" + name + "_trigger" in setting
                    else self.rocket.parachutes[num].trigger,
                    samplingRate=setting["parachute_" + name + "_samplingRate"]
                    if "parachute_" + name + "_samplingRate" in setting
                    else self.rocket.parachutes[num].samplingRate,
                    lag=setting["parachute_" + name + "_lag"]
                    if "parachute_" + name + "_lag" in setting
                    else self.rocket.parachutes[num].lag,
                    noise=(
                        setting["parachute_" + name + "_noise_mean"]
                        if "parachute_" + name + "_noise_mean" in setting
                        else self.rocket.parachutes[num].noise[0],
                        setting["parachute_" + name + "_noise_std"]
                        if "parachute_" + name + "_noise_std" in setting
                        else self.rocket.parachutes[num].noise[1],
                        setting["parachute_" + name + "_noise_corr"]
                        if "parachute_" + name + "_noise_corr" in setting
                        else self.rocket.parachutes[num].noise[2],
                    ),
                )

            rocketDispersion.setRailButtons(
                distanceToCM=[
                    setting["positionFirstRailButton"],
                    setting["positionSecondRailButton"],
                ]
                if "positionFirstRailButton" or "positionSecondRailButton" in setting
                else self.rocket.RBdistanceToCM,
                angularPosition=setting["railButtonAngularPosition"]
                if "railButtonAngularPosition" in setting
                else self.rocket.angularPosition,
            )

            # Run trajectory simulation
            try:
                TestFlight = Flight(
                    rocket=rocketDispersion,
                    environment=envDispersion,
                    inclination=setting["inclination"]
                    if "inclination" in setting
                    else self.flight.inclination,
                    heading=setting["heading"]
                    if "heading" in setting
                    else self.flight.heading,
                    # initialSolution=setting["initialSolution"] if "initialSolution" in setting else self.flight.initialSolution,
                    terminateOnApogee=setting["terminateOnApogee"]
                    if "terminateOnApogee" in setting
                    else False,
                    maxTime=setting["maxTime"] if "maxTime" in setting else 600,
                    maxTimeStep=setting["maxTimeStep"]
                    if "maxTimeStep" in setting
                    else np.inf,
                    minTimeStep=setting["minTimeStep"]
                    if "minTimeStep" in setting
                    else 0,
                    rtol=setting["rtol"] if "rtol" in setting else 1e-6,
                    atol=setting["atol"]
                    if "atol" in setting
                    else 6 * [1e-3] + 4 * [1e-6] + 3 * [1e-3],
                    timeOvershoot=setting["timeOvershoot"]
                    if "timeOvershoot" in setting
                    else True,
                    verbose=setting["verbose"] if "verbose" in setting else False,
                )

                self.export_flight_data(
                    setting,
                    TestFlight,
                    process_time() - start_time,
                    dispersion_input_file,
                    dispersion_output_file,
                )
            except Exception as E:
                print(E)
                print(traceback.format_exc())
                self.export_flight_error(setting, dispersion_error_file)

            # Register time
            out.update(
                f"Curent iteration: {i:06d} | Average Time per Iteration: {(process_time() - initial_cpu_time)/i:2.6f} s | Estimated time left: {int((number_of_simulations - i)*((process_time() - initial_cpu_time)/i))} s"
            )

        # Done

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

    def importResults(self, dispersion_output_file):

        """Import dispersion results from .txt file"""

        # Initialize variable to store all results
        dispersion_general_results = []

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
        # Get file
        dispersion_output_file = open(dispersion_output_file, "r+")

        # Read each line of the file and convert to dict
        for line in dispersion_output_file:
            # Skip comments lines
            if line[0] != "{":
                continue
            # Eval results and store them
            flight_result = eval(line)
            dispersion_general_results.append(flight_result)
            for parameter_key, parameter_value in flight_result.items():
                dispersion_results[parameter_key].append(parameter_value)

        # Close data file
        dispersion_output_file.close()

        # Number of flights simulated
        self.N = len(dispersion_general_results)

        return dispersion_results

    def meanOutOfRailTime(self, dispersion_results):
        print(
            f'Out of Rail Time -         Mean Value: {np.mean(dispersion_results["outOfRailTime"]):0.3f} s'
        )
        print(
            f'Out of Rail Time - Standard Deviation: {np.std(dispersion_results["outOfRailTime"]):0.3f} s'
        )

        return None

    def plotOutOfRailTime(self, dispersion_results):

        self.meanOutOfRailTime(dispersion_results)

        plt.figure()
        plt.hist(dispersion_results["outOfRailTime"], bins=int(self.N**0.5))
        plt.title("Out of Rail Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Number of Occurences")
        plt.show()

        return None

    def meanOutOfRailVelocity(self, dispersion_results):
        print(
            f'Out of Rail Velocity -         Mean Value: {np.mean(dispersion_results["outOfRailVelocity"]):0.3f} m/s'
        )
        print(
            f'Out of Rail Velocity - Standard Deviation: {np.std(dispersion_results["outOfRailVelocity"]):0.3f} m/s'
        )

        return None

    def plotOutOfRailVelocity(self, dispersion_results):

        self.meanOutOfRailVelocity(dispersion_results)

        plt.figure()
        plt.hist(dispersion_results["outOfRailVelocity"], bins=int(self.N**0.5))
        plt.title("Out of Rail Velocity")
        plt.xlabel("Velocity (m/s)")
        plt.ylabel("Number of Occurences")
        plt.show()

        return None

    def meanApogeeTime(self, dispersion_results):
        print(
            f'Impact Time -         Mean Value: {np.mean(dispersion_results["impactTime"]):0.3f} s'
        )
        print(
            f'Impact Time - Standard Deviation: {np.std(dispersion_results["impactTime"]):0.3f} s'
        )

        return None

    def plotApogeeTime(self, dispersion_results):

        self.meanApogeeTime(dispersion_results)

        plt.figure()
        plt.hist(dispersion_results["impactTime"], bins=int(self.N**0.5))
        plt.title("Impact Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Number of Occurences")
        plt.show()

        return None

    def meanApogeeAltitude(self, dispersion_results):
        print(
            f'Apogee Altitude -         Mean Value: {np.mean(dispersion_results["apogeeAltitude"]):0.3f} m'
        )
        print(
            f'Apogee Altitude - Standard Deviation: {np.std(dispersion_results["apogeeAltitude"]):0.3f} m'
        )

        return None

    def plotApogeeAltitude(self, dispersion_results):

        self.meanApogeeAltitude(dispersion_results)

        plt.figure()
        plt.hist(dispersion_results["apogeeAltitude"], bins=int(self.N**0.5))
        plt.title("Apogee Altitude")
        plt.xlabel("Altitude (m)")
        plt.ylabel("Number of Occurences")
        plt.show()

        return None

    def meanApogeeXPosition(self, dispersion_results):
        print(
            f'Apogee X Position -         Mean Value: {np.mean(dispersion_results["apogeeX"]):0.3f} m'
        )
        print(
            f'Apogee X Position - Standard Deviation: {np.std(dispersion_results["apogeeX"]):0.3f} m'
        )

        return None

    def plotApogeeXPosition(self, dispersion_results):

        self.meanApogeeAltitude(dispersion_results)

        plt.figure()
        plt.hist(dispersion_results["apogeeX"], bins=int(self.N**0.5))
        plt.title("Apogee X Position")
        plt.xlabel("Apogee X Position (m)")
        plt.ylabel("Number of Occurences")
        plt.show()

        return None

    def meanApogeeYPosition(self, dispersion_results):
        print(
            f'Apogee Y Position -         Mean Value: {np.mean(dispersion_results["apogeeY"]):0.3f} m'
        )
        print(
            f'Apogee Y Position - Standard Deviation: {np.std(dispersion_results["apogeeY"]):0.3f} m'
        )

        return None

    def plotApogeeYPosition(self, dispersion_results):

        self.meanApogeeAltitude(dispersion_results)

        plt.figure()
        plt.hist(dispersion_results["apogeeY"], bins=int(self.N**0.5))
        plt.title("Apogee Y Position")
        plt.xlabel("Apogee Y Position (m)")
        plt.ylabel("Number of Occurences")
        plt.show()

        return None

    def meanImpactTime(self, dispersion_results):
        print(
            f'Impact Time -         Mean Value: {np.mean(dispersion_results["impactTime"]):0.3f} s'
        )
        print(
            f'Impact Time - Standard Deviation: {np.std(dispersion_results["impactTime"]):0.3f} s'
        )

        return None

    def plotImpactTime(self, dispersion_results):

        self.meanImpactTime(dispersion_results)

        plt.figure()
        plt.hist(dispersion_results["impactTime"], bins=int(self.N**0.5))
        plt.title("Impact Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Number of Occurences")
        plt.show()

        return None

    def meanImpactXPosition(self, dispersion_results):
        print(
            f'Impact X Position -         Mean Value: {np.mean(dispersion_results["impactX"]):0.3f} m'
        )
        print(
            f'Impact X Position - Standard Deviation: {np.std(dispersion_results["impactX"]):0.3f} m'
        )

        return None

    def plotImpactXPosition(self, dispersion_results):

        self.meanImpactXPosition(dispersion_results)

        plt.figure()
        plt.hist(dispersion_results["impactX"], bins=int(self.N**0.5))
        plt.title("Impact X Position")
        plt.xlabel("Impact X Position (m)")
        plt.ylabel("Number of Occurences")
        plt.show()

        return None

    def meanImpactYPosition(self, dispersion_results):
        print(
            f'Impact Y Position -         Mean Value: {np.mean(dispersion_results["impactY"]):0.3f} m'
        )
        print(
            f'Impact Y Position - Standard Deviation: {np.std(dispersion_results["impactY"]):0.3f} m'
        )

        return None

    def plotImpactYPosition(self, dispersion_results):

        self.meanImpactYPosition(dispersion_results)

        plt.figure()
        plt.hist(dispersion_results["impactY"], bins=int(self.N**0.5))
        plt.title("Impact Y Position")
        plt.xlabel("Impact Y Position (m)")
        plt.ylabel("Number of Occurences")
        plt.show()

        return None

    def meanImpactVelocity(self, dispersion_results):
        print(
            f'Impact Velocity -         Mean Value: {np.mean(dispersion_results["impactVelocity"]):0.3f} m/s'
        )
        print(
            f'Impact Velocity - Standard Deviation: {np.std(dispersion_results["impactVelocity"]):0.3f} m/s'
        )

        return None

    def plotImpactVelocity(self, dispersion_results):

        self.meanImpactVelocity(dispersion_results)

        plt.figure()
        plt.hist(dispersion_results["impactVelocity"], bins=int(self.N**0.5))
        plt.title("Impact Velocity")
        plt.xlim(-35, 0)
        plt.xlabel("Velocity (m/s)")
        plt.ylabel("Number of Occurences")
        plt.show()

        return None

    def meanStaticMargin(self, dispersion_results):
        print(
            f'Initial Static Margin -             Mean Value: {np.mean(dispersion_results["initialStaticMargin"]):0.3f} c'
        )
        print(
            f'Initial Static Margin -     Standard Deviation: {np.std(dispersion_results["initialStaticMargin"]):0.3f} c'
        )

        print(
            f'Out of Rail Static Margin -         Mean Value: {np.mean(dispersion_results["outOfRailStaticMargin"]):0.3f} c'
        )
        print(
            f'Out of Rail Static Margin - Standard Deviation: {np.std(dispersion_results["outOfRailStaticMargin"]):0.3f} c'
        )

        print(
            f'Final Static Margin -               Mean Value: {np.mean(dispersion_results["finalStaticMargin"]):0.3f} c'
        )
        print(
            f'Final Static Margin -       Standard Deviation: {np.std(dispersion_results["finalStaticMargin"]):0.3f} c'
        )

        return None

    def plotStaticMargin(self, dispersion_results):

        self.meanStaticMargin(dispersion_results)

        plt.figure()
        plt.hist(
            dispersion_results["initialStaticMargin"],
            label="Initial",
            bins=int(self.N**0.5),
        )
        plt.hist(
            dispersion_results["outOfRailStaticMargin"],
            label="Out of Rail",
            bins=int(self.N**0.5),
        )
        plt.hist(
            dispersion_results["finalStaticMargin"],
            label="Final",
            bins=int(self.N**0.5),
        )
        plt.legend()
        plt.title("Static Margin")
        plt.xlabel("Static Margin (c)")
        plt.ylabel("Number of Occurences")
        plt.show()

        return None

    def meanMaximumVelocity(self, dispersion_results):
        print(
            f'Maximum Velocity -         Mean Value: {np.mean(dispersion_results["maxVelocity"]):0.3f} m/s'
        )
        print(
            f'Maximum Velocity - Standard Deviation: {np.std(dispersion_results["maxVelocity"]):0.3f} m/s'
        )

        return None

    def plotMaximumVelocity(self, dispersion_results):

        self.meanMaximumVelocity(dispersion_results)

        plt.figure()
        plt.hist(dispersion_results["maxVelocity"], bins=int(self.N**0.5))
        plt.title("Maximum Velocity")
        plt.xlabel("Velocity (m/s)")
        plt.ylabel("Number of Occurences")
        plt.show()

        return None

    def meanNumberOfParachuteEvents(self, dispersion_results):
        print(
            f'Number of Parachute Events -         Mean Value: {np.mean(dispersion_results["numberOfEvents"]):0.3f} s'
        )
        print(
            f'Number of Parachute Events - Standard Deviation: {np.std(dispersion_results["numberOfEvents"]):0.3f} s'
        )

        return None

    def plotNumberOfParachuteEvents(self, dispersion_results):

        self.meanNumberOfParachuteEvents(dispersion_results)

        plt.figure()
        plt.hist(dispersion_results["numberOfEvents"])
        plt.title("Parachute Events")
        plt.xlabel("Number of Parachute Events")
        plt.ylabel("Number of Occurences")
        plt.show()

        return None

    def meanDrogueTriggerTime(self, dispersion_results):
        print(
            f'Drogue Trigger Time -         Mean Value: {np.mean(dispersion_results["drogueTriggerTime"]):0.3f} s'
        )
        print(
            f'Drogue Trigger Time - Standard Deviation: {np.std(dispersion_results["drogueTriggerTime"]):0.3f} s'
        )

        return None

    def plotDrogueTriggerTime(self, dispersion_results):

        self.meanDrogueTriggerTime(dispersion_results)

        plt.figure()
        plt.hist(dispersion_results["drogueTriggerTime"], bins=int(self.N**0.5))
        plt.title("Drogue Trigger Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Number of Occurences")
        plt.show()

        return None

    def meanDrogueFullyInflatedTime(self, dispersion_results):
        print(
            f'Drogue Fully Inflated Time -         Mean Value: {np.mean(dispersion_results["drogueInflatedTime"]):0.3f} s'
        )
        print(
            f'Drogue Fully Inflated Time - Standard Deviation: {np.std(dispersion_results["drogueInflatedTime"]):0.3f} s'
        )

        return None

    def plotDrogueFullyInflatedTime(self, dispersion_results):

        self.meanDrogueFullyInflatedTime(dispersion_results)

        plt.figure()
        plt.hist(dispersion_results["drogueInflatedTime"], bins=int(self.N**0.5))
        plt.title("Drogue Fully Inflated Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Number of Occurences")
        plt.show()

        return None

    def meanDrogueFullyVelocity(self, dispersion_results):
        print(
            f'Drogue Parachute Fully Inflated Velocity -         Mean Value: {np.mean(dispersion_results["drogueInflatedVelocity"]):0.3f} m/s'
        )
        print(
            f'Drogue Parachute Fully Inflated Velocity - Standard Deviation: {np.std(dispersion_results["drogueInflatedVelocity"]):0.3f} m/s'
        )

        return None

    def plotDrogueFullyVelocity(self, dispersion_results):

        self.meanDrogueFullyVelocity(dispersion_results)

        plt.figure()
        plt.hist(dispersion_results["drogueInflatedVelocity"], bins=int(self.N**0.5))
        plt.title("Drogue Parachute Fully Inflated Velocity")
        plt.xlabel("Velocity m/s)")
        plt.ylabel("Number of Occurences")
        plt.show()

        return None

    def plotEllipses(self, dispersion_results, image, realLandingPoint):

        # Import background map
        img = imread(image)

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

        # Create plot figure
        plt.figure(num=None, figsize=(8, 6), dpi=150, facecolor="w", edgecolor="k")
        ax = plt.subplot(111)

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
            ax.add_artist(impactEll)

        # Calculate error ellipses for apogee
        apogeeCov = np.cov(apogeeX, apogeeY)
        apogeeVals, apogeeVecs = eigsorted(apogeeCov)
        apogeeTheta = np.degrees(np.arctan2(*apogeeVecs[:, 0][::-1]))
        apogeeW, apogeeH = 2 * np.sqrt(apogeeVals)

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
            ax.add_artist(apogeeEll)

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
        if realLandingPoint != None:
            plt.scatter(
                realLandingPoint[0],
                realLandingPoint[1],
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
        plt.imshow(img, zorder=0, extent=[-3000 - dx, 3000 - dx, -3000 - dy, 3000 - dy])
        plt.axhline(0, color="black", linewidth=0.5)
        plt.axvline(0, color="black", linewidth=0.5)
        plt.xlim(-3000, 3000)
        plt.ylim(-3000, 3000)

        # Save plot and show result
        plt.savefig(str(self.filename) + ".pdf", bbox_inches="tight", pad_inches=0)
        plt.savefig(str(self.filename) + ".svg", bbox_inches="tight", pad_inches=0)
        plt.show()

    def meanLateralWindSpeed(self, dispersion_results):
        print(
            f'Lateral Surface Wind Speed -         Mean Value: {np.mean(dispersion_results["lateralWind"]):0.3f} m/s'
        )
        print(
            f'Lateral Surface Wind Speed - Standard Deviation: {np.std(dispersion_results["lateralWind"]):0.3f} m/s'
        )

    def plotLateralWindSpeed(self, dispersion_results):

        self.meanLateralWindSpeed(dispersion_results)

        plt.figure()
        plt.hist(dispersion_results["lateralWind"], bins=int(self.N**0.5))
        plt.title("Lateral Surface Wind Speed")
        plt.xlabel("Velocity (m/s)")
        plt.ylabel("Number of Occurences")
        plt.show()

    def meanFrontalWindSpeed(self, dispersion_results):
        print(
            f'Frontal Surface Wind Speed -         Mean Value: {np.mean(dispersion_results["frontalWind"]):0.3f} m/s'
        )
        print(
            f'Frontal Surface Wind Speed - Standard Deviation: {np.std(dispersion_results["frontalWind"]):0.3f} m/s'
        )

    def plotFrontalWindSpeed(self, dispersion_results):

        self.meanFrontalWindSpeed(dispersion_results)

        plt.figure()
        plt.hist(dispersion_results["frontalWind"], bins=int(self.N**0.5))
        plt.title("Frontal Surface Wind Speed")
        plt.xlabel("Velocity (m/s)")
        plt.ylabel("Number of Occurences")
        plt.show()

    def info(self):

        dispersion_results = self.importResults(self.filename)

        self.meanApogeeAltitude(dispersion_results)

        self.meanOutOfRailVelocity(dispersion_results)

        self.meanStaticMargin(dispersion_results)

        self.meanLateralWindSpeed(dispersion_results)

        self.meanFrontalWindSpeed(dispersion_results)

        self.meanOutOfRailTime(dispersion_results)

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

    def allInfo(self):
        dispersion_results = self.importResults(self.filename)

        self.plotEllipses(dispersion_results, self.image, self.realLandingPoint)

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
