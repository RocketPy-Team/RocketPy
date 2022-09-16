# -*- coding: utf-8 -*-

__author__ = ""
__copyright__ = "Copyright 20XX, Projeto Jupiter"
__license__ = "MIT"


from rocketpy import *
import netCDF4


from datetime import datetime
from os import _Environ
from time import process_time, perf_counter, time
import glob
import traceback


import numpy as np
from numpy.random import (
    normal,
    uniform,
    choice,
    beta,
    binomial,
    chisquare,
    dirichlet,
    exponential,
    f,
    gamma,
    geometric,
    gumbel,
    hypergeometric,
    laplace,
    logistic,
    lognormal,
    logseries,
    multinomial,
    multivariate_normal,
    negative_binomial,
    noncentral_chisquare,
    noncentral_f,
    pareto,
    poisson,
    power,
    rayleigh,
    standard_cauchy,
    standard_exponential,
    standard_gamma,
    standard_normal,
    standard_t,
    triangular,
    vonmises,
    wald,
    weibull,
    zipf,
)
from IPython.display import display
from rocketpy.Environment import Environment
import matplotlib.pyplot as plt
from imageio import imread
from matplotlib.patches import Ellipse


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

    def __init__(  # TODO: init should only intialize the variables
        self,
        filename,
        number_of_simulations,  # TODO: needs to be obligatory
        flight=None,
        image=None,
        dispersionDict={},
        environment=None,
        motor=None,
        rocket=None,
        distributionType="normal",
        realLandingPoint=None,
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

        # Save parameters
        self.filename = filename
        self.image = image
        self.realLandingPoint = realLandingPoint

        # Run a new simulation
        if number_of_simulations > 0:

            self.flight = flight
            self.environment = flight.env if not environment else environment
            self.rocket = flight.rocket if not rocket else rocket
            self.motor = flight.rocket.motor if not motor else motor

            analysis_parameters = {i: j for i, j in dispersionDict.items()}

            def flight_settings(analysis_parameters, number_of_simulations):
                i = 0
                while i < number_of_simulations:
                    # Generate a flight setting
                    flight_setting = {}
                    for parameter_key, parameter_value in analysis_parameters.items():
                        if type(parameter_value) is tuple:
                            if distributionType == "normal" or distributionType == None:
                                flight_setting[parameter_key] = normal(*parameter_value)
                            if distributionType == "beta":
                                flight_setting[parameter_key] = beta(*parameter_value)
                            if distributionType == "binomial":
                                flight_setting[parameter_key] = binomial(
                                    *parameter_value
                                )
                            if distributionType == "chisquare":
                                flight_setting[parameter_key] = chisquare(
                                    *parameter_value
                                )
                            if distributionType == "dirichlet":
                                flight_setting[parameter_key] = dirichlet(
                                    *parameter_value
                                )
                            if distributionType == "exponential":
                                flight_setting[parameter_key] = exponential(
                                    *parameter_value
                                )
                            if distributionType == "f":
                                flight_setting[parameter_key] = f(*parameter_value)
                            if distributionType == "gamma":
                                flight_setting[parameter_key] = gamma(*parameter_value)
                            if distributionType == "geometric":
                                flight_setting[parameter_key] = geometric(
                                    *parameter_value
                                )
                            if distributionType == "gumbel":
                                flight_setting[parameter_key] = gumbel(*parameter_value)
                            if distributionType == "hypergeometric":
                                flight_setting[parameter_key] = hypergeometric(
                                    *parameter_value
                                )
                            if distributionType == "laplace":
                                flight_setting[parameter_key] = laplace(
                                    *parameter_value
                                )
                            if distributionType == "logistic":
                                flight_setting[parameter_key] = logistic(
                                    *parameter_value
                                )
                            if distributionType == "lognormal":
                                flight_setting[parameter_key] = lognormal(
                                    *parameter_value
                                )
                            if distributionType == "logseries":
                                flight_setting[parameter_key] = logseries(
                                    *parameter_value
                                )
                            if distributionType == "multinomial":
                                flight_setting[parameter_key] = multinomial(
                                    *parameter_value
                                )
                            if distributionType == "multivariate_normal":
                                flight_setting[parameter_key] = multivariate_normal(
                                    *parameter_value
                                )
                            if distributionType == "negative_binomial":
                                flight_setting[parameter_key] = negative_binomial(
                                    *parameter_value
                                )
                            if distributionType == "noncentral_chisquare":
                                flight_setting[parameter_key] = noncentral_chisquare(
                                    *parameter_value
                                )
                            if distributionType == "noncentral_f":
                                flight_setting[parameter_key] = noncentral_f(
                                    *parameter_value
                                )
                            if distributionType == "pareto":
                                flight_setting[parameter_key] = pareto(*parameter_value)
                            if distributionType == "poisson":
                                flight_setting[parameter_key] = poisson(
                                    *parameter_value
                                )
                            if distributionType == "power":
                                flight_setting[parameter_key] = power(*parameter_value)
                            if distributionType == "rayleigh":
                                flight_setting[parameter_key] = rayleigh(
                                    *parameter_value
                                )
                            if distributionType == "standard_cauchy":
                                flight_setting[parameter_key] = standard_cauchy(
                                    *parameter_value
                                )
                            if distributionType == "standard_exponential":
                                flight_setting[parameter_key] = standard_exponential(
                                    *parameter_value
                                )
                            if distributionType == "standard_gamma":
                                flight_setting[parameter_key] = standard_gamma(
                                    *parameter_value
                                )
                            if distributionType == "standard_normal":
                                flight_setting[parameter_key] = standard_normal(
                                    *parameter_value
                                )
                            if distributionType == "standard_t":
                                flight_setting[parameter_key] = standard_t(
                                    *parameter_value
                                )
                            if distributionType == "triangular":
                                flight_setting[parameter_key] = triangular(
                                    *parameter_value
                                )
                            if distributionType == "uniform":
                                flight_setting[parameter_key] = uniform(
                                    *parameter_value
                                )
                            if distributionType == "vonmises":
                                flight_setting[parameter_key] = vonmises(
                                    *parameter_value
                                )
                            if distributionType == "wald":
                                flight_setting[parameter_key] = wald(*parameter_value)
                            if distributionType == "weibull":
                                flight_setting[parameter_key] = weibull(
                                    *parameter_value
                                )
                            if distributionType == "zipf":
                                flight_setting[parameter_key] = zipf(*parameter_value)
                        else:
                            flight_setting[parameter_key] = choice(parameter_value)

                    # Skip if certain values are negative, which happens due to the normal curve but isnt realistic
                    if (
                        "lag_rec" in analysis_parameters
                        and flight_setting["lag_rec"] < 0
                    ):  # TODO: change "rec" to be more specifci
                        continue
                    if (
                        "lag_se" in analysis_parameters and flight_setting["lag_se"] < 0
                    ):  # TODO: change "se" to be more specifci
                        continue
                    # Update counter
                    i += 1
                    # Yield a flight setting
                    yield flight_setting

            def export_flight_data(flight_setting, flight_data, exec_time):
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
                if len(flight_data.parachuteEvents) > 0:
                    flight_result["drogueTriggerTime"] = flight_data.parachuteEvents[0][
                        0
                    ]
                    flight_result["drogueInflatedTime"] = (
                        flight_data.parachuteEvents[0][0]
                        + flight_data.parachuteEvents[0][1].lag
                    )
                    flight_result["drogueInflatedVelocity"] = flight_data.v(
                        flight_data.parachuteEvents[0][0]
                        + flight_data.parachuteEvents[0][1].lag
                    )
                else:
                    flight_result["drogueTriggerTime"] = 0
                    flight_result["drogueInflatedTime"] = 0
                    flight_result["drogueInflatedVelocity"] = 0

                # Write flight setting and results to file
                dispersion_input_file.write(str(flight_setting) + "\n")
                dispersion_output_file.write(str(flight_result) + "\n")

            def export_flight_error(flight_setting):
                dispersion_error_file.write(str(flight_setting) + "\n")

            # Basic analysis info

            # Create data files for inputs, outputs and error logging
            dispersion_error_file = open(str(filename) + ".disp_errors.txt", "w")
            dispersion_input_file = open(str(filename) + ".disp_inputs.txt", "w")
            dispersion_output_file = open(str(filename) + ".disp_outputs.txt", "w")

            # Initialize counter and timer
            i = 0

            initial_wall_time = time()
            initial_cpu_time = process_time()

            # Iterate over flight settings
            out = display("Starting", display_id=True)
            for setting in flight_settings(analysis_parameters, number_of_simulations):
                start_time = process_time()
                i += 1

                # Creates copy of environment
                envDispersion = self.environment

                # Apply environment parameters variations on each iteration if possible
                envDispersion.railLength = (
                    setting["railLength"]
                    if "railLength" in setting
                    else envDispersion.rL
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
                    setting["longitude"]
                    if "longitude" in setting
                    else envDispersion.lon
                )
                envDispersion.elevation = (
                    setting["elevation"]
                    if "elevation" in setting
                    else envDispersion.elevation
                )
                envDispersion.datum = (
                    setting["datum"] if "datum" in setting else envDispersion.datum
                )
                if "ensembleMember" in setting:
                    envDispersion.selectEnsembleMember(setting["ensembleMember"])

                # Creates copy of motor
                motorDispersion = self.motor

                # Apply motor parameters variations on each iteration if possible
                motorDispersion = SolidMotor(
                    thrustSource=setting["thrustSource"]
                    if "thrustSource" in setting
                    else motorDispersion.thrustSource,
                    burnOut=setting["burnOut"]
                    if "burnOut" in setting
                    else motorDispersion.burnOut,
                    grainNumber=setting["grainNumber"]
                    if "grainNumber" in setting
                    else motorDispersion.grainNumber,
                    grainDensity=setting["grainDensity"]
                    if "grainDensity" in setting
                    else motorDispersion.grainDensity,
                    grainOuterRadius=setting["grainOuterRadius"]
                    if "grainOuterRadius" in setting
                    else motorDispersion.grainOuterRadius,
                    grainInitialInnerRadius=setting["grainInitialInnerRadius"]
                    if "grainInitialInnerRadius" in setting
                    else motorDispersion.grainInitialInnerRadius,
                    grainInitialHeight=setting["grainInitialHeight"]
                    if "grainInitialHeight" in setting
                    else motorDispersion.grainInitialHeight,
                    grainSeparation=setting["grainSeparation"]
                    if "grainSeparation" in setting
                    else motorDispersion.grainSeparation,
                    nozzleRadius=setting["nozzleRadius"]
                    if "nozzleRadius" in setting
                    else motorDispersion.nozzleRadius,
                    throatRadius=setting["throatRadius"]
                    if "throatRadius" in setting
                    else motorDispersion.throatRadius,
                    reshapeThrustCurve=setting["reshapeThrustCurve"]
                    if "reshapeThrustCurve" in setting
                    else motorDispersion.reshapeThrustCurve,
                    interpolationMethod=setting["interpolationMethod"]
                    if "interpolationMethod" in setting
                    else motorDispersion.interpolationMethod,
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
                    radius=setting["radius"]
                    if "radius" in setting
                    else self.rocket.radius,
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
                    n=setting["n"] if "n" in setting else self.rocket.numberOfFins,
                    rootChord=setting["rootChord"]
                    if "rootChord" in setting
                    else self.rocket.rootChord,
                    tipChord=setting["tipChord"]
                    if "tipChord" in setting
                    else self.rocket.tipChord,
                    span=setting["span"] if "span" in setting else self.rocket.span,
                    distanceToCM=setting["finDistanceToCM"]
                    if "finDistanceToCM" in setting
                    else self.rocket.distanceRocketFins,
                    radius=setting["radius"]
                    if "radius" in setting
                    else self.rocket.finRadius,
                    airfoil=setting["airfoil"]
                    if "airfoil" in setting
                    else self.rocket.finAirfoil,
                )
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

                # Add parachute
                rocketDispersion.addParachute(
                    name=setting["name"]
                    if "name" in setting
                    else self.rocket.parachuteName,
                    CdS=setting["CdS"]
                    if "CdS" in setting
                    else self.rocket.parachuteCdS,
                    trigger=setting["trigger"]
                    if "trigger" in setting
                    else self.rocket.parachuteTrigger,
                    samplingRate=setting["samplingRate"]
                    if "samplingRate" in setting
                    else self.rocket.parachuteSamplingRate,
                    lag=setting["lag_rec"]
                    if "lag_rec" in setting
                    else self.rocket.lag_rec + setting["lag_se"]
                    if "lag_se" in setting
                    else self.rocket.parachuteLag,
                    noise=setting["noise"]
                    if "noise" in setting
                    else self.rocket.parachuteNoise,
                )

                rocketDispersion.setRailButtons(
                    distanceToCM=setting["RBdistanceToCM"]
                    if "RBdistanceToCM" in setting
                    else self.rocket.RBdistanceToCM,
                    angularPosition=setting["angularPosition"]
                    if "angularPosition" in setting
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
                        else self.flight.terminateOnApogee,
                        maxTime=setting["maxTime"]
                        if "maxTime" in setting
                        else self.flight.maxTime,
                        maxTimeStep=setting["maxTimeStep"]
                        if "maxTimeStep" in setting
                        else self.flight.maxTimeStep,
                        minTimeStep=setting["minTimeStep"]
                        if "minTimeStep" in setting
                        else self.flight.minTimeStep,
                        rtol=setting["rtol"] if "rtol" in setting else self.flight.rtol,
                        atol=setting["atol"] if "atol" in setting else self.flight.atol,
                        timeOvershoot=setting["timeOvershoot"]
                        if "timeOvershoot" in setting
                        else self.flight.timeOvershoot,
                        verbose=False,
                    )

                    export_flight_data(setting, TestFlight, process_time() - start_time)
                except Exception as E:
                    print(E)
                    print(traceback.format_exc())
                    export_flight_error(setting)

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

    def importingDispersionResultsFromFile(self, dispersion_output_file):

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

    def yield_flight_setting(self):
        """Yields a flight setting for the simulation"""
        yield None

    def export_flight_settings(self):
        """Saves flight results in a .txt"""
        return None

    def export_flight_error(self):
        """Saves flight error in a .txt"""
        return None

    def runDispersion(self):
        """Runs the given number of simulations and saves the data"""
        return None

    def plotOutOfRailTime(self, dispersion_results):
        print(
            f'Out of Rail Time -         Mean Value: {np.mean(dispersion_results["outOfRailTime"]):0.3f} s'
        )
        print(
            f'Out of Rail Time - Standard Deviation: {np.std(dispersion_results["outOfRailTime"]):0.3f} s'
        )

        plt.figure()
        plt.hist(dispersion_results["outOfRailTime"], bins=int(self.N**0.5))
        plt.title("Out of Rail Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Number of Occurences")
        plt.show()

        return None

    def plotOutOfRailVelocity(self, dispersion_results):
        print(
            f'Out of Rail Velocity -         Mean Value: {np.mean(dispersion_results["outOfRailVelocity"]):0.3f} m/s'
        )
        print(
            f'Out of Rail Velocity - Standard Deviation: {np.std(dispersion_results["outOfRailVelocity"]):0.3f} m/s'
        )

        plt.figure()
        plt.hist(dispersion_results["outOfRailVelocity"], bins=int(self.N**0.5))
        plt.title("Out of Rail Velocity")
        plt.xlabel("Velocity (m/s)")
        plt.ylabel("Number of Occurences")
        plt.show()

        return None

    def plotApogeeTime(self, dispersion_results):
        print(
            f'Impact Time -         Mean Value: {np.mean(dispersion_results["impactTime"]):0.3f} s'
        )
        print(
            f'Impact Time - Standard Deviation: {np.std(dispersion_results["impactTime"]):0.3f} s'
        )

        plt.figure()
        plt.hist(dispersion_results["impactTime"], bins=int(self.N**0.5))
        plt.title("Impact Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Number of Occurences")
        plt.show()

        return None

    def plotApogeeAltitude(self, dispersion_results):
        print(
            f'Apogee Altitude -         Mean Value: {np.mean(dispersion_results["apogeeAltitude"]):0.3f} m'
        )
        print(
            f'Apogee Altitude - Standard Deviation: {np.std(dispersion_results["apogeeAltitude"]):0.3f} m'
        )

        plt.figure()
        plt.hist(dispersion_results["apogeeAltitude"], bins=int(self.N**0.5))
        plt.title("Apogee Altitude")
        plt.xlabel("Altitude (m)")
        plt.ylabel("Number of Occurences")
        plt.show()

        return None

    def plotApogeeXPosition(self, dispersion_results):
        print(
            f'Apogee X Position -         Mean Value: {np.mean(dispersion_results["apogeeX"]):0.3f} m'
        )
        print(
            f'Apogee X Position - Standard Deviation: {np.std(dispersion_results["apogeeX"]):0.3f} m'
        )

        plt.figure()
        plt.hist(dispersion_results["apogeeX"], bins=int(self.N**0.5))
        plt.title("Apogee X Position")
        plt.xlabel("Apogee X Position (m)")
        plt.ylabel("Number of Occurences")
        plt.show()

        return None

    def plotApogeeYPosition(self, dispersion_results):
        print(
            f'Apogee Y Position -         Mean Value: {np.mean(dispersion_results["apogeeY"]):0.3f} m'
        )
        print(
            f'Apogee Y Position - Standard Deviation: {np.std(dispersion_results["apogeeY"]):0.3f} m'
        )

        plt.figure()
        plt.hist(dispersion_results["apogeeY"], bins=int(self.N**0.5))
        plt.title("Apogee Y Position")
        plt.xlabel("Apogee Y Position (m)")
        plt.ylabel("Number of Occurences")
        plt.show()

        return None

    def plotImpactTime(self, dispersion_results):
        print(
            f'Impact Time -         Mean Value: {np.mean(dispersion_results["impactTime"]):0.3f} s'
        )
        print(
            f'Impact Time - Standard Deviation: {np.std(dispersion_results["impactTime"]):0.3f} s'
        )

        plt.figure()
        plt.hist(dispersion_results["impactTime"], bins=int(self.N**0.5))
        plt.title("Impact Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Number of Occurences")
        plt.show()

        return None

    def plotImpactXPosition(self, dispersion_results):
        print(
            f'Impact X Position -         Mean Value: {np.mean(dispersion_results["impactX"]):0.3f} m'
        )
        print(
            f'Impact X Position - Standard Deviation: {np.std(dispersion_results["impactX"]):0.3f} m'
        )

        plt.figure()
        plt.hist(dispersion_results["impactX"], bins=int(self.N**0.5))
        plt.title("Impact X Position")
        plt.xlabel("Impact X Position (m)")
        plt.ylabel("Number of Occurences")
        plt.show()

        return None

    def plotImpactYPosition(self, dispersion_results):
        print(
            f'Impact Y Position -         Mean Value: {np.mean(dispersion_results["impactY"]):0.3f} m'
        )
        print(
            f'Impact Y Position - Standard Deviation: {np.std(dispersion_results["impactY"]):0.3f} m'
        )

        plt.figure()
        plt.hist(dispersion_results["impactY"], bins=int(self.N**0.5))
        plt.title("Impact Y Position")
        plt.xlabel("Impact Y Position (m)")
        plt.ylabel("Number of Occurences")
        plt.show()

        return None

    def plotImpactVelocity(self, dispersion_results):
        print(
            f'Impact Velocity -         Mean Value: {np.mean(dispersion_results["impactVelocity"]):0.3f} m/s'
        )
        print(
            f'Impact Velocity - Standard Deviation: {np.std(dispersion_results["impactVelocity"]):0.3f} m/s'
        )

        plt.figure()
        plt.hist(dispersion_results["impactVelocity"], bins=int(self.N**0.5))
        plt.title("Impact Velocity")
        plt.xlim(-35, 0)
        plt.xlabel("Velocity (m/s)")
        plt.ylabel("Number of Occurences")
        plt.show()

        return None

    def plotStaticMargin(self, dispersion_results):
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

    def plotMaximumVelocity(self, dispersion_results):
        print(
            f'Maximum Velocity -         Mean Value: {np.mean(dispersion_results["maxVelocity"]):0.3f} m/s'
        )
        print(
            f'Maximum Velocity - Standard Deviation: {np.std(dispersion_results["maxVelocity"]):0.3f} m/s'
        )

        plt.figure()
        plt.hist(dispersion_results["maxVelocity"], bins=int(self.N**0.5))
        plt.title("Maximum Velocity")
        plt.xlabel("Velocity (m/s)")
        plt.ylabel("Number of Occurences")
        plt.show()

        return None

    def plotNumberOfParachuteEvents(self, dispersion_results):
        plt.figure()
        plt.hist(dispersion_results["numberOfEvents"])
        plt.title("Parachute Events")
        plt.xlabel("Number of Parachute Events")
        plt.ylabel("Number of Occurences")
        plt.show()

        return None

    def plotDrogueTriggerTime(self, dispersion_results):
        print(
            f'Drogue Trigger Time -         Mean Value: {np.mean(dispersion_results["drogueTriggerTime"]):0.3f} s'
        )
        print(
            f'Drogue Trigger Time - Standard Deviation: {np.std(dispersion_results["drogueTriggerTime"]):0.3f} s'
        )

        plt.figure()
        plt.hist(dispersion_results["drogueTriggerTime"], bins=int(self.N**0.5))
        plt.title("Drogue Trigger Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Number of Occurences")
        plt.show()

        return None

    def plotDrogueFullyInflatedTime(self, dispersion_results):
        print(
            f'Drogue Fully Inflated Time -         Mean Value: {np.mean(dispersion_results["drogueInflatedTime"]):0.3f} s'
        )
        print(
            f'Drogue Fully Inflated Time - Standard Deviation: {np.std(dispersion_results["drogueInflatedTime"]):0.3f} s'
        )

        plt.figure()
        plt.hist(dispersion_results["drogueInflatedTime"], bins=int(self.N**0.5))
        plt.title("Drogue Fully Inflated Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Number of Occurences")
        plt.show()

        return None

    def plotDrogueFullyVelocity(self, dispersion_results):
        print(
            f'Drogue Parachute Fully Inflated Velocity -         Mean Value: {np.mean(dispersion_results["drogueInflatedVelocity"]):0.3f} m/s'
        )
        print(
            f'Drogue Parachute Fully Inflated Velocity - Standard Deviation: {np.std(dispersion_results["drogueInflatedVelocity"]):0.3f} m/s'
        )

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

    def plotLateralWindSpeed(self, dispersion_results):
        print(
            f'Lateral Surface Wind Speed -         Mean Value: {np.mean(dispersion_results["lateralWind"]):0.3f} m/s'
        )
        print(
            f'Lateral Surface Wind Speed - Standard Deviation: {np.std(dispersion_results["lateralWind"]):0.3f} m/s'
        )

        plt.figure()
        plt.hist(dispersion_results["lateralWind"], bins=int(self.N**0.5))
        plt.title("Lateral Surface Wind Speed")
        plt.xlabel("Velocity (m/s)")
        plt.ylabel("Number of Occurences")
        plt.show()

    def plotFrontalWindSpeed(self, dispersion_results):
        print(
            f'Frontal Surface Wind Speed -         Mean Value: {np.mean(dispersion_results["frontalWind"]):0.3f} m/s'
        )
        print(
            f'Frontal Surface Wind Speed - Standard Deviation: {np.std(dispersion_results["frontalWind"]):0.3f} m/s'
        )

        plt.figure()
        plt.hist(dispersion_results["frontalWind"], bins=int(self.N**0.5))
        plt.title("Frontal Surface Wind Speed")
        plt.xlabel("Velocity (m/s)")
        plt.ylabel("Number of Occurences")
        plt.show()

    def info(self):
        dispersion_results = self.importingDispersionResultsFromFile(self.filename)

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
