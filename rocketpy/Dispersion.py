# -*- coding: utf-8 -*-

__author__ = ""
__copyright__ = "Copyright 20XX, Projeto Jupiter"
__license__ = "MIT"


from datetime import datetime
from os import _Environ
from time import process_time, perf_counter, time
import glob


import numpy as np
from numpy.random import normal, uniform, choice, beta, binomial, chisquare, dirichlet, exponential, f, gamma, geometric, gumbel, hypergeometric, laplace, logistic, lognormal, logseries, multinomial, multivariate_normal, negative_binomial, noncentral_chisquare, noncentral_f, pareto, poisson, power, rayleigh, standard_cauchy, standard_exponential, standard_gamma, standard_normal, standard_t, triangular, vonmises, wald, weibull, zipf
from IPython.display import display
from rocketpy.Environment import Environment
import matplotlib.pyplot as plt
from imageio import imread
from matplotlib.patches import Ellipse

from rocketpy import *


class Dispersion:
    """

    Documentation

    """

    def __init__(
        self,
        number_of_simulations,
        flight,
        filename,
        dispersionDict,  # Dicionario exemplo: {"mass": (desvio padrão, valor ~opcional~)}
        environment=None,
        motor=None,
        rocket=None,
        distributionType=None,  
    ):

        self.flight = flight
        self.environment = flight.env if not environment else environment
        self.rocket = flight.rocket if not rocket else rocket
        self.motor = flight.rocket.motor if not motor else motor

        analysis_parameters = {i: j for i, j in dispersionDict.items()}

        i = 0
        while i < number_of_simulations:
            # Generate a flight setting
            flight_setting = {}
            for parameter_key, parameter_value in analysis_parameters.items():
                if type(parameter_value) is tuple:
                    parameter_value = (parameter_value[1], parameter_value[0])
                    if distributionType == "normal":
                        flight_setting[parameter_key] = normal(*parameter_value)
                    if distributionType == "beta":
                        flight_setting[parameter_key] = beta(*parameter_value)
                    if distributionType == "binomial":
                        flight_setting[parameter_key] = binomial(*parameter_value)
                    if distributionType == "chisquare":
                        flight_setting[parameter_key] = chisquare(*parameter_value)
                    if distributionType == "dirichlet":
                        flight_setting[parameter_key] = dirichlet(*parameter_value)
                    if distributionType == "exponential":
                        flight_setting[parameter_key] = exponential(*parameter_value)
                    if distributionType == "f":
                        flight_setting[parameter_key] = f(*parameter_value)
                    if distributionType == "gamma":
                        flight_setting[parameter_key] = gamma(*parameter_value)
                    if distributionType == "geometric":
                        flight_setting[parameter_key] = geometric(*parameter_value)
                    if distributionType == "gumbel":
                        flight_setting[parameter_key] = gumbel(*parameter_value)
                    if distributionType == "hypergeometric":
                        flight_setting[parameter_key] = hypergeometric(*parameter_value)
                    if distributionType == "laplace":
                        flight_setting[parameter_key] = laplace(*parameter_value)
                    if distributionType == "logistic":
                        flight_setting[parameter_key] = logistic(*parameter_value)
                    if distributionType == "lognormal":
                        flight_setting[parameter_key] = lognormal(*parameter_value)
                    if distributionType == "logseries":
                        flight_setting[parameter_key] = logseries(*parameter_value)
                    if distributionType == "multinomial":
                        flight_setting[parameter_key] = multinomial(*parameter_value)
                    if distributionType == "multivariate_normal":
                        flight_setting[parameter_key] = multivariate_normal(*parameter_value)
                    if distributionType == "negative_binomial":
                        flight_setting[parameter_key] = negative_binomial(*parameter_value)
                    if distributionType == "noncentral_chisquare":
                        flight_setting[parameter_key] = noncentral_chisquare(*parameter_value)
                    if distributionType == "noncentral_f":
                        flight_setting[parameter_key] = noncentral_f(*parameter_value)
                    if distributionType == "pareto":
                        flight_setting[parameter_key] = pareto(*parameter_value)
                    if distributionType == "poisson":
                        flight_setting[parameter_key] = poisson(*parameter_value)
                    if distributionType == "power":
                        flight_setting[parameter_key] = power(*parameter_value)
                    if distributionType == "rayleigh":
                        flight_setting[parameter_key] = rayleigh(*parameter_value)
                    if distributionType == "standard_cauchy":
                        flight_setting[parameter_key] = standard_cauchy(*parameter_value)
                    if distributionType == "standard_exponential":
                        flight_setting[parameter_key] = standard_exponential(*parameter_value)
                    if distributionType == "standard_gamma":
                        flight_setting[parameter_key] = standard_gamma(*parameter_value)
                    if distributionType == "standard_normal":
                        flight_setting[parameter_key] = standard_normal(*parameter_value)
                    if distributionType == "standard_t":
                        flight_setting[parameter_key] = standard_t(*parameter_value)
                    if distributionType == "triangular":
                        flight_setting[parameter_key] = triangular(*parameter_value)
                    if distributionType == "uniform":
                        flight_setting[parameter_key] = uniform(*parameter_value)
                    if distributionType == "vonmises":
                        flight_setting[parameter_key] = vonmises(*parameter_value)
                    if distributionType == "wald":
                        flight_setting[parameter_key] = wald(*parameter_value)
                    if distributionType == "weibull":
                        flight_setting[parameter_key] = weibull(*parameter_value)
                    if distributionType == "zipf":
                        flight_setting[parameter_key] = zipf(*parameter_value)
                else:
                    flight_setting[parameter_key] = choice(parameter_value)

            # Skip if certain values are negative, which happens due to the normal curve but isnt realistic
            if flight_setting["lag_rec"] < 0 or flight_setting["lag_se"] < 0:
                continue

            # Update counter
            i += 1
            # Yield a flight setting
            yield flight_setting

        # Basic analysis info
        self.filename = filename

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
        for setting in flight_setting:
            start_time = process_time()
            i += 1

            # Creates copy of environment
            envDispersion = self.environment

            # Apply environment parameters variations on each iteration if possible
            envDispersion.railLength = (
                setting["railLength"]
                if "railLength" in setting
                else envDispersion.railLength
            )
            envDispersion.gravity = (
                setting["gravity"] if "gravity" in setting 
                else envDispersion.gravity
            )
            envDispersion.date = (
                setting["date"] if "date" in setting 
                else envDispersion.date
            )
            envDispersion.latitude = (
                setting["latitude"] if "latitude" in setting 
                else envDispersion.latitude
            )
            envDispersion.longitude = (
                setting["longitude"]
                if "longitude" in setting
                else envDispersion.longitude
            )
            envDispersion.elevation = (
                setting["elevation"]
                if "elevation" in setting
                else envDispersion.elevation
            )
            envDispersion.datum = (
                setting["datum"] if "datum" in setting 
                else envDispersion.datum
            )
            if "ensembleMember" in setting:
                envDispersion.selectEnsembleMember(setting["ensembleMember"])

            
            # Creates copy of motor
            motorDispersion = self.motor

            # Apply motor parameters variations on each iteration if possible
            motorDispersion.thrustSource = (
                setting["thrustSource"]
                if "thrustSource" in setting
                else motorDispersion.thrustSource
            )
            motorDispersion.burnOut = (
                setting["burnOut"]
                if "burnOut" in setting
                else motorDispersion.burnOut
            )
            motorDispersion.grainNumber = (
                setting["grainNumber"]
                if "grainNumber" in setting
                else motorDispersion.grainNumber
            )
            motorDispersion.grainDensity = (
                setting["grainDensity"]
                if "grainDensity" in setting
                else motorDispersion.grainDensity
            )
            motorDispersion.grainOuterRadius = (
                setting["grainOuterRadius"]
                if "grainOuterRadius" in setting
                else motorDispersion.grainOuterRadius
            )
            motorDispersion.grainInitialInnerRadius = (
                setting["grainInitialInnerRadius"]
                if "grainInitialInnerRadius" in setting
                else motorDispersion.grainInitialInnerRadius
            )
            motorDispersion.grainInitialHeight = (
                setting["grainInitialHeight"]
                if "grainInitialHeight" in setting
                else motorDispersion.grainInitialHeight
            )
            motorDispersion.grainSeparation = (
                setting["grainSeparation"]
                if "grainSeparation" in setting
                else motorDispersion.grainSeparation
            )
            motorDispersion.nozzleRadius = (
                setting["nozzleRadius"]
                if "nozzleRadius" in setting
                else motorDispersion.nozzleRadius
            )
            motorDispersion.throatRadius = (
                setting["throatRadius"]
                if "throatRadius" in setting
                else motorDispersion.throatRadius
            )
            motorDispersion.reshapeThrustCurve = (
                setting["reshapeThrustCurve"]
                if "reshapeThrustCurve" in setting
                else motorDispersion.reshapeThrustCurve
            )
            motorDispersion.interpolationMethod = (      
                setting["interpolationMethod"]
                if "interpolationMethod" in setting
                else motorDispersion.interpolationMethod
            )
            
            # Creates copy of motor
            rocketDispersion = self.rocket

            # Apply rocket parameters variations on each iteration if possible
            rocketDispersion.motor = (
                setting["motor"]
                if "motor" in setting
                else rocketDispersion.motor
            )
            rocketDispersion.radius = (
                setting["radius"]
                if "radius" in setting
                else rocketDispersion.radius
            )
            rocketDispersion.rocketMass = (
                setting["rocketMass"]
                if "rocketMass" in setting
                else rocketDispersion.rocketMass
            )
            rocketDispersion.rocketMass = (
                setting["inertiaI"]
                if "inertiaI" in setting
                else rocketDispersion.inertiaI
            )
            rocketDispersion.inertiaZ = (
                setting["inertiaZ"]
                if "inertiaZ" in setting
                else rocketDispersion.inertiaZ
            )
            rocketDispersion.distanceRocketNozzle = (
                setting["distanceRocketNozzle"]
                if "distanceRocketNozzle" in setting
                else rocketDispersion.distanceRocketNozzle
            )
            rocketDispersion.distanceRocketPropellant = (
                setting["distanceRocketPropellant"]
                if "distanceRocketPropellant" in setting
                else rocketDispersion.distanceRocketPropellant
            )
            rocketDispersion.powerOffDrag = (
                setting["powerOffDrag"]
                if "powerOffDrag" in setting
                else rocketDispersion.powerOffDrag
            )
            rocketDispersion.powerOnDrag = (
                setting["powerOnDrag"]
                if "powerOnDrag" in setting
                else rocketDispersion.powerOnDrag
            )


            # Add rocket nose, fins and tail
            rocketDispersion.addNose(
                length=setting["noseLength"] if "noseLength" in setting else rocket.noseLength,
                kind=setting["kind"]if "kind" in setting else rocket.kind,
                distanceToCM=setting["noseDistanceToCM"]if "noseLength" in setting else rocket.rtol,
            )
            rocketDispersion.addFins(
                n=setting["n"] if "n" in setting else rocket.n,
                rootChord=setting["finRootChord"] if "finRootChord" in setting else rocket.finRootChord,
                tipChord=setting["finTipChord"] if "finTipChord" in setting else rocket.finTipChord,
                span=setting["finSpan"] if "finSpan" in setting else rocket.finSpan,
                distanceToCM=setting["finDistanceToCM"] if "finDistanceToCM" in setting else rocket.finDistanceToCM,
                radius=setting["radius"] if "radius" in setting else rocket.radius, 
                airfoil=setting["airfoil"] if "airfoil" in setting else rocket.airfoil
            )
            rocketDispersion.addTail(
               topRadius =setting["topRadius"] if "topRadius" in setting else rocket.topRadius,
               bottomRadius=setting["bottomRadius"] if "bottomRadius" in setting else rocket.bottomRadius, 
               length=setting["length"] if "length" in setting else rocket.length, 
               distanceToCM=setting["distanceToCM"] if "distanceToCM" in setting else rocket.distanceToCM
            )
           
            # Add parachute
            rocketDispersion.addParachute(
                name=setting["name"] if "name" in setting else rocket.name,
                CdS=setting["CdS"] if "CdS" in setting else rocket.CdS,
                trigger=setting["trigger"] if "trigger" in setting else rocket.trigger,
                samplingRate=setting["samplingRate"] if "samplingRate" in setting else rocket.samplingRate,
                lag=setting["lag_rec"] if "lag_rec" in setting else rocket.lag_rec + setting["lag_se"] if "lag_se" in setting else rocket.lag_se,
                noise=setting["noise"] if "noise" in setting else rocket.noise,
            )

            # Run trajectory simulation
            try:
                TestFlight = Flight(
                    rocket=rocketDispersion,
                    environment=envDispersion,
                    inclination=envDispersion.inclination,
                    heading=envDispersion.heading,
                    initialSolution=setting["initialSolution"] if "initialSolution" in setting else flight.initialSolution,
                    terminateOnApogee=setting["terminateOnApogee"] if "terminateOnApogee" in setting else flight.terminateOnApogee,
                    maxTime=setting["maxTime"] if "maxTime" in setting else flight.maxTime,
                    maxTimeStep=setting["maxTimeStep"] if "maxTimeStep" in setting else flight.maxTimeStep,
                    minTimeStep=setting["minTimeStep"] if "minTimeStep" in setting else flight.minTimeStep,
                    rtol=setting["rtol"] if "rtol" in setting else flight.rtol,
                    atol=setting["atol"] if "atol" in setting else flight.atol,
                    timeOvershoot=setting["timeOvershoot"] if "timeOvershoot" in setting else flight.timeOvershoot,
                    verbose=setting["verbose"] if "verbose" in setting else flight.verbose,
                )
                export_flight_data(setting, TestFlight, process_time() - start_time)
            except Exception as E:
                print(E)
                export_flight_error(setting)

            # Register time
            out.update(
                f"Curent iteration: {i:06d} | Average Time per Iteration: {(process_time() - initial_cpu_time)/i:2.6f} s"
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
            "parachuteTriggerTime": [],
            "parachuteInflatedTime": [],
            "parachuteInflatedVelocity": [],
            "executionTime": [],
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

        # Print number of flights simulated
        return dispersion_results

    def plotOutOfRailTime(self, dispersion_results):
        print(
            f'Out of Rail Time -         Mean Value: {np.mean(dispersion_results["outOfRailTime"]):0.3f} s'
        )
        print(
            f'Out of Rail Time - Standard Deviation: {np.std(dispersion_results["outOfRailTime"]):0.3f} s'
        )

        plt.figure()
        plt.hist(dispersion_results["outOfRailTime"], bins=int(N**0.5))
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
        plt.hist(dispersion_results["outOfRailVelocity"], bins=int(N**0.5))
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
        plt.hist(dispersion_results["impactTime"], bins=int(N**0.5))
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
        plt.hist(dispersion_results["apogeeAltitude"], bins=int(N**0.5))
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
        plt.hist(dispersion_results["apogeeX"], bins=int(N**0.5))
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
        plt.hist(dispersion_results["apogeeY"], bins=int(N**0.5))
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
        plt.hist(dispersion_results["impactTime"], bins=int(N**0.5))
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
        plt.hist(dispersion_results["impactX"], bins=int(N**0.5))
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
        plt.hist(dispersion_results["impactY"], bins=int(N**0.5))
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
        plt.hist(dispersion_results["impactVelocity"], bins=int(N**0.5))
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
            bins=int(N**0.5),
        )
        plt.hist(
            dispersion_results["outOfRailStaticMargin"],
            label="Out of Rail",
            bins=int(N**0.5),
        )
        plt.hist(
            dispersion_results["finalStaticMargin"], label="Final", bins=int(N**0.5)
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
        plt.hist(dispersion_results["maxVelocity"], bins=int(N**0.5))
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

    def plotParachuteTriggerTime(self, dispersion_results):
        print(
            f'Parachute Trigger Time -         Mean Value: {np.mean(dispersion_results["parachuteTriggerTime"]):0.3f} s'
        )
        print(
            f'Parachute Trigger Time - Standard Deviation: {np.std(dispersion_results["parachuteTriggerTime"]):0.3f} s'
        )

        plt.figure()
        plt.hist(dispersion_results["parachuteTriggerTime"], bins=int(N**0.5))
        plt.title("Parachute Trigger Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Number of Occurences")
        plt.show()

        return None

    def plotParachuteFullyInflatedTime(self, dispersion_results):
        print(
            f'Parachute Fully Inflated Time -         Mean Value: {np.mean(dispersion_results["parachuteInflatedTime"]):0.3f} s'
        )
        print(
            f'Parachute Fully Inflated Time - Standard Deviation: {np.std(dispersion_results["parachuteInflatedTime"]):0.3f} s'
        )

        plt.figure()
        plt.hist(dispersion_results["parachuteInflatedTime"], bins=int(N**0.5))
        plt.title("Parachute Fully Inflated Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Number of Occurences")
        plt.show()

        return None

    def plotParachuteFullyVelocity(self, dispersion_results):
        print(
            f'Drogue Parachute Fully Inflated Velocity -         Mean Value: {np.mean(dispersion_results["parachuteInflatedVelocity"]):0.3f} m/s'
        )
        print(
            f'Drogue Parachute Fully Inflated Velocity - Standard Deviation: {np.std(dispersion_results["parachuteInflatedVelocity"]):0.3f} m/s'
        )

        plt.figure()
        plt.hist(dispersion_results["parachuteInflatedVelocity"], bins=int(N**0.5))
        plt.title("Drogue Parachute Fully Inflated Velocity")
        plt.xlabel("Velocity m/s)")
        plt.ylabel("Number of Occurences")
        plt.show()

        return None

    def plotEllipses(self):

        # Import background map
        img = imread("dispersion_analysis_inputs/Valetudo_basemap_final.jpg")

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
        plt.scatter(
            411.89,
            -61.07,
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
        plt.imshow(img, zorder=0, extent=[-1000 - dx, 1000 - dx, -1000 - dy, 1000 - dy])
        plt.axhline(0, color="black", linewidth=0.5)
        plt.axvline(0, color="black", linewidth=0.5)
        plt.xlim(-100, 700)
        plt.ylim(-300, 300)

        # Save plot and show result
        plt.savefig(str(filename) + ".pdf", bbox_inches="tight", pad_inches=0)
        plt.savefig(str(filename) + ".svg", bbox_inches="tight", pad_inches=0)
        plt.show()


    def plotOutOfRailVelocityByMass(self, minMass, maxMass, numPoints):
        def outOfRailVelocity(mass):
            self.rocket.mass = mass
            newMassFlight = Flight(
                self.rocket,
                self.env,
                inclination=80,
                heading=90,
                initialSolution=None,
                terminateOnApogee=False,
                maxTime=600,
                maxTimeStep=np.inf,
                minTimeStep=0,
                rtol=0.000001,
                atol=6 * [0.001] + 4 * [0.000001] + 3 * [0.001],
                timeOvershoot=True,
                verbose=False,
            )
            return newMassFlight.outOfRailVelocity

        outOfRailVelocityByMass = Function(
            outOfRailVelocity, inputs="Mass (kg)", outputs="Out of Rail Velocity (m/s)"
        )
        outOfRailVelocityByMass.plot(minMass, maxMass, numPoints)

        return None
