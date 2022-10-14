# -*- coding: utf-8 -*-

__author__ = "Mateus Stano Junqueira, Sofia Lopes Suesdek Rocha, Guilherme Fernandes Alves, Bruno Abdulklech Sorban"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"


import math
import traceback
import warnings
from time import process_time, time

import matplotlib.pyplot as plt
import numpy as np
import simplekml
from imageio import imread
from IPython.display import display
from matplotlib.patches import Ellipse
from numpy.random import *


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
