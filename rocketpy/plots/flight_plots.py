# -*- coding: utf-8 -*-

import time
import warnings

import matplotlib.pyplot as plt
import numpy as np

__author__ = "Guilherme Fernandes Alves"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

## Future improvements:
# Allow user to choose the units of the plots
# Allow user to choose the color pallet of the plots
# Add masses plots since it can vary significantly for multi-stage rockets


class flight_plots:
    """class to plot flight data
    Here you also can:
    - Print important information about the flight
    - See animations of the flight
    - Compare plots from different flights
    - Compare flights from different rocket simulators
    """

    def __init__(
        self,
        trajectory_list,
        names_list=None,
    ):
        """_summary_

        Parameters
        ----------
        trajectory_list : list
            List of Flight objects
        names_list : list, optional
            (the default is None, which [default_description])

        Returns
        -------
        _type_
            _description_
        """
        self.names_list = names_list

        if isinstance(trajectory_list, list):
            self.trajectory_list = trajectory_list
        # elif isinstance(trajectory_list, Flight):
        #     self.trajectory_list = [trajectory_list]
        #     self.names_list = [trajectory_list.__name__]
        else:
            raise TypeError("trajectory_list must be a list of Flight objects")

        self.names_list = (
            [("Trajectory " + str(i + 1)) for i in range(len(self.trajectory_list))]
            if names_list == None
            else names_list
        )

    # Start definition of Prints methods, no plots here for now

    def printInitialConditionsData(self):
        """Prints all initial conditions data available about the flights passed
        by the trajectory_list.

        Parameters
        ----------
        None

        Return
        ------
        None
        """

        for index, flight in enumerate(self.trajectory_list):

            print("\nInitial Conditions for Flight: ", self.names_list[index])

            # Post-process results
            if flight.postProcessed is False:
                flight.postProcess()
            print(
                "Position - x: {:.2f} m | y: {:.2f} m | z: {:.2f} m".format(
                    flight.x(0), flight.y(0), flight.z(0)
                )
            )
            print(
                "Velocity - Vx: {:.2f} m/s | Vy: {:.2f} m/s | Vz: {:.2f} m/s".format(
                    flight.vx(0), flight.vy(0), flight.vz(0)
                )
            )
            print(
                "Attitude - e0: {:.3f} | e1: {:.3f} | e2: {:.3f} | e3: {:.3f}".format(
                    flight.e0(0), flight.e1(0), flight.e2(0), flight.e3(0)
                )
            )
            print(
                "Euler Angles - Spin φ : {:.2f}° | Nutation θ: {:.2f}° | Precession ψ: {:.2f}°".format(
                    flight.phi(0), flight.theta(0), flight.psi(0)
                )
            )
            print(
                "Angular Velocity - ω1: {:.2f} rad/s | ω2: {:.2f} rad/s| ω3: {:.2f} rad/s".format(
                    flight.w1(0), flight.w2(0), flight.w3(0)
                )
            )

        return None

    def printNumericalIntegrationSettings(self):
        """Prints out the Numerical Integration settings available about the
        flights passed by the trajectory_list.

        Parameters
        ----------
        None

        Return
        ------
        None
        """
        for index, flight in enumerate(self.trajectory_list):
            print(
                "\nNumerical Integration Settings of Flight: ", self.names_list[index]
            )
            print("Maximum Allowed Flight Time: {:f} s".format(flight.maxTime))
            print("Maximum Allowed Time Step: {:f} s".format(flight.maxTimeStep))
            print("Minimum Allowed Time Step: {:e} s".format(flight.minTimeStep))
            print("Relative Error Tolerance: ", flight.rtol)
            print("Absolute Error Tolerance: ", flight.atol)
            print("Allow Event Overshoot: ", flight.timeOvershoot)
            print("Terminate Simulation on Apogee: ", flight.terminateOnApogee)
            print("Number of Time Steps Used: ", len(flight.timeSteps))
            print(
                "Number of Derivative Functions Evaluation: ",
                sum(flight.functionEvaluationsPerTimeStep),
            )
            print(
                "Average Function Evaluations per Time Step: {:3f}".format(
                    sum(flight.functionEvaluationsPerTimeStep) / len(flight.timeSteps)
                )
            )

        return None

    def printSurfaceWindConditions(self):
        """Prints out the Surface Wind Conditions available about the flights
        passed by the trajectory_list.

        Returns
        -------
        None
        """
        for index, flight in enumerate(self.trajectory_list):
            if flight.postProcessed is False:
                flight.postProcess()
            print("\nSurface Wind Conditions of Flight: ", self.names_list[index])
            print(
                "Frontal Surface Wind Speed: {:.2f} m/s".format(
                    flight.frontalSurfaceWind
                )
            )
            print(
                "Lateral Surface Wind Speed: {:.2f} m/s".format(
                    flight.lateralSurfaceWind
                )
            )

        return None

    def printLaunchRailConditions(self):
        """Prints out the Launch Rail Conditions available about the flights
        passed by the trajectory_list.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        for index, flight in enumerate(self.trajectory_list):
            print("\nLaunch Rail Orientation of Flight: ", self.names_list[index])
            print("Launch Rail Inclination: {:.2f}°".format(flight.inclination))
            print("Launch Rail Heading: {:.2f}°".format(flight.heading))
        return None

    def printOutOfRailConditions(self):
        """Prints out the Out of Rail Conditions available about the flights
        passed by the trajectory_list.

        Returns
        -------
        None
        """
        for index, flight in enumerate(self.trajectory_list):
            if flight.postProcessed is False:
                flight.postProcess()
            print("\nRail Departure State of Flight: ", self.names_list[index])
            print("Rail Departure Time: {:.3f} s".format(flight.outOfRailTime))
            print(
                "Rail Departure Velocity: {:.3f} m/s".format(flight.outOfRailVelocity)
            )
            print(
                "Rail Departure Static Margin: {:.3f} c".format(
                    flight.staticMargin(flight.outOfRailTime)
                )
            )
            print(
                "Rail Departure Angle of Attack: {:.3f}°".format(
                    flight.angleOfAttack(flight.outOfRailTime)
                )
            )
            print(
                "Rail Departure Thrust-Weight Ratio: {:.3f}".format(
                    flight.rocket.thrustToWeight(flight.outOfRailTime)
                )
            )
            print(
                "Rail Departure Reynolds Number: {:.3e}".format(
                    flight.ReynoldsNumber(flight.outOfRailTime)
                )
            )

        return None

    def printBurnOutConditions(self):
        """Prints out the Burn Out Conditions available about the flights
        passed by the trajectory_list.

        Returns
        -------
        None
        """
        for index, flight in enumerate(self.trajectory_list):
            if flight.postProcessed is False:
                flight.postProcess()
            print("\nBurnOut State of Flight: ", self.names_list[index])
            print("BurnOut time: {:.3f} s".format(flight.rocket.motor.burnOutTime))
            print(
                "Altitude at burnOut: {:.3f} m (AGL)".format(
                    flight.z(flight.rocket.motor.burnOutTime) - flight.env.elevation
                )
            )
            print(
                "Rocket velocity at burnOut: {:.3f} m/s".format(
                    flight.speed(flight.rocket.motor.burnOutTime)
                )
            )
            print(
                "Freestream velocity at burnOut: {:.3f} m/s".format(
                    (
                        flight.streamVelocityX(flight.rocket.motor.burnOutTime) ** 2
                        + flight.streamVelocityY(flight.rocket.motor.burnOutTime) ** 2
                        + flight.streamVelocityZ(flight.rocket.motor.burnOutTime) ** 2
                    )
                    ** 0.5
                )
            )
            print(
                "Mach Number at burnOut: {:.3f}".format(
                    flight.MachNumber(flight.rocket.motor.burnOutTime)
                )
            )
            print(
                "Kinetic energy at burnOut: {:.3e} J".format(
                    flight.kineticEnergy(flight.rocket.motor.burnOutTime)
                )
            )

        return None

    def printApogeeConditions(self):
        """Prints out the Apogee Conditions available about the flights
        passed by the trajectory_list.

        Returns
        -------
        None
        """
        for index, flight in enumerate(self.trajectory_list):
            if flight.postProcessed is False:
                flight.postProcess()
            print("\nApogee State of Flight: ", self.names_list[index])
            print(
                "Apogee Altitude: {:.3f} m (ASL) | {:.3f} m (AGL)".format(
                    flight.apogee, flight.apogee - flight.env.elevation
                )
            )
            print("Apogee Time: {:.3f} s".format(flight.apogeeTime))
            print(
                "Apogee Freestream Speed: {:.3f} m/s".format(
                    flight.apogeeFreestreamSpeed
                )
            )

        return None

    def printEventsRegistered(self):
        """Prints out the Events Registered available about the flights
        passed by the trajectory_list.

        Returns
        -------
        None
        """
        for index, flight in enumerate(self.trajectory_list):
            if flight.postProcessed is False:
                flight.postProcess()
            print("\nParachute Events of Flight: ", self.names_list[index])
            if len(flight.parachuteEvents) == 0:
                print("No Parachute Events Were Triggered.")
            for event in flight.parachuteEvents:
                triggerTime = event[0]
                parachute = event[1]
                openTime = triggerTime + parachute.lag
                velocity = flight.freestreamSpeed(openTime)
                altitude = flight.z(openTime)
                name = parachute.name.title()
                print(name + " Ejection Triggered at: {:.3f} s".format(triggerTime))
                print(name + " Parachute Inflated at: {:.3f} s".format(openTime))
                print(
                    name
                    + " Parachute Inflated with Freestream Speed of: {:.3f} m/s".format(
                        velocity
                    )
                )
                print(
                    name
                    + " Parachute Inflated at Height of: {:.3f} m (AGL)".format(
                        altitude - flight.env.elevation
                    )
                )
        return None

    def printImpactConditions(self):
        """Prints out the Impact Conditions available about the flights
        passed by the trajectory_list.

        Returns
        -------
        None
        """
        for index, flight in enumerate(self.trajectory_list):
            if flight.postProcessed is False:
                flight.postProcess()
            if len(flight.impactState) != 0:
                print("\nImpact Conditions of Flight: ", self.names_list[index])
                print("X Impact: {:.3f} m".format(flight.xImpact))
                print("Y Impact: {:.3f} m".format(flight.yImpact))
                print("Time of Impact: {:.3f} s".format(flight.tFinal))
                print("Velocity at Impact: {:.3f} m/s".format(flight.impactVelocity))
            elif flight.terminateOnApogee is False:
                print("End of Simulation of Flight: ", flight.names_list[index])
                print("Time: {:.3f} s".format(flight.solution[-1][0]))
                print("Altitude: {:.3f} m".format(flight.solution[-1][3]))

        return None

    def printMaximumValues(self):
        """Prints out the Maximum Values available about the flights
        passed by the trajectory_list.

        Returns
        -------
        None
        """
        for index, flight in enumerate(self.trajectory_list):
            print("\nMaximum Values of Flight: ", self.names_list[index])
            print(
                "Maximum Speed: {:.3f} m/s at {:.2f} s".format(
                    flight.maxSpeed, flight.maxSpeedTime
                )
            )
            print(
                "Maximum Mach Number: {:.3f} Mach at {:.2f} s".format(
                    flight.maxMachNumber, flight.maxMachNumberTime
                )
            )
            print(
                "Maximum Reynolds Number: {:.3e} at {:.2f} s".format(
                    flight.maxReynoldsNumber, flight.maxReynoldsNumberTime
                )
            )
            print(
                "Maximum Dynamic Pressure: {:.3e} Pa at {:.2f} s".format(
                    flight.maxDynamicPressure, flight.maxDynamicPressureTime
                )
            )
            print(
                "Maximum Acceleration: {:.3f} m/s² at {:.2f} s".format(
                    flight.maxAcceleration, flight.maxAccelerationTime
                )
            )
            print(
                "Maximum Gs: {:.3f} g at {:.2f} s".format(
                    flight.maxAcceleration / flight.env.g, flight.maxAccelerationTime
                )
            )
            print(
                "Maximum Upper Rail Button Normal Force: {:.3f} N".format(
                    flight.maxRailButton1NormalForce
                )
            )
            print(
                "Maximum Upper Rail Button Shear Force: {:.3f} N".format(
                    flight.maxRailButton1ShearForce
                )
            )
            print(
                "Maximum Lower Rail Button Normal Force: {:.3f} N".format(
                    flight.maxRailButton2NormalForce
                )
            )
            print(
                "Maximum Lower Rail Button Shear Force: {:.3f} N".format(
                    flight.maxRailButton2ShearForce
                )
            )
        return None

    # Start definition of 'basic' plots methods, the traditional RocketPy plots

    def plot3dTrajectory(self, savefig=False):
        """Plot a 3D graph of the trajectory

        Parameters
        ----------
        savefig: str, optional
            If a string is passed, the figure will be saved with the name passed.
            Default is False.

        Return
        ------
        None
        """

        warnings.warn(
            "plot3dTrajectory is going to be deprecated, use compareFlightTrajectories3D instead.",
        )
        self.compareFlightTrajectories3D(legend=False, savefig=savefig)

        return None

    def plotLinearKinematicsData(self):
        """Prints out all Kinematics graphs available about the Flight

        Parameters
        ----------
        None

        Return
        ------
        None
        """

        for index, flight in enumerate(self.trajectory_list):

            if flight.postProcessed is False:
                flight.postProcess()

            # Velocity and acceleration plots
            fig2 = plt.figure(figsize=(9, 12))
            fig2.suptitle(
                "Linear Kinematics Data of Flight: {}".format(self.names_list[index])
            )

            ax1 = plt.subplot(414)
            ax1.plot(flight.vx[:, 0], flight.vx[:, 1], color="#ff7f0e")
            ax1.set_xlim(0, flight.tFinal)
            ax1.set_title("Velocity X | Acceleration X")
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Velocity X (m/s)", color="#ff7f0e")
            ax1.tick_params("y", colors="#ff7f0e")
            ax1.grid(True)

            ax1up = ax1.twinx()
            ax1up.plot(flight.ax[:, 0], flight.ax[:, 1], color="#1f77b4")
            ax1up.set_ylabel("Acceleration X (m/s²)", color="#1f77b4")
            ax1up.tick_params("y", colors="#1f77b4")

            ax2 = plt.subplot(413)
            ax2.plot(flight.vy[:, 0], flight.vy[:, 1], color="#ff7f0e")
            ax2.set_xlim(0, flight.tFinal)
            ax2.set_title("Velocity Y | Acceleration Y")
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Velocity Y (m/s)", color="#ff7f0e")
            ax2.tick_params("y", colors="#ff7f0e")
            ax2.grid(True)

            ax2up = ax2.twinx()
            ax2up.plot(flight.ay[:, 0], flight.ay[:, 1], color="#1f77b4")
            ax2up.set_ylabel("Acceleration Y (m/s²)", color="#1f77b4")
            ax2up.tick_params("y", colors="#1f77b4")

            ax3 = plt.subplot(412)
            ax3.plot(flight.vz[:, 0], flight.vz[:, 1], color="#ff7f0e")
            ax3.set_xlim(0, flight.tFinal)
            ax3.set_title("Velocity Z | Acceleration Z")
            ax3.set_xlabel("Time (s)")
            ax3.set_ylabel("Velocity Z (m/s)", color="#ff7f0e")
            ax3.tick_params("y", colors="#ff7f0e")
            ax3.grid(True)

            ax3up = ax3.twinx()
            ax3up.plot(flight.az[:, 0], flight.az[:, 1], color="#1f77b4")
            ax3up.set_ylabel("Acceleration Z (m/s²)", color="#1f77b4")
            ax3up.tick_params("y", colors="#1f77b4")

            ax4 = plt.subplot(411)
            ax4.plot(flight.speed[:, 0], flight.speed[:, 1], color="#ff7f0e")
            ax4.set_xlim(0, flight.tFinal)
            ax4.set_title("Velocity Magnitude | Acceleration Magnitude")
            ax4.set_xlabel("Time (s)")
            ax4.set_ylabel("Velocity (m/s)", color="#ff7f0e")
            ax4.tick_params("y", colors="#ff7f0e")
            ax4.grid(True)

            ax4up = ax4.twinx()
            ax4up.plot(
                flight.acceleration[:, 0], flight.acceleration[:, 1], color="#1f77b4"
            )
            ax4up.set_ylabel("Acceleration (m/s²)", color="#1f77b4")
            ax4up.tick_params("y", colors="#1f77b4")

            plt.subplots_adjust(hspace=0.5)
            plt.show()
        return None

    def plotAttitudeData(self):
        """Prints out all Angular position graphs available about the Flight

        Parameters
        ----------
        None

        Return
        ------
        None
        """

        for index, flight in enumerate(self.trajectory_list):

            # Post-process results
            if flight.postProcessed is False:
                flight.postProcess()

            # Get index of time before parachute event
            if len(flight.parachuteEvents) > 0:
                eventTime = (
                    flight.parachuteEvents[0][0] + flight.parachuteEvents[0][1].lag
                )
                eventTimeIndex = np.nonzero(flight.x[:, 0] == eventTime)[0][0]
            else:
                eventTime = flight.tFinal
                eventTimeIndex = -1

            # Angular position plots
            fig3 = plt.figure(figsize=(9, 12))
            fig3.suptitle("Euler Angles of Flight: {}".format(self.names_list[index]))

            ax1 = plt.subplot(411)
            ax1.plot(flight.e0[:, 0], flight.e0[:, 1], label="$e_0$")
            ax1.plot(flight.e1[:, 0], flight.e1[:, 1], label="$e_1$")
            ax1.plot(flight.e2[:, 0], flight.e2[:, 1], label="$e_2$")
            ax1.plot(flight.e3[:, 0], flight.e3[:, 1], label="$e_3$")
            ax1.set_xlim(0, eventTime)
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Euler Parameters")
            ax1.set_title("Euler Parameters")
            ax1.legend()
            ax1.grid(True)

            ax2 = plt.subplot(412)
            ax2.plot(flight.psi[:, 0], flight.psi[:, 1])
            ax2.set_xlim(0, eventTime)
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("ψ (°)")
            ax2.set_title("Euler Precession Angle")
            ax2.grid(True)

            ax3 = plt.subplot(413)
            ax3.plot(flight.theta[:, 0], flight.theta[:, 1], label="θ - Nutation")
            ax3.set_xlim(0, eventTime)
            ax3.set_xlabel("Time (s)")
            ax3.set_ylabel("θ (°)")
            ax3.set_title("Euler Nutation Angle")
            ax3.grid(True)

            ax4 = plt.subplot(414)
            ax4.plot(flight.phi[:, 0], flight.phi[:, 1], label="φ - Spin")
            ax4.set_xlim(0, eventTime)
            ax4.set_xlabel("Time (s)")
            ax4.set_ylabel("φ (°)")
            ax4.set_title("Euler Spin Angle")
            ax4.grid(True)

            plt.subplots_adjust(hspace=0.5)
            plt.show()

        return None

    def plotFlightPathAngleData(self):
        """Prints out Flight path and Rocket Attitude angle graphs available
        about the Flight

        Parameters
        ----------
        None

        Return
        ------
        None
        """
        for index, flight in enumerate(self.trajectory_list):

            # Post-process results
            if flight.postProcessed is False:
                flight.postProcess()

            # Get index of time before parachute event
            if len(flight.parachuteEvents) > 0:
                eventTime = (
                    flight.parachuteEvents[0][0] + flight.parachuteEvents[0][1].lag
                )
                eventTimeIndex = np.nonzero(flight.x[:, 0] == eventTime)[0][0]
            else:
                eventTime = flight.tFinal
                eventTimeIndex = -1

            # Path, Attitude and Lateral Attitude Angle
            # Angular position plots
            fig5 = plt.figure(figsize=(9, 6))
            fig5.suptitle(
                "Flight Path and Attitude Data of Flight: {}".format(
                    self.names_list[index]
                )
            )

            ax1 = plt.subplot(211)
            ax1.plot(
                flight.pathAngle[:, 0],
                flight.pathAngle[:, 1],
                label="Flight Path Angle",
            )
            ax1.plot(
                flight.attitudeAngle[:, 0],
                flight.attitudeAngle[:, 1],
                label="Rocket Attitude Angle",
            )
            ax1.set_xlim(0, eventTime)
            ax1.legend()
            ax1.grid(True)
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Angle (°)")
            ax1.set_title("Flight Path and Attitude Angle")

            ax2 = plt.subplot(212)
            ax2.plot(
                flight.lateralAttitudeAngle[:, 0], flight.lateralAttitudeAngle[:, 1]
            )
            ax2.set_xlim(0, eventTime)
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Lateral Attitude Angle (°)")
            ax2.set_title("Lateral Attitude Angle")
            ax2.grid(True)

            plt.subplots_adjust(hspace=0.5)
            plt.show()

        return None

    def plotAngularKinematicsData(self):
        """Prints out all Angular velocity and acceleration graphs available
        about the Flight

        Parameters
        ----------
        None

        Return
        ------
        None
        """

        for index, flight in enumerate(self.trajectory_list):
            # Post-process results
            if flight.postProcessed is False:
                flight.postProcess()

            # Get index of time before parachute event
            if len(flight.parachuteEvents) > 0:
                eventTime = (
                    flight.parachuteEvents[0][0] + flight.parachuteEvents[0][1].lag
                )
                eventTimeIndex = np.nonzero(flight.x[:, 0] == eventTime)[0][0]
            else:
                eventTime = flight.tFinal
                eventTimeIndex = -1

            # Angular velocity and acceleration plots
            fig4 = plt.figure(figsize=(9, 9))
            fig4.suptitle(
                "Angular Kinematics Data of Flight: {}".format(self.names_list[index])
            )

            ax1 = plt.subplot(311)
            ax1.plot(flight.w1[:, 0], flight.w1[:, 1], color="#ff7f0e")
            ax1.set_xlim(0, eventTime)
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel(r"Angular Velocity - ${\omega_1}$ (rad/s)", color="#ff7f0e")
            ax1.set_title(
                r"Angular Velocity ${\omega_1}$ | Angular Acceleration ${\alpha_1}$"
            )
            ax1.tick_params("y", colors="#ff7f0e")
            ax1.grid(True)

            ax1up = ax1.twinx()
            ax1up.plot(flight.alpha1[:, 0], flight.alpha1[:, 1], color="#1f77b4")
            ax1up.set_ylabel(
                r"Angular Acceleration - ${\alpha_1}$ (rad/s²)", color="#1f77b4"
            )
            ax1up.tick_params("y", colors="#1f77b4")

            ax2 = plt.subplot(312)
            ax2.plot(flight.w2[:, 0], flight.w2[:, 1], color="#ff7f0e")
            ax2.set_xlim(0, eventTime)
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel(r"Angular Velocity - ${\omega_2}$ (rad/s)", color="#ff7f0e")
            ax2.set_title(
                r"Angular Velocity ${\omega_2}$ | Angular Acceleration ${\alpha_2}$"
            )
            ax2.tick_params("y", colors="#ff7f0e")
            ax2.grid(True)

            ax2up = ax2.twinx()
            ax2up.plot(flight.alpha2[:, 0], flight.alpha2[:, 1], color="#1f77b4")
            ax2up.set_ylabel(
                r"Angular Acceleration - ${\alpha_2}$ (rad/s²)", color="#1f77b4"
            )
            ax2up.tick_params("y", colors="#1f77b4")

            ax3 = plt.subplot(313)
            ax3.plot(flight.w3[:, 0], flight.w3[:, 1], color="#ff7f0e")
            ax3.set_xlim(0, eventTime)
            ax3.set_xlabel("Time (s)")
            ax3.set_ylabel(r"Angular Velocity - ${\omega_3}$ (rad/s)", color="#ff7f0e")
            ax3.set_title(
                r"Angular Velocity ${\omega_3}$ | Angular Acceleration ${\alpha_3}$"
            )
            ax3.tick_params("y", colors="#ff7f0e")
            ax3.grid(True)

            ax3up = ax3.twinx()
            ax3up.plot(flight.alpha3[:, 0], flight.alpha3[:, 1], color="#1f77b4")
            ax3up.set_ylabel(
                r"Angular Acceleration - ${\alpha_3}$ (rad/s²)", color="#1f77b4"
            )
            ax3up.tick_params("y", colors="#1f77b4")

            plt.subplots_adjust(hspace=0.5)
            plt.show()

        return None

    def plotTrajectoryForceData(self):
        """Prints out all Forces and Moments graphs available about the Flight

        Parameters
        ----------
        None

        Return
        ------
        None
        """
        for index, flight in enumerate(self.trajectory_list):
            # Post-process results
            if flight.postProcessed is False:
                flight.postProcess()

            # Get index of out of rail time
            outOfRailTimeIndexes = np.nonzero(flight.x[:, 0] == flight.outOfRailTime)
            outOfRailTimeIndex = (
                -1 if len(outOfRailTimeIndexes) == 0 else outOfRailTimeIndexes[0][0]
            )

            # Get index of time before parachute event
            if len(flight.parachuteEvents) > 0:
                eventTime = (
                    flight.parachuteEvents[0][0] + flight.parachuteEvents[0][1].lag
                )
                eventTimeIndex = np.nonzero(flight.x[:, 0] == eventTime)[0][0]
            else:
                eventTime = flight.tFinal
                eventTimeIndex = -1

            # Rail Button Forces
            if flight.rocket.railButtons is not None:
                fig6 = plt.figure(figsize=(9, 6))
                fig6.suptitle(
                    "Rail Button Forces of Flight: {}".format(self.names_list[index])
                )

                ax1 = plt.subplot(211)
                ax1.plot(
                    flight.railButton1NormalForce[:outOfRailTimeIndex, 0],
                    flight.railButton1NormalForce[:outOfRailTimeIndex, 1],
                    label="Upper Rail Button",
                )
                ax1.plot(
                    flight.railButton2NormalForce[:outOfRailTimeIndex, 0],
                    flight.railButton2NormalForce[:outOfRailTimeIndex, 1],
                    label="Lower Rail Button",
                )
                ax1.set_xlim(
                    0,
                    flight.outOfRailTime if flight.outOfRailTime > 0 else flight.tFinal,
                )
                ax1.legend()
                ax1.grid(True)
                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("Normal Force (N)")
                ax1.set_title("Rail Buttons Normal Force")

                ax2 = plt.subplot(212)
                ax2.plot(
                    flight.railButton1ShearForce[:outOfRailTimeIndex, 0],
                    flight.railButton1ShearForce[:outOfRailTimeIndex, 1],
                    label="Upper Rail Button",
                )
                ax2.plot(
                    flight.railButton2ShearForce[:outOfRailTimeIndex, 0],
                    flight.railButton2ShearForce[:outOfRailTimeIndex, 1],
                    label="Lower Rail Button",
                )
                ax2.set_xlim(
                    0,
                    flight.outOfRailTime if flight.outOfRailTime > 0 else flight.tFinal,
                )
                ax2.legend()
                ax2.grid(True)
                ax2.set_xlabel("Time (s)")
                ax2.set_ylabel("Shear Force (N)")
                ax2.set_title("Rail Buttons Shear Force")

                plt.subplots_adjust(hspace=0.5)
                plt.show()

            # Aerodynamic force and moment plots
            fig7 = plt.figure(figsize=(9, 12))
            fig7.suptitle(
                "Aerodynamic Forces and Moments of Flight: {}".format(
                    self.names_list[index]
                )
            )

            ax1 = plt.subplot(411)
            ax1.plot(
                flight.aerodynamicLift[:eventTimeIndex, 0],
                flight.aerodynamicLift[:eventTimeIndex, 1],
                label="Resultant",
            )
            ax1.plot(
                flight.R1[:eventTimeIndex, 0], flight.R1[:eventTimeIndex, 1], label="R1"
            )
            ax1.plot(
                flight.R2[:eventTimeIndex, 0], flight.R2[:eventTimeIndex, 1], label="R2"
            )
            ax1.set_xlim(0, eventTime)
            ax1.legend()
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Lift Force (N)")
            ax1.set_title("Aerodynamic Lift Resultant Force")
            ax1.grid()

            ax2 = plt.subplot(412)
            ax2.plot(
                flight.aerodynamicDrag[:eventTimeIndex, 0],
                flight.aerodynamicDrag[:eventTimeIndex, 1],
            )
            ax2.set_xlim(0, eventTime)
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Drag Force (N)")
            ax2.set_title("Aerodynamic Drag Force")
            ax2.grid()

            ax3 = plt.subplot(413)
            ax3.plot(
                flight.aerodynamicBendingMoment[:eventTimeIndex, 0],
                flight.aerodynamicBendingMoment[:eventTimeIndex, 1],
                label="Resultant",
            )
            ax3.plot(
                flight.M1[:eventTimeIndex, 0], flight.M1[:eventTimeIndex, 1], label="M1"
            )
            ax3.plot(
                flight.M2[:eventTimeIndex, 0], flight.M2[:eventTimeIndex, 1], label="M2"
            )
            ax3.set_xlim(0, eventTime)
            ax3.legend()
            ax3.set_xlabel("Time (s)")
            ax3.set_ylabel("Bending Moment (N m)")
            ax3.set_title("Aerodynamic Bending Resultant Moment")
            ax3.grid()

            ax4 = plt.subplot(414)
            ax4.plot(
                flight.aerodynamicSpinMoment[:eventTimeIndex, 0],
                flight.aerodynamicSpinMoment[:eventTimeIndex, 1],
            )
            ax4.set_xlim(0, eventTime)
            ax4.set_xlabel("Time (s)")
            ax4.set_ylabel("Spin Moment (N m)")
            ax4.set_title("Aerodynamic Spin Moment")
            ax4.grid()

            plt.subplots_adjust(hspace=0.5)
            plt.show()

        return None

    def plotEnergyData(self):
        """Prints out all Energy components graphs available about the Flight

        Returns
        -------
        None
        """
        for index, flight in enumerate(self.trajectory_list):

            # Post-process results
            if flight.postProcessed is False:
                flight.postProcess()

            # Get index of out of rail time
            outOfRailTimeIndexes = np.nonzero(flight.x[:, 0] == flight.outOfRailTime)
            outOfRailTimeIndex = (
                -1 if len(outOfRailTimeIndexes) == 0 else outOfRailTimeIndexes[0][0]
            )

            # Get index of time before parachute event
            if len(flight.parachuteEvents) > 0:
                eventTime = (
                    flight.parachuteEvents[0][0] + flight.parachuteEvents[0][1].lag
                )
                eventTimeIndex = np.nonzero(flight.x[:, 0] == eventTime)[0][0]
            else:
                eventTime = flight.tFinal
                eventTimeIndex = -1

            fig8 = plt.figure(figsize=(9, 9))
            fig8.suptitle(
                "Energy Components of Flight: {}".format(self.names_list[index])
            )

            ax1 = plt.subplot(411)
            ax1.plot(
                flight.kineticEnergy[:, 0],
                flight.kineticEnergy[:, 1],
                label="Kinetic Energy",
            )
            ax1.plot(
                flight.rotationalEnergy[:, 0],
                flight.rotationalEnergy[:, 1],
                label="Rotational Energy",
            )
            ax1.plot(
                flight.translationalEnergy[:, 0],
                flight.translationalEnergy[:, 1],
                label="Translational Energy",
            )
            ax1.set_xlim(
                0, flight.apogeeTime if flight.apogeeTime != 0.0 else flight.tFinal
            )
            ax1.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
            ax1.set_title("Kinetic Energy Components")
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Energy (J)")

            ax1.legend()
            ax1.grid()

            ax2 = plt.subplot(412)
            ax2.plot(
                flight.totalEnergy[:, 0], flight.totalEnergy[:, 1], label="Total Energy"
            )
            ax2.plot(
                flight.kineticEnergy[:, 0],
                flight.kineticEnergy[:, 1],
                label="Kinetic Energy",
            )
            ax2.plot(
                flight.potentialEnergy[:, 0],
                flight.potentialEnergy[:, 1],
                label="Potential Energy",
            )
            ax2.set_xlim(
                0, flight.apogeeTime if flight.apogeeTime != 0.0 else flight.tFinal
            )
            ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
            ax2.set_title("Total Mechanical Energy Components")
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Energy (J)")
            ax2.legend()
            ax2.grid()

            ax3 = plt.subplot(413)
            ax3.plot(
                flight.thrustPower[:, 0],
                flight.thrustPower[:, 1],
                label="|Thrust Power|",
            )
            ax3.set_xlim(0, flight.rocket.motor.burnOutTime)
            ax3.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
            ax3.set_title("Thrust Absolute Power")
            ax3.set_xlabel("Time (s)")
            ax3.set_ylabel("Power (W)")
            ax3.legend()
            ax3.grid()

            ax4 = plt.subplot(414)
            ax4.plot(
                flight.dragPower[:, 0], -flight.dragPower[:, 1], label="|Drag Power|"
            )
            ax4.set_xlim(
                0, flight.apogeeTime if flight.apogeeTime != 0.0 else flight.tFinal
            )
            ax3.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
            ax4.set_title("Drag Absolute Power")
            ax4.set_xlabel("Time (s)")
            ax4.set_ylabel("Power (W)")
            ax4.legend()
            ax4.grid()

            plt.subplots_adjust(hspace=1)
            plt.show()

        return None

    def plotFluidMechanicsData(self):
        """Prints out a summary of the Fluid Mechanics graphs available about
        the Flight

        Parameters
        ----------
        None

        Return
        ------
        None
        """
        for index, flight in enumerate(self.trajectory_list):
            # Post-process results
            if flight.postProcessed is False:
                flight.postProcess()

            # Get index of out of rail time
            outOfRailTimeIndexes = np.nonzero(flight.x[:, 0] == flight.outOfRailTime)
            outOfRailTimeIndex = (
                -1 if len(outOfRailTimeIndexes) == 0 else outOfRailTimeIndexes[0][0]
            )

            # Trajectory Fluid Mechanics Plots
            fig10 = plt.figure(figsize=(9, 12))
            fig10.suptitle(
                "Fluid Mechanics Components of Flight: {}".format(
                    self.names_list[index]
                )
            )

            ax1 = plt.subplot(411)
            ax1.plot(flight.MachNumber[:, 0], flight.MachNumber[:, 1])
            ax1.set_xlim(0, flight.tFinal)
            ax1.set_title("Mach Number")
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Mach Number")
            ax1.grid()

            ax2 = plt.subplot(412)
            ax2.plot(flight.ReynoldsNumber[:, 0], flight.ReynoldsNumber[:, 1])
            ax2.set_xlim(0, flight.tFinal)
            ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
            ax2.set_title("Reynolds Number")
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Reynolds Number")
            ax2.grid()

            ax3 = plt.subplot(413)
            ax3.plot(
                flight.dynamicPressure[:, 0],
                flight.dynamicPressure[:, 1],
                label="Dynamic Pressure",
            )
            ax3.plot(
                flight.totalPressure[:, 0],
                flight.totalPressure[:, 1],
                label="Total Pressure",
            )
            ax3.plot(
                flight.pressure[:, 0], flight.pressure[:, 1], label="Static Pressure"
            )
            ax3.set_xlim(0, flight.tFinal)
            ax3.legend()
            ax3.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
            ax3.set_title("Total and Dynamic Pressure")
            ax3.set_xlabel("Time (s)")
            ax3.set_ylabel("Pressure (Pa)")
            ax3.grid()

            ax4 = plt.subplot(414)
            ax4.plot(flight.angleOfAttack[:, 0], flight.angleOfAttack[:, 1])
            # Make sure bottom and top limits are different
            if flight.outOfRailTime * flight.angleOfAttack(flight.outOfRailTime) != 0:
                ax4.set_xlim(flight.outOfRailTime, 10 * flight.outOfRailTime + 1)
                ax4.set_ylim(0, flight.angleOfAttack(flight.outOfRailTime))
            ax4.set_title("Angle of Attack")
            ax4.set_xlabel("Time (s)")
            ax4.set_ylabel("Angle of Attack (°)")
            ax4.grid()

            plt.subplots_adjust(hspace=0.5)
            plt.show()

        return None

    def plotStabilityAndControlData(self):
        """Prints out Rocket Stability and Control parameters graphs available
        about the Flight

        Parameters
        ----------
        None

        Return
        ------
        None
        """
        for index, flight in enumerate(self.trajectory_list):
            print(
                "Stability And Control Data of Flight: ".format(self.names_list[index])
            )
            # Post-process results
            if flight.postProcessed is False:
                flight.postProcess()

            fig9 = plt.figure(figsize=(9, 6))
            fig9.suptitle(
                "Stability and Control Components of Flight: {}".format(
                    self.names_list[index]
                )
            )

            ax1 = plt.subplot(211)
            ax1.plot(flight.staticMargin[:, 0], flight.staticMargin[:, 1])
            ax1.set_xlim(0, flight.staticMargin[:, 0][-1])
            ax1.set_title("Static Margin")
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Static Margin (c)")
            ax1.grid()

            ax2 = plt.subplot(212)
            maxAttitude = max(flight.attitudeFrequencyResponse[:, 1])
            maxAttitude = maxAttitude if maxAttitude != 0 else 1
            ax2.plot(
                flight.attitudeFrequencyResponse[:, 0],
                flight.attitudeFrequencyResponse[:, 1] / maxAttitude,
                label="Attitude Angle",
            )
            maxOmega1 = max(flight.omega1FrequencyResponse[:, 1])
            maxOmega1 = maxOmega1 if maxOmega1 != 0 else 1
            ax2.plot(
                flight.omega1FrequencyResponse[:, 0],
                flight.omega1FrequencyResponse[:, 1] / maxOmega1,
                label=r"$\omega_1$",
            )
            maxOmega2 = max(flight.omega2FrequencyResponse[:, 1])
            maxOmega2 = maxOmega2 if maxOmega2 != 0 else 1
            ax2.plot(
                flight.omega2FrequencyResponse[:, 0],
                flight.omega2FrequencyResponse[:, 1] / maxOmega2,
                label=r"$\omega_2$",
            )
            maxOmega3 = max(flight.omega3FrequencyResponse[:, 1])
            maxOmega3 = maxOmega3 if maxOmega3 != 0 else 1
            ax2.plot(
                flight.omega3FrequencyResponse[:, 0],
                flight.omega3FrequencyResponse[:, 1] / maxOmega3,
                label=r"$\omega_3$",
            )
            ax2.set_title("Frequency Response")
            ax2.set_xlabel("Frequency (Hz)")
            ax2.set_ylabel("Amplitude Magnitude Normalized")
            ax2.set_xlim(0, 5)
            ax2.legend()
            ax2.grid()

            plt.subplots_adjust(hspace=0.5)
            plt.show()

        return None

    def plotPressureSignals(self):
        """Prints out all Parachute Trigger Pressure Signals.
        This function can be called also for plot pressure data for flights
        without Parachutes, in this case the Pressure Signals will be simply
        the pressure provided by the atmosphericModel, at Flight z positions.
        This means that no noise will be considered if at least one parachute
        has not been added.

        This function aims to help the engineer to visually check if there
        are anomalies with the Flight Simulation.

        Parameters
        ----------
        None

        Return
        ------
        None
        """
        for index, flight in enumerate(self.trajectory_list):
            # Post-process results
            if flight.postProcessed is False:
                flight.postProcess()

            if len(flight.rocket.parachutes) == 0:
                plt.figure()
                ax1 = plt.subplot(111)
                ax1.plot(flight.z[:, 0], flight.env.pressure(flight.z[:, 1]))
                ax1.set_title(
                    "Pressure at Rocket's Altitude, Flight: {}".format(
                        self.names_list[index]
                    )
                )
                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("Pressure (Pa)")
                ax1.set_xlim(0, flight.tFinal)
                ax1.grid()

                plt.show()

            else:
                for parachute in flight.rocket.parachutes:
                    print("Parachute: ", parachute.name)
                    parachute.noiseSignalFunction()
                    parachute.noisyPressureSignalFunction()
                    parachute.cleanPressureSignalFunction()

        return None

    # Start definition of 'compare' plots methods

    def comparePositions(self):
        """_summary_

        Returns
        -------
        None
        """
        fig = plt.figure(figsize=(7, 10))  # width, height
        fig.suptitle("Rocket Position Comparison", fontsize=16, y=1.02, x=0.5)

        ax1 = plt.subplot(312)
        max_time = 0
        for index, flight in enumerate(self.trajectory_list):
            if flight.postProcessed is False:
                flight.postProcess()
            ax1.plot(
                flight.x[:, 0],
                flight.x[:, 1],
                # color=self.colors_scale[index],
            )
            max_time = flight.tFinal if flight.tFinal > max_time else max_time
        ax1.set_xlim(0, max_time)
        ax1.set_title(
            "{} x {}".format(flight.x.getOutputs()[0], flight.x.getInputs()[0])
        )
        ax1.set_xlabel(flight.x.getInputs()[0])
        ax1.set_ylabel(flight.x.getOutputs()[0])
        ax1.grid(True)

        ax2 = plt.subplot(313)
        for index, flight in enumerate(self.trajectory_list):
            ax2.plot(
                flight.y[:, 0],
                flight.y[:, 1],
                # color=self.colors_scale[index],
            )
        ax2.set_xlim(0, max_time)
        ax2.set_title(
            "{} x {}".format(flight.y.getOutputs()[0], flight.y.getInputs()[0])
        )
        ax2.set_xlabel(flight.y.getInputs()[0])
        ax2.set_ylabel(flight.y.getOutputs()[0])
        ax2.grid(True)

        ax3 = plt.subplot(311)
        for index, flight in enumerate(self.trajectory_list):
            ax3.plot(
                flight.z[:, 0],
                flight.z[:, 1],
                label=self.names_list[index],
                # color=self.colors_scale[index],
            )
        ax3.set_xlim(0, max_time)
        ax3.set_title(
            "{} x {}".format(flight.z.getOutputs()[0], flight.z.getInputs()[0])
        )
        ax3.set_xlabel(flight.z.getInputs()[0])
        ax3.set_ylabel(flight.z.getInputs()[0])
        ax3.grid(True)

        fig.legend(
            loc="upper center",
            ncol=len(self.names_list),
            fancybox=True,
            shadow=True,
            fontsize=10,
            bbox_to_anchor=(0.5, 0.995),
        )
        fig.tight_layout()

        return None

    def compareVelocities(self):
        """_summary_

        Returns
        -------
        None
        """

        fig = plt.figure(figsize=(7, 10))  # width, height
        fig.suptitle("Rocket Velocity Comparison", fontsize=16, y=1.02, x=0.5)

        ax1 = plt.subplot(412)
        max_time = 0
        for index, flight in enumerate(self.trajectory_list):
            if flight.postProcessed is False:
                flight.postProcess()
            ax1.plot(
                flight.vx[:, 0],
                flight.vx[:, 1],
                # color=self.colors_scale[index],
            )
            max_time = flight.tFinal if flight.tFinal > max_time else max_time
        ax1.set_xlim(0, max_time)
        ax1.set_title(
            "{} x {}".format(flight.vx.getOutputs()[0], flight.vx.getInputs()[0])
        )
        ax1.set_xlabel(flight.vx.getInputs()[0])
        ax1.set_ylabel(flight.vx.getOutputs()[0])
        ax1.grid(True)

        ax2 = plt.subplot(413)
        for index, flight in enumerate(self.trajectory_list):
            ax2.plot(
                flight.vy[:, 0],
                flight.vy[:, 1],
                # color=self.colors_scale[index],
            )
        ax2.set_xlim(0, max_time)
        ax2.set_title(
            "{} x {}".format(flight.vy.getOutputs()[0], flight.vy.getInputs()[0])
        )
        ax2.set_xlabel(flight.vy.getInputs()[0])
        ax2.set_ylabel(flight.vy.getOutputs()[0])
        ax2.grid(True)

        ax3 = plt.subplot(414)
        for index, flight in enumerate(self.trajectory_list):
            ax3.plot(
                flight.vz[:, 0],
                flight.vz[:, 1],
                label=self.names_list[index],
                # color=self.colors_scale[index],
            )
        ax3.set_xlim(0, max_time)
        ax3.set_title(
            "{} x {}".format(flight.vz.getOutputs()[0], flight.vz.getInputs()[0])
        )
        ax3.set_xlabel(flight.vz.getInputs()[0])
        ax3.set_ylabel(flight.vz.getOutputs()[0])
        ax3.grid(True)

        ax4 = plt.subplot(411)
        for index, flight in enumerate(self.trajectory_list):
            ax4.plot(
                flight.speed[:, 0],
                flight.speed[:, 1],
                # color=self.colors_scale[index],
            )
        ax4.set_xlim(0, max_time)
        ax4.set_title(
            "{} x {}".format(flight.speed.getOutputs()[0], flight.speed.getInputs()[0])
        )
        ax4.set_xlabel(flight.speed.getInputs()[0])
        ax4.set_ylabel(flight.speed.getOutputs()[0])
        ax4.grid(True)

        fig.legend(
            loc="upper center",
            ncol=len(
                self.names_list
            ),  # TODO: Need to be more flexible here, changing the number of rows as well
            fancybox=True,
            shadow=True,
            fontsize=10,
            bbox_to_anchor=(0.5, 0.995),
        )
        fig.tight_layout()

        return None

    def compareStreamVelocities(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        fig = plt.figure(figsize=(7, 10))  # width, height
        fig.suptitle(
            "Rocket Freestream Velocity Comparison", fontsize=16, y=1.02, x=0.5
        )

        ax1 = plt.subplot(411)
        max_time = 0
        for index, flight in enumerate(self.trajectory_list):
            if flight.postProcessed is False:
                flight.postProcess()
            ax1.plot(
                flight.freestreamSpeed[:, 0],
                flight.freestreamSpeed[:, 1],
                label=self.names_list[index],
                # color=self.colors_scale[index],
            )
            max_time = flight.tFinal if flight.tFinal > max_time else max_time
        ax1.set_xlim(0, max_time)
        ax1.set_title(
            "{} x {}".format(
                flight.freestreamSpeed.getOutputs()[0],
                flight.freestreamSpeed.getInputs()[0],
            )
        )
        ax1.set_xlabel(flight.freestreamSpeed.getInputs()[0])
        ax1.set_ylabel(flight.freestreamSpeed.getOutputs()[0])
        ax1.grid(True)

        ax2 = plt.subplot(412)
        for index, flight in enumerate(self.trajectory_list):
            ax2.plot(
                flight.streamVelocityX[:, 0],
                flight.streamVelocityX[:, 1],
                # color=self.colors_scale[index],
            )
        ax2.set_xlim(0, max_time)
        ax2.set_title(
            "{} x {}".format(
                flight.streamVelocityX.getOutputs()[0],
                flight.streamVelocityX.getInputs()[0],
            )
        )
        ax2.set_xlabel(flight.streamVelocityX.getInputs()[0])
        ax2.set_ylabel(flight.streamVelocityX.getOutputs()[0])
        ax2.grid(True)

        ax3 = plt.subplot(413)
        for index, flight in enumerate(self.trajectory_list):
            ax3.plot(
                flight.streamVelocityY[:, 0],
                flight.streamVelocityY[:, 1],
                # color=self.colors_scale[index],
            )
        ax3.set_xlim(0, max_time)
        ax3.set_title(
            "{} x {}".format(
                flight.streamVelocityY.getOutputs()[0],
                flight.streamVelocityY.getInputs()[0],
            )
        )
        ax3.set_xlabel(flight.streamVelocityY.getInputs()[0])
        ax3.set_ylabel(flight.streamVelocityY.getOutputs()[0])
        ax3.grid(True)

        ax4 = plt.subplot(414)
        for index, flight in enumerate(self.trajectory_list):
            ax4.plot(
                flight.streamVelocityZ[:, 0],
                flight.streamVelocityZ[:, 1],
                # color=self.colors_scale[index],
            )
        ax4.set_xlim(0, max_time)
        ax4.set_title(
            "{} x {}".format(
                flight.streamVelocityZ.getOutputs()[0],
                flight.streamVelocityZ.getInputs()[0],
            )
        )
        ax4.set_xlabel(flight.streamVelocityZ.getInputs()[0])
        ax4.set_ylabel(flight.streamVelocityZ.getOutputs()[0])
        ax4.grid(True)

        fig.legend(
            loc="upper center",
            ncol=len(self.names_list),
            fancybox=True,
            shadow=True,
            fontsize=10,
            bbox_to_anchor=(0.5, 0.995),
        )
        # TODO: Create a option to insert watermark or not, including RocketPy logo
        # fig.text(
        #     x=0.8,
        #     y=0,
        #     s="created with RocketPy",
        #     fontsize=10,
        #     color="black",
        #     alpha=1,
        #     ha="center",
        #     va="center",
        #     rotation=0,
        # )
        fig.tight_layout()
        return None

    def compareAccelerations(self):
        """_summary_

        Returns
        -------
        None
        """

        fig = plt.figure(figsize=(7, 10))  # width, height
        fig.suptitle("Rocket Acceleration Comparison", fontsize=16, y=1.02, x=0.5)

        ax1 = plt.subplot(412)
        max_time = 0
        for index, flight in enumerate(self.trajectory_list):
            if flight.postProcessed is False:
                flight.postProcess()
            ax1.plot(
                flight.ax[:, 0],
                flight.ax[:, 1],
                # color=self.colors_scale[index],
            )
            max_time = flight.tFinal if flight.tFinal > max_time else max_time
        ax1.set_xlim(0, max_time)
        ax1.set_title(
            "{} x {}".format(flight.ax.getOutputs()[0], flight.ax.getInputs()[0])
        )
        ax1.set_xlabel(flight.ax.getInputs()[0])
        ax1.set_ylabel(flight.ax.getOutputs()[0])
        ax1.grid(True)

        ax2 = plt.subplot(413)
        for index, flight in enumerate(self.trajectory_list):
            ax2.plot(
                flight.vy[:, 0],
                flight.vy[:, 1],
                # color=self.colors_scale[index],
            )
        ax2.set_xlim(0, max_time)
        ax2.set_title(
            "{} x {}".format(flight.ay.getOutputs()[0], flight.ay.getInputs()[0])
        )
        ax2.set_xlabel(flight.ay.getInputs()[0])
        ax2.set_ylabel(flight.ay.getOutputs()[0])
        ax2.grid(True)

        ax3 = plt.subplot(414)
        for index, flight in enumerate(self.trajectory_list):
            ax3.plot(
                flight.vz[:, 0],
                flight.vz[:, 1],
                label=self.names_list[index],
                # color=self.colors_scale[index],
            )
        ax3.set_xlim(0, max_time)
        ax3.set_title(
            "{} x {}".format(flight.az.getOutputs()[0], flight.az.getInputs()[0])
        )
        ax3.set_xlabel(flight.vz.getInputs()[0])
        ax3.set_ylabel(flight.vz.getOutputs()[0])
        ax3.grid(True)

        ax4 = plt.subplot(411)
        for index, flight in enumerate(self.trajectory_list):
            ax4.plot(
                flight.acceleration[:, 0],
                flight.acceleration[:, 1],
                # color=self.colors_scale[index],
            )
        ax4.set_xlim(0, max_time)
        ax4.set_title(
            "{} x {}".format(
                flight.acceleration.getOutputs()[0], flight.acceleration.getInputs()[0]
            )
        )
        ax4.set_xlabel(flight.acceleration.getInputs()[0])
        ax4.set_ylabel(flight.acceleration.getOutputs()[0])
        ax4.grid(True)

        fig.legend(
            loc="upper center",
            ncol=len(self.names_list),
            fancybox=True,
            shadow=True,
            fontsize=10,
            bbox_to_anchor=(0.5, 0.995),
        )
        fig.tight_layout()

        return None

    def compareEulerAngles(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        fig = plt.figure(figsize=(7, 10))  # width, height
        fig.suptitle("Euler Angles Comparison", fontsize=16, y=1.02, x=0.5)

        ax1 = plt.subplot(311)
        max_time = 0
        for index, flight in enumerate(self.trajectory_list):
            if flight.postProcessed is False:
                flight.postProcess()
            ax1.plot(
                flight.phi[:, 0],
                flight.phi[:, 1],
                # color=self.colors_scale[index],
            )
            max_time = flight.tFinal if flight.tFinal > max_time else max_time
        ax1.set_xlim(0, max_time)
        ax1.set_title(
            "{} x {}".format(flight.phi.getOutputs()[0], flight.phi.getInputs()[0])
        )
        ax1.set_xlabel(flight.phi.getInputs()[0])
        ax1.set_ylabel(flight.phi.getOutputs()[0])
        ax1.grid(True)

        ax2 = plt.subplot(312)
        for index, flight in enumerate(self.trajectory_list):
            ax2.plot(
                flight.theta[:, 0],
                flight.theta[:, 1],
                # color=self.colors_scale[index],
            )
        ax2.set_xlim(0, max_time)
        ax2.set_title(
            "{} x {}".format(flight.theta.getOutputs()[0], flight.theta.getInputs()[0])
        )
        ax2.set_xlabel(flight.theta.getInputs()[0])
        ax2.set_ylabel(flight.theta.getOutputs()[0])
        ax2.grid(True)

        ax3 = plt.subplot(313)
        for index, flight in enumerate(self.trajectory_list):
            ax3.plot(
                flight.psi[:, 0],
                flight.psi[:, 1],
                label=self.names_list[index],
                # color=self.colors_scale[index],
            )
        ax3.set_xlim(0, max_time)
        ax3.set_title(
            "{} x {}".format(flight.psi.getOutputs()[0], flight.psi.getInputs()[0])
        )
        ax3.set_xlabel(flight.psi.getInputs()[0])
        ax3.set_ylabel(flight.psi.getOutputs()[0])
        ax3.grid(True)

        fig.legend(
            loc="upper center",
            ncol=len(self.names_list),
            fancybox=True,
            shadow=True,
            fontsize=10,
            bbox_to_anchor=(0.5, 0.995),
        )
        fig.tight_layout()

        return None

    def compareQuaternions(self):

        fig = plt.figure(figsize=(10, 20 / 3))  # width, height
        fig.suptitle("Quaternions Comparison", fontsize=16, y=1.06, x=0.5)

        ax1 = plt.subplot(221)
        max_time = 0
        for index, flight in enumerate(self.trajectory_list):
            if flight.postProcessed is False:
                flight.postProcess()
            ax1.plot(
                flight.e0[:, 0],
                flight.e0[:, 1],
                label=self.names_list[index],
                # color=self.colors_scale[index],
            )
            max_time = flight.tFinal if flight.tFinal > max_time else max_time
        ax1.set_xlim(0, max_time)
        ax1.set_title(
            "{} x {}".format(flight.e0.getOutputs()[0], flight.e0.getInputs()[0])
        )
        ax1.set_xlabel(flight.e0.getInputs()[0])
        ax1.set_ylabel(flight.e0.getOutputs()[0])
        ax1.grid(True)

        ax2 = plt.subplot(222)
        for index, flight in enumerate(self.trajectory_list):
            ax2.plot(
                flight.e1[:, 0],
                flight.e1[:, 1],
                # color=self.colors_scale[index],
            )
        ax2.set_xlim(0, max_time)
        ax2.set_title(
            "{} x {}".format(flight.e1.getOutputs()[0], flight.e1.getInputs()[0])
        )
        ax2.set_xlabel(flight.e1.getInputs()[0])
        ax2.set_ylabel(flight.e1.getOutputs()[0])
        ax2.grid(True)

        ax3 = plt.subplot(223)
        for index, flight in enumerate(self.trajectory_list):
            ax3.plot(
                flight.e2[:, 0],
                flight.e2[:, 1],
                # color=self.colors_scale[index],
            )
        ax3.set_xlim(0, max_time)
        ax3.set_title(
            "{} x {}".format(flight.e2.getOutputs()[0], flight.e2.getInputs()[0])
        )
        ax3.set_xlabel(flight.e2.getInputs()[0])
        ax3.set_ylabel(flight.e2.getOutputs()[0])
        ax3.grid(True)

        ax4 = plt.subplot(224)
        for index, flight in enumerate(self.trajectory_list):
            ax4.plot(
                flight.e3[:, 0],
                flight.e3[:, 1],
                # color=self.colors_scale[index],
            )
        ax4.set_xlim(0, max_time)
        ax4.set_title(
            "{} x {}".format(flight.e3.getOutputs()[0], flight.e3.getInputs()[0])
        )
        ax4.set_xlabel(flight.e3.getInputs()[0])
        ax4.set_ylabel(flight.e3.getOutputs()[0])
        ax4.grid(True)

        fig.legend(
            loc="upper center",
            ncol=len(self.names_list),
            fancybox=True,
            shadow=True,
            fontsize=10,
            bbox_to_anchor=(0.5, 1),
        )
        fig.tight_layout()

        return None

    def compareAttitudeAngles(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """

        fig = plt.figure(figsize=(7, 10))  # width, height
        fig.suptitle("Attitude Angles Comparison", fontsize=16, y=1.06, x=0.5)

        ax1 = plt.subplot(311)
        max_time = 0
        for index, flight in enumerate(self.trajectory_list):
            if flight.postProcessed is False:
                flight.postProcess()
            ax1.plot(
                flight.pathAngle[:, 0],
                flight.pathAngle[:, 1],
                label=self.names_list[index],
                # color=self.colors_scale[index],
            )
            max_time = flight.tFinal if flight.tFinal > max_time else max_time

        ax1.set_xlim(0, max_time)
        ax1.set_title(
            "{} x {}".format(
                flight.pathAngle.getOutputs()[0], flight.pathAngle.getInputs()[0]
            )
        )
        ax1.set_xlabel(flight.pathAngle.getInputs()[0])
        ax1.set_ylabel(flight.pathAngle.getOutputs()[0])
        ax1.grid(True)

        ax2 = plt.subplot(312)
        for index, flight in enumerate(self.trajectory_list):
            ax2.plot(
                flight.attitudeAngle[:, 0],
                flight.attitudeAngle[:, 1],
                # color=self.colors_scale[index],
            )
        ax2.set_xlim(0, max_time)
        ax2.set_title(
            "{} x {}".format(
                flight.attitudeAngle.getOutputs()[0],
                flight.attitudeAngle.getInputs()[0],
            )
        )
        ax2.set_xlabel(flight.attitudeAngle.getInputs()[0])
        ax2.set_ylabel(flight.attitudeAngle.getOutputs()[0])
        ax2.grid(True)

        ax3 = plt.subplot(313)
        for index, flight in enumerate(self.trajectory_list):
            ax3.plot(
                flight.lateralAttitudeAngle[:, 0],
                flight.lateralAttitudeAngle[:, 1],
                # color=self.colors_scale[index],
            )
        ax3.set_xlim(0, max_time)
        ax3.set_title(
            "{} x {}".format(
                flight.lateralAttitudeAngle.getOutputs()[0],
                flight.lateralAttitudeAngle.getInputs()[0],
            )
        )
        ax3.set_xlabel(flight.lateralAttitudeAngle.getInputs()[0])
        ax3.set_ylabel(flight.lateralAttitudeAngle.getOutputs()[0])
        ax3.grid(True)

        fig.legend(
            loc="upper center",
            ncol=len(self.names_list),
            fancybox=True,
            shadow=True,
            fontsize=10,
            bbox_to_anchor=(0.5, 1),
        )
        fig.tight_layout()

        return None

    def compareAngularVelocities(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        fig = plt.figure(figsize=(7, 10))  # width, height
        fig.suptitle("Angular Velocities Comparison", fontsize=16, y=1.06, x=0.5)

        ax1 = plt.subplot(311)
        max_time = 0
        for index, flight in enumerate(self.trajectory_list):
            if flight.postProcessed is False:
                flight.postProcess()
            ax1.plot(
                flight.w1[:, 0],
                flight.w1[:, 1],
                label=self.names_list[index],
                # color=self.colors_scale[index],
            )
            max_time = flight.tFinal if flight.tFinal > max_time else max_time
        ax1.set_xlim(0, max_time)
        ax1.set_title(
            "{} x {}".format(flight.w1.getOutputs()[0], flight.w1.getInputs()[0])
        )
        ax1.set_xlabel(flight.w1.getInputs()[0])
        ax1.set_ylabel(flight.w1.getOutputs()[0])
        ax1.grid(True)

        ax2 = plt.subplot(312)
        for index, flight in enumerate(self.trajectory_list):
            ax2.plot(
                flight.w2[:, 0],
                flight.w2[:, 1],
                # color=self.colors_scale[index],
            )
        ax2.set_xlim(0, max_time)
        ax2.set_title(
            "{} x {}".format(flight.w2.getOutputs()[0], flight.w2.getInputs()[0])
        )
        ax2.set_xlabel(flight.w2.getInputs()[0])
        ax2.set_ylabel(flight.w2.getOutputs()[0])
        ax2.grid(True)

        ax3 = plt.subplot(313)
        for index, flight in enumerate(self.trajectory_list):
            ax3.plot(
                flight.w3[:, 0],
                flight.w3[:, 1],
                # color=self.colors_scale[index],
            )
        ax3.set_xlim(0, max_time)
        ax3.set_title(
            "{} x {}".format(flight.w3.getOutputs()[0], flight.w3.getInputs()[0])
        )
        ax3.set_xlabel(flight.w3.getInputs()[0])
        ax3.set_ylabel(flight.w3.getOutputs()[0])
        ax3.grid(True)

        fig.legend(
            loc="upper center",
            ncol=len(self.names_list),
            fancybox=True,
            shadow=True,
            fontsize=10,
            bbox_to_anchor=(0.5, 1),
        )
        fig.tight_layout()

        return None

    def compareAngularAccelerations(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        fig = plt.figure(figsize=(7, 10))  # width, height
        fig.suptitle("Angular Accelerations Comparison", fontsize=16, y=1.06, x=0.5)

        ax1 = plt.subplot(311)
        max_time = 0
        for index, flight in enumerate(self.trajectory_list):
            if flight.postProcessed is False:
                flight.postProcess()
            ax1.plot(
                flight.alpha1[:, 0],
                flight.alpha1[:, 1],
                label=self.names_list[index],
                # color=self.colors_scale[index],
            )
            max_time = flight.tFinal if flight.tFinal > max_time else max_time
        ax1.set_xlim(0, max_time)
        ax1.set_title(
            "{} x {}".format(
                flight.alpha1.getOutputs()[0], flight.alpha1.getInputs()[0]
            )
        )
        ax1.set_xlabel(flight.alpha1.getInputs()[0])
        ax1.set_ylabel(flight.alpha1.getOutputs()[0])
        ax1.grid(True)

        ax2 = plt.subplot(312)
        for index, flight in enumerate(self.trajectory_list):
            ax2.plot(
                flight.alpha2[:, 0],
                flight.alpha2[:, 1],
                # color=self.colors_scale[index],
            )
        ax2.set_xlim(0, max_time)
        ax2.set_title(
            "{} x {}".format(
                flight.alpha2.getOutputs()[0], flight.alpha2.getInputs()[0]
            )
        )
        ax2.set_xlabel(flight.alpha2.getInputs()[0])
        ax2.set_ylabel(flight.alpha2.getOutputs()[0])
        ax2.grid(True)

        ax3 = plt.subplot(313)
        for index, flight in enumerate(self.trajectory_list):
            ax3.plot(
                flight.alpha3[:, 0],
                flight.alpha3[:, 1],
                # color=self.colors_scale[index],
            )
        ax3.set_xlim(0, max_time)
        ax3.set_title(
            "{} x {}".format(
                flight.alpha3.getOutputs()[0], flight.alpha3.getInputs()[0]
            )
        )
        ax3.set_xlabel(flight.alpha3.getInputs()[0])
        ax3.set_ylabel(flight.alpha3.getOutputs()[0])
        ax3.grid(True)

        fig.legend(
            loc="upper center",
            ncol=len(self.names_list),
            fancybox=True,
            shadow=True,
            fontsize=10,
            bbox_to_anchor=(0.5, 1),
        )
        fig.tight_layout()

        return None

    def compareAerodynamicForces(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        fig = plt.figure(figsize=(7, 20 / 3))  # width, height
        fig.suptitle("Aerodynamic Forces Comparison", fontsize=16, y=1.06, x=0.5)

        ax1 = plt.subplot(211)
        max_time = 0
        for index, flight in enumerate(self.trajectory_list):
            if flight.postProcessed is False:
                flight.postProcess()
            ax1.plot(
                flight.aerodynamicDrag[:, 0],
                flight.aerodynamicDrag[:, 1],
                label=self.names_list[index],
                # color=self.colors_scale[index],
            )
            max_time = flight.tFinal if flight.tFinal > max_time else max_time
        ax1.set_xlim(0, max_time)
        ax1.set_title(
            "{} x {}".format(
                flight.aerodynamicDrag.getOutputs()[0],
                flight.aerodynamicDrag.getInputs()[0],
            )
        )
        ax1.set_xlabel(flight.aerodynamicDrag.getInputs()[0])
        ax1.set_ylabel(flight.aerodynamicDrag.getOutputs()[0])
        ax1.grid(True)

        ax2 = plt.subplot(212)
        for index, flight in enumerate(self.trajectory_list):
            ax2.plot(
                flight.aerodynamicLift[:, 0],
                flight.aerodynamicLift[:, 1],
                # color=self.colors_scale[index],
            )
        ax2.set_xlim(0, max_time)
        ax2.set_title(
            "{} x {}".format(
                flight.aerodynamicLift.getOutputs()[0],
                flight.aerodynamicLift.getInputs()[0],
            )
        )
        ax2.set_xlabel(flight.aerodynamicLift.getInputs()[0])
        ax2.set_ylabel(flight.aerodynamicLift.getOutputs()[0])
        ax2.grid(True)

        fig.legend(
            loc="upper center",
            ncol=len(self.names_list),
            fancybox=True,
            shadow=True,
            fontsize=10,
            bbox_to_anchor=(0.5, 1),
        )
        fig.tight_layout()

        return None

    def compareAerodynamicMoments(self):
        """_summary_

        Returns
        -------
        None
        """
        fig = plt.figure(figsize=(7, 20 / 3))  # width, height
        fig.suptitle("Aerodynamic Moments Comparison", fontsize=16, y=1.06, x=0.5)

        ax1 = plt.subplot(211)
        max_time = 0
        for index, flight in enumerate(self.trajectory_list):
            if flight.postProcessed is False:
                flight.postProcess()
            ax1.plot(
                flight.aerodynamicBendingMoment[:, 0],
                flight.aerodynamicBendingMoment[:, 1],
                label=self.names_list[index],
                # color=self.colors_scale[index],
            )
            max_time = flight.tFinal if flight.tFinal > max_time else max_time
        ax1.set_xlim(0, max_time)
        ax1.set_title(
            "{} x {}".format(
                flight.aerodynamicBendingMoment.getOutputs()[0],
                flight.aerodynamicBendingMoment.getInputs()[0],
            )
        )
        ax1.set_xlabel(flight.aerodynamicBendingMoment.getInputs()[0])
        ax1.set_ylabel(flight.aerodynamicBendingMoment.getOutputs()[0])
        ax1.grid(True)

        ax2 = plt.subplot(212)
        for index, flight in enumerate(self.trajectory_list):
            ax2.plot(
                flight.aerodynamicSpinMoment[:, 0],
                flight.aerodynamicSpinMoment[:, 1],
                # color=self.colors_scale[index],
            )
        ax2.set_xlim(0, max_time)
        ax2.set_title(
            "{} x {}".format(
                flight.aerodynamicSpinMoment.getOutputs()[0],
                flight.aerodynamicSpinMoment.getInputs()[0],
            )
        )
        ax2.set_xlabel(flight.aerodynamicSpinMoment.getInputs()[0])
        ax2.set_ylabel(flight.aerodynamicSpinMoment.getOutputs()[0])
        ax2.grid(True)

        fig.legend(
            loc="upper center",
            ncol=len(self.names_list),
            fancybox=True,
            shadow=True,
            fontsize=10,
            bbox_to_anchor=(0.5, 1),
        )
        fig.tight_layout()

        return None

    def compareEnergies(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        fig = plt.figure(figsize=(7, 50 / 3))  # width, height
        fig.suptitle("Energies Comparison", fontsize=16, y=1.06, x=0.5)

        ax1 = plt.subplot(511)
        max_time = 0
        for index, flight in enumerate(self.trajectory_list):
            if flight.postProcessed is False:
                flight.postProcess()
            ax1.plot(
                flight.kineticEnergy[:, 0],
                flight.kineticEnergy[:, 1],
                label=self.names_list[index],
                # color=self.colors_scale[index],
            )
            max_time = flight.tFinal if flight.tFinal > max_time else max_time
        ax1.set_xlim(0, max_time)
        ax1.set_title(
            "{} x {}".format(
                flight.kineticEnergy.getOutputs()[0],
                flight.kineticEnergy.getInputs()[0],
            )
        )
        ax1.set_xlabel(flight.kineticEnergy.getInputs()[0])
        ax1.set_ylabel(flight.kineticEnergy.getOutputs()[0])
        ax1.grid(True)

        ax2 = plt.subplot(512)
        for index, flight in enumerate(self.trajectory_list):
            ax2.plot(
                flight.rotationalEnergy[:, 0],
                flight.rotationalEnergy[:, 1],
                # color=self.colors_scale[index],
            )
        ax2.set_xlim(0, max_time)
        ax2.set_title(
            "{} x {}".format(
                flight.rotationalEnergy.getOutputs()[0],
                flight.rotationalEnergy.getInputs()[0],
            )
        )
        ax2.set_xlabel(flight.rotationalEnergy.getInputs()[0])
        ax2.set_ylabel(flight.rotationalEnergy.getOutputs()[0])
        ax2.grid(True)

        ax3 = plt.subplot(513)
        for index, flight in enumerate(self.trajectory_list):
            ax3.plot(
                flight.translationalEnergy[:, 0],
                flight.translationalEnergy[:, 1],
                # color=self.colors_scale[index],
            )
        ax3.set_xlim(0, max_time)
        ax3.set_title(
            "{} x {}".format(
                flight.translationalEnergy.getOutputs()[0],
                flight.translationalEnergy.getInputs()[0],
            )
        )
        ax3.set_xlabel(flight.translationalEnergy.getInputs()[0])
        ax3.set_ylabel(flight.translationalEnergy.getOutputs()[0])
        ax3.grid(True)

        ax4 = plt.subplot(514)
        for index, flight in enumerate(self.trajectory_list):
            ax4.plot(
                flight.potentialEnergy[:, 0],
                flight.potentialEnergy[:, 1],
                # color=self.colors_scale[index],
            )
        ax4.set_xlim(0, max_time)
        ax4.set_title(
            "{} x {}".format(
                flight.potentialEnergy.getOutputs()[0],
                flight.potentialEnergy.getInputs()[0],
            )
        )
        ax4.set_xlabel(flight.potentialEnergy.getInputs()[0])
        ax4.set_ylabel(flight.potentialEnergy.getOutputs()[0])
        ax4.grid(True)

        ax5 = plt.subplot(515)
        for index, flight in enumerate(self.trajectory_list):
            ax5.plot(
                flight.totalEnergy[:, 0],
                flight.totalEnergy[:, 1],
                # color=self.colors_scale[index],
            )
        ax5.set_xlim(0, max_time)
        ax5.set_title(
            "{} x {}".format(
                flight.totalEnergy.getOutputs()[0],
                flight.totalEnergy.getInputs()[0],
            )
        )
        ax5.set_xlabel(flight.totalEnergy.getInputs()[0])
        ax5.set_ylabel(flight.totalEnergy.getOutputs()[0])
        ax5.grid(True)

        fig.legend(
            loc="upper center",
            ncol=len(self.names_list),
            fancybox=True,
            shadow=True,
            fontsize=10,
            bbox_to_anchor=(0.5, 1),
        )
        fig.tight_layout()

        return None

    def comparePowers(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        fig = plt.figure(figsize=(7, 20 / 3))  # width, height
        fig.suptitle("Powers Comparison", fontsize=16, y=1.06, x=0.5)

        ax1 = plt.subplot(211)
        max_time = 0
        for index, flight in enumerate(self.trajectory_list):
            if flight.postProcessed is False:
                flight.postProcess()
            ax1.plot(
                flight.thrustPower[:, 0],
                flight.thrustPower[:, 1],
                label=self.names_list[index],
                # color=self.colors_scale[index],
            )
            max_time = flight.tFinal if flight.tFinal > max_time else max_time
        ax1.set_xlim(0, max_time)
        ax1.set_title(
            "{} x {}".format(
                flight.thrustPower.getOutputs()[0],
                flight.thrustPower.getInputs()[0],
            )
        )

        ax1.set_xlabel(flight.thrustPower.getInputs()[0])
        ax1.set_ylabel(flight.thrustPower.getOutputs()[0])
        ax1.grid(True)

        ax2 = plt.subplot(212)
        for index, flight in enumerate(self.trajectory_list):
            ax2.plot(
                flight.dragPower[:, 0],
                flight.dragPower[:, 1],
                # color=self.colors_scale[index],
            )
        ax2.set_xlim(0, max_time)
        ax2.set_title(
            "{} x {}".format(
                flight.dragPower.getOutputs()[0],
                flight.dragPower.getInputs()[0],
            )
        )
        ax2.set_xlabel(flight.dragPower.getInputs()[0])
        ax2.set_ylabel(flight.dragPower.getOutputs()[0])
        ax2.grid(True)

        fig.legend(
            loc="upper center",
            ncol=len(self.names_list),
            fancybox=True,
            shadow=True,
            fontsize=10,
            bbox_to_anchor=(0.5, 1),
        )
        fig.tight_layout()

        return None

    def compareRailButtonsForces(self):
        """_summary_

        Returns
        -------
        None
        """
        fig = plt.figure(figsize=(10, 20 / 3))  # width, height
        fig.suptitle("Rail Buttons Forces Comparison", fontsize=16, y=1.06, x=0.5)

        ax1 = plt.subplot(221)
        max_time = 0
        for index, flight in enumerate(self.trajectory_list):
            if flight.rocket.railButtons is None:
                continue
            if flight.postProcessed is False:
                flight.postProcess()
            ax1.plot(
                flight.railButton1NormalForce[:, 0],
                flight.railButton1NormalForce[:, 1],
                label=self.names_list[index],
                # color=self.colors_scale[index],
            )
            max_time = flight.tFinal if flight.tFinal > max_time else max_time

            ax1.set_title(
                "{} x {}".format(
                    flight.railButton1NormalForce.getOutputs()[0],
                    flight.railButton1NormalForce.getInputs()[0],
                )
            )
            ax1.set_xlabel(flight.railButton1NormalForce.getInputs()[0])
            ax1.set_ylabel(flight.railButton1NormalForce.getOutputs()[0])
        ax1.set_xlim(0, max_time)
        ax1.grid(True)

        ax2 = plt.subplot(223)
        for index, flight in enumerate(self.trajectory_list):
            if flight.rocket.railButtons is None:
                continue
            ax2.plot(
                flight.railButton2NormalForce[:, 0],
                flight.railButton2NormalForce[:, 1],
                # color=self.colors_scale[index],
            )
            ax2.set_title(
                "{} x {}".format(
                    flight.railButton2NormalForce.getOutputs()[0],
                    flight.railButton2NormalForce.getInputs()[0],
                )
            )
            ax2.set_xlabel(flight.railButton2NormalForce.getInputs()[0])
            ax2.set_ylabel(flight.railButton2NormalForce.getOutputs()[0])
        ax2.set_xlim(0, max_time)
        ax2.grid(True)

        ax3 = plt.subplot(222)
        for index, flight in enumerate(self.trajectory_list):
            if flight.rocket.railButtons is None:
                continue
            ax3.plot(
                flight.railButton1ShearForce[:, 0],
                flight.railButton1ShearForce[:, 1],
                # color=self.colors_scale[index],
            )

            ax3.set_title(
                "{} x {}".format(
                    flight.railButton1ShearForce.getOutputs()[0],
                    flight.railButton1ShearForce.getInputs()[0],
                )
            )
            ax3.set_xlabel(flight.railButton1ShearForce.getInputs()[0])
            ax3.set_ylabel(flight.railButton1ShearForce.getOutputs()[0])
        ax3.set_xlim(0, max_time)
        ax3.grid(True)

        ax4 = plt.subplot(224)
        for index, flight in enumerate(self.trajectory_list):
            if flight.rocket.railButtons is None:
                continue
            ax4.plot(
                flight.railButton2ShearForce[:, 0],
                flight.railButton2ShearForce[:, 1],
                # color=self.colors_scale[index],
            )

            ax4.set_title(
                "{} x {}".format(
                    flight.railButton2ShearForce.getOutputs()[0],
                    flight.railButton2ShearForce.getInputs()[0],
                )
            )
            ax4.set_xlabel(flight.railButton2ShearForce.getInputs()[0])
            ax4.set_ylabel(flight.railButton2ShearForce.getOutputs()[0])
        ax4.set_xlim(0, max_time)
        ax4.grid(True)

        fig.legend(
            loc="upper center",
            ncol=len(self.names_list),
            fancybox=True,
            shadow=True,
            fontsize=10,
            bbox_to_anchor=(0.5, 1),
        )
        fig.tight_layout()

        return None

    def compareAnglesOfAttack(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        fig = plt.figure(figsize=(7, 10 / 3))  # width, height
        fig.suptitle("Angles of Attack Comparison", fontsize=16, y=1.06, x=0.5)

        ax1 = plt.subplot(111)
        max_time = 0
        for index, flight in enumerate(self.trajectory_list):
            if flight.postProcessed is False:
                flight.postProcess()
            ax1.plot(
                flight.angleOfAttack[:, 0],
                flight.angleOfAttack[:, 1],
                label=self.names_list[index],
                # color=self.colors_scale[index],
            )
            max_time = flight.tFinal if flight.tFinal > max_time else max_time
        ax1.set_xlim(0, max_time)
        ax1.set_title(
            "{} x {}".format(
                flight.angleOfAttack.getOutputs()[0],
                flight.angleOfAttack.getInputs()[0],
            )
        )
        ax1.set_xlabel(flight.angleOfAttack.getInputs()[0])
        ax1.set_ylabel(flight.angleOfAttack.getOutputs()[0])
        ax1.grid(True)

        # TODO: Maybe simplify this code with the use of a function to add legend
        fig.legend(
            loc="upper center",
            ncol=len(self.names_list),
            fancybox=True,
            shadow=True,
            fontsize=10,
            bbox_to_anchor=(0.5, 1),
        )
        fig.tight_layout()

        return None

    # TODO: Static Margin is not working properly, we need to understand why!
    def compareStaticMargins(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        fig = plt.figure(figsize=(7, 10 / 3))  # width, height
        fig.suptitle("Static Margins Comparison", fontsize=16, y=1.06, x=0.5)

        ax1 = plt.subplot(111)
        max_time = 0
        for index, flight in enumerate(self.trajectory_list):
            if flight.postProcessed is False:
                flight.postProcess()
            ax1.plot(
                flight.staticMargin[:, 0],
                flight.staticMargin[:, 1],
                label=self.names_list[index],
                # color=self.colors_scale[index],
            )
            max_time = flight.tFinal if flight.tFinal > max_time else max_time
        ax1.set_xlim(0, max_time)
        ax1.set_title(
            "{} x {}".format(
                flight.staticMargin.getOutputs()[0],
                flight.staticMargin.getInputs()[0],
            )
        )
        ax1.set_xlabel(flight.staticMargin.getInputs()[0])
        ax1.set_ylabel(flight.staticMargin.getOutputs()[0])
        ax1.grid(True)

        fig.legend(
            loc="upper center",
            ncol=len(self.names_list),
            fancybox=True,
            shadow=True,
            fontsize=10,
            bbox_to_anchor=(0.5, 1),
        )
        fig.tight_layout()

        return None

    def compareFluidMechanics(self):
        """_summary_

        Returns
        -------
        None
        """
        fig = plt.figure(figsize=(10, 20 / 3))  # width, height
        fig.suptitle("Fluid Mechanics Comparison", fontsize=16, y=1.06, x=0.5)

        ax1 = plt.subplot(221)
        max_time = 0
        for index, flight in enumerate(self.trajectory_list):
            if flight.postProcessed is False:
                flight.postProcess()
            ax1.plot(
                flight.MachNumber[:, 0],
                flight.MachNumber[:, 1],
                label=self.names_list[index],
                # color=self.colors_scale[index],
            )
            max_time = flight.tFinal if flight.tFinal > max_time else max_time
        ax1.set_xlim(0, max_time)
        ax1.set_title(
            "{} x {}".format(
                flight.MachNumber.getOutputs()[0], flight.MachNumber.getInputs()[0]
            )
        )
        ax1.set_xlabel(flight.MachNumber.getInputs()[0])
        ax1.set_ylabel(flight.MachNumber.getOutputs()[0])
        ax1.grid(True)

        ax2 = plt.subplot(222)
        for index, flight in enumerate(self.trajectory_list):
            ax2.plot(
                flight.ReynoldsNumber[:, 0],
                flight.ReynoldsNumber[:, 1],
                # color=self.colors_scale[index],
            )
        ax2.set_xlim(0, max_time)
        ax2.set_title(
            "{} x {}".format(
                flight.ReynoldsNumber.getOutputs()[0],
                flight.ReynoldsNumber.getInputs()[0],
            )
        )
        ax2.set_xlabel(flight.ReynoldsNumber.getInputs()[0])
        ax2.set_ylabel(flight.ReynoldsNumber.getOutputs()[0])
        ax2.grid(True)

        ax3 = plt.subplot(223)
        for index, flight in enumerate(self.trajectory_list):
            ax3.plot(
                flight.dynamicPressure[:, 0],
                flight.dynamicPressure[:, 1],
                # color=self.colors_scale[index],
            )
        ax3.set_xlim(0, max_time)
        ax3.set_title(
            "{} x {}".format(
                flight.dynamicPressure.getOutputs()[0],
                flight.dynamicPressure.getInputs()[0],
            )
        )
        ax3.set_xlabel(flight.dynamicPressure.getInputs()[0])
        ax3.set_ylabel(flight.dynamicPressure.getOutputs()[0])
        ax3.grid(True)

        ax4 = plt.subplot(224)
        for index, flight in enumerate(self.trajectory_list):
            ax4.plot(
                flight.totalPressure[:, 0],
                flight.totalPressure[:, 1],
                # color=self.colors_scale[index],
            )
        ax4.set_xlim(0, max_time)
        ax4.set_title(
            "{} x {}".format(
                flight.totalPressure.getOutputs()[0],
                flight.totalPressure.getInputs()[0],
            )
        )
        ax4.set_xlabel(flight.totalPressure.getInputs()[0])
        ax4.set_ylabel(flight.totalPressure.getOutputs()[0])
        ax4.grid(True)

        fig.legend(
            loc="upper center",
            ncol=len(self.names_list),
            fancybox=True,
            shadow=True,
            fontsize=10,
            bbox_to_anchor=(0.5, 1),
        )
        fig.tight_layout()

        return None

    def compareAttitudeFrequencyResponses(self):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        fig = plt.figure(figsize=(10, 20 / 3))  # width, height
        fig.suptitle(
            "Attitude Frequency Responses Comparison", fontsize=16, y=1.06, x=0.5
        )

        ax1 = plt.subplot(221)
        for index, flight in enumerate(self.trajectory_list):
            if flight.postProcessed is False:
                flight.postProcess()
            ax1.plot(
                flight.attitudeFrequencyResponse[:, 0],
                flight.attitudeFrequencyResponse[:, 1],
                label=self.names_list[index],
                # color=self.colors_scale[index],
            )
        ax1.set_title(
            "{} x {}".format(
                flight.attitudeFrequencyResponse.getOutputs()[0],
                flight.attitudeFrequencyResponse.getInputs()[0],
            )
        )
        ax1.set_xlabel(flight.attitudeFrequencyResponse.getInputs()[0])
        ax1.set_ylabel(flight.attitudeFrequencyResponse.getOutputs()[0])
        ax1.grid(True)

        ax2 = plt.subplot(222)
        for index, flight in enumerate(self.trajectory_list):
            ax2.plot(
                flight.omega1FrequencyResponse[:, 0],
                flight.omega1FrequencyResponse[:, 1],
                # color=self.colors_scale[index],
            )
        ax2.set_title(
            "{} x {}".format(
                flight.omega1FrequencyResponse.getOutputs()[0],
                flight.omega1FrequencyResponse.getInputs()[0],
            )
        )
        ax2.set_xlabel(flight.omega1FrequencyResponse.getInputs()[0])
        ax2.set_ylabel(flight.omega1FrequencyResponse.getOutputs()[0])
        ax2.grid(True)

        ax3 = plt.subplot(223)
        for index, flight in enumerate(self.trajectory_list):
            ax3.plot(
                flight.omega2FrequencyResponse[:, 0],
                flight.omega2FrequencyResponse[:, 1],
                # color=self.colors_scale[index],
            )
        ax3.set_title(
            "{} x {}".format(
                flight.omega2FrequencyResponse.getOutputs()[0],
                flight.omega2FrequencyResponse.getInputs()[0],
            )
        )
        ax3.set_xlabel(flight.omega2FrequencyResponse.getInputs()[0])
        ax3.set_ylabel(flight.omega2FrequencyResponse.getOutputs()[0])
        ax3.grid(True)

        ax4 = plt.subplot(224)
        for index, flight in enumerate(self.trajectory_list):
            ax4.plot(
                flight.omega3FrequencyResponse[:, 0],
                flight.omega3FrequencyResponse[:, 1],
                # color=self.colors_scale[index],
            )

        ax4.set_title(
            "{} x {}".format(
                flight.omega3FrequencyResponse.getOutputs()[0],
                flight.omega3FrequencyResponse.getInputs()[0],
            )
        )
        ax4.set_xlabel(flight.omega3FrequencyResponse.getInputs()[0])
        ax4.set_ylabel(flight.omega3FrequencyResponse.getOutputs()[0])
        ax4.grid(True)

        fig.legend(
            loc="upper center",
            ncol=len(self.names_list),
            fancybox=True,
            shadow=True,
            fontsize=10,
            bbox_to_anchor=(0.5, 1),
        )

        fig.tight_layout()

        return None

    def comparePressureSignals(self):
        """_summary_"""
        print("Still not implemented")
        pass

    def compareFinFlutterAnalysis(self):
        # Should only work if the fin flutter analysis was ran before. # TODO: Add boolean!
        print("Still not implemented yet!")
        return None

    @staticmethod
    def compareTrajectories3D(trajectory_list, names_list, legend=None):
        """Creates a trajectory plot combining the trajectories listed.
        This function was created based two source-codes:
        - Mateus Stano: https://github.com/RocketPy-Team/Hackathon_2020/pull/123
        - Dyllon Preston: https://github.com/Dyllon-P/MBS-Template/blob/main/MBS.py
        Also, some of the credits go to Georgia Tech Experimental Rocketry Club (GTXR)
        as well.
        The final function was created by the RocketPy Team.

        Parameters
        ----------
        trajectory_list : list, array
            List of trajectories. Must be in the form of [trajectory_1, trajectory_2, ..., trajectory_n]
            where each element is a list with the arrays regarding positions in x, y and z [x, y, z].
            The trajectories must be in the same reference frame. The z coordinate must be referenced
            to the ground or to the sea level, but it is important that all trajectories are passed
            in the same reference.
        names_list : list, optional
            List of strings with the name of each trajectory inputted. The names must be in
            the same order as the trajectories in trajectory_list. If no names are passed, the
            trajectories will be named as "Trajectory 1", "Trajectory 2", ..., "Trajectory n".
        legend : boolean, optional
            Whether legend will or will not be plotted. Default is True

        Returns
        -------
        None
        """

        # TODO: Allow the user to set the colors or color style
        # TODO: Allow the user to set the line style

        # Initialize variables
        maxX, maxY, maxZ, minX, minY, minZ, maxXY, minXY = 0, 0, 0, 0, 0, 0, 0, 0

        # Create the figure
        fig1 = plt.figure(figsize=(7, 7))
        fig1.suptitle("Flight Trajectories Comparison", fontsize=16, y=0.95, x=0.5)
        ax1 = plt.subplot(
            111,
            projection="3d",
        )

        # Iterate through trajectories
        for index, flight in enumerate(trajectory_list):

            x, y, z = flight

            # Find max/min values for each component
            maxX = max(x) if max(x) > maxX else maxX
            maxY = max(y) if max(y) > maxY else maxY
            maxZ = max(z) if max(z) > maxZ else maxZ
            minX = min(x) if min(x) < minX else minX
            minY = min(x) if min(x) < minX else minX
            minZ = min(z) if min(z) < minZ else minZ
            maxXY = max(maxX, maxY) if max(maxX, maxY) > maxXY else maxXY
            minXY = min(minX, minY) if min(minX, minY) > minXY else minXY

            # Add Trajectory as a plot in main figure
            ax1.plot(x, y, z, linewidth="2", label=names_list[index])

        # Plot settings
        # TODO: Don't know why, but tha z_label is not working properly
        ax1.scatter(0, 0, 0, color="black", s=10, marker="o")
        ax1.set_xlabel("X - East (m)")
        ax1.set_ylabel("Y - North (m)")
        ax1.set_zlabel("Z - Altitude (m)")
        ax1.set_zlim3d([minZ, maxZ])
        ax1.set_ylim3d([minXY, maxXY])
        ax1.set_xlim3d([minXY, maxXY])
        ax1.view_init(15, 45)

        # Add legend
        if legend:
            fig1.legend()

        fig1.tight_layout()

        return None

    def compareFlightTrajectories3D(self, legend=None, savefig=None):
        """Creates a trajectory plot that is the combination of the trajectories of
        the Flight objects passed via a Python list.

        Parameters
        ----------
        legend : boolean, optional
            Whether legend will or will not be included. Default is True
        savefig : string, optional
            If a string is passed, the figure will be saved in the path passed.

        Returns
        -------
        None

        """

        # Iterate through Flight objects and create a list of trajectories
        trajectory_list = []
        for index, flight in enumerate(self.trajectory_list):

            # Check post process
            if flight.postProcessed is False:
                flight.postProcess()

            # Get trajectories
            x = flight.x[:, 1]
            y = flight.y[:, 1]
            z = flight.z[:, 1] - flight.env.elevation
            trajectory_list.append([x, y, z])

        # Call compareTrajectories3D function to do the hard work
        self.compareTrajectories3D(trajectory_list, self.names_list, legend=legend)

        return None

    def compareFlightTrajectories2D(self, legend=None):
        """...
        Let it chose the two planes...
        - XY projection plot
        - XZ projection plot
        - YZ projection plot
        """
        print("Still not implemented yet!")
        pass

    @staticmethod
    def compareFlightSimulators():
        """Allow the user to compare a flight from RocketPy (or more than one)
        against a flight from another simulator (e.g. OpenRocket, Cambridge, etc.)
        Still not implemented yet.
        Should also allow comparison between RocketPy and actual flight data.
        """
        print("Still not implemented yet!")
        pass

    # Start definition of animations methods

    def animate(self, start=0, stop=None, fps=12, speed=4, elev=None, azim=None):
        """Plays an animation of the flight. Not implemented yet. Only
        kinda works outside notebook.
        """
        for index, flight in enumerate(self.trajectory_list):
            # Set up stopping time
            stop = flight.tFinal if stop is None else stop
            # Speed = 4 makes it almost real time - matplotlib is way to slow
            # Set up graph
            fig = plt.figure(figsize=(18, 15))
            fig.suptitle("Flight: {}".format(self.names_list[index]))
            axes = fig.gca(projection="3d")
            # Initialize time
            timeRange = np.linspace(start, stop, fps * (stop - start))
            # Initialize first frame
            axes.set_title("Trajectory and Velocity Animation")
            axes.set_xlabel("X (m)")
            axes.set_ylabel("Y (m)")
            axes.set_zlabel("Z (m)")
            axes.view_init(elev, azim)
            R = axes.quiver(0, 0, 0, 0, 0, 0, color="r", label="Rocket")
            V = axes.quiver(0, 0, 0, 0, 0, 0, color="g", label="Velocity")
            W = axes.quiver(0, 0, 0, 0, 0, 0, color="b", label="Wind")
            S = axes.quiver(0, 0, 0, 0, 0, 0, color="black", label="Freestream")
            axes.legend()
            # Animate
            for t in timeRange:
                R.remove()
                V.remove()
                W.remove()
                S.remove()
                # Calculate rocket position
                Rx, Ry, Rz = flight.x(t), flight.y(t), flight.z(t)
                Ru = 1 * (
                    2 * (flight.e1(t) * flight.e3(t) + flight.e0(t) * flight.e2(t))
                )
                Rv = 1 * (
                    2 * (flight.e2(t) * flight.e3(t) - flight.e0(t) * flight.e1(t))
                )
                Rw = 1 * (1 - 2 * (flight.e1(t) ** 2 + flight.e2(t) ** 2))
                # Calculate rocket Mach number
                Vx = flight.vx(t) / 340.40
                Vy = flight.vy(t) / 340.40
                Vz = flight.vz(t) / 340.40
                # Calculate wind Mach Number
                z = flight.z(t)
                Wx = flight.env.windVelocityX(z) / 20
                Wy = flight.env.windVelocityY(z) / 20
                # Calculate freestream Mach Number
                Sx = flight.streamVelocityX(t) / 340.40
                Sy = flight.streamVelocityY(t) / 340.40
                Sz = flight.streamVelocityZ(t) / 340.40
                # Plot Quivers
                R = axes.quiver(Rx, Ry, Rz, Ru, Rv, Rw, color="r")
                V = axes.quiver(Rx, Ry, Rz, -Vx, -Vy, -Vz, color="g")
                W = axes.quiver(Rx - Vx, Ry - Vy, Rz - Vz, Wx, Wy, 0, color="b")
                S = axes.quiver(Rx, Ry, Rz, Sx, Sy, Sz, color="black")
                # Adjust axis
                axes.set_xlim(Rx - 1, Rx + 1)
                axes.set_ylim(Ry - 1, Ry + 1)
                axes.set_zlim(Rz - 1, Rz + 1)
                # plt.pause(1/(fps*speed))
                try:
                    plt.pause(1 / (fps * speed))
                except:
                    time.sleep(1 / (fps * speed))

    def info(self):
        """Prints out a summary of the data available about the Flight.

        Parameters
        ----------
        None

        Return
        ------
        None
        """

        # Print initial conditions
        self.printInitialConditionsData()

        # Print surface wind conditions
        self.printSurfaceWindConditions()

        # Print launch rail orientation
        self.printLaunchRailConditions()

        # Print out of rail conditions
        self.printOutOfRailConditions()

        # Print burnOut conditions
        self.printBurnOutConditions()

        # Print apogee conditions
        self.printApogeeConditions()

        # Print events registered
        self.printEventsRegistered()

        # Print impact conditions
        self.printImpactConditions()

        # Print maximum values
        self.printMaximumValues()

        # Print Numerical Integration Information
        self.printNumericalIntegrationSettings()

        return None

    def allInfo(self, mode="basic"):
        """Prints out all data and graphs available about the Flight.
        It call info() and then all the plots available.

        Parameters
        ----------
        mode : str, optional
            The level of detail to print. The default is "basic".
            Options are "compare" and "basic".
            "compare" prints all data and graphs available.
            "basic" prints will basically repeat the code inside a for loop.

        Return
        ------
        None
        """
        if mode == "basic":

            # Print a summary of data about the flight
            self.info()

            # Plot flight trajectory in a 3D plot
            self.plot3dTrajectory()

            # Plot
            self.plotLinearKinematicsData()

            # Plot
            self.plotFlightPathAngleData()

            # Plot
            self.plotAttitudeData()

            # Plot
            self.plotAngularKinematicsData()

            # Plot
            self.plotTrajectoryForceData()

            # Plot
            self.plotEnergyData()

            # Plot
            self.plotFluidMechanicsData()

            # Plot pressure signals recorded by the sensors
            self.plotPressureSignals()

            # Plot Stability and Control Data
            self.plotStabilityAndControlData()

        elif mode == "compare":

            self.info()

            self.compareFlightTrajectories3D()

            self.comparePositions()

            self.compareVelocities()

            self.compareStreamVelocities()

            self.compareAccelerations()

            self.compareAngularVelocities()

            self.compareAngularAccelerations()

            self.compareEulerAngles()

            self.compareQuaternions()

            self.compareAttitudeAngles()

            self.compareAnglesOfAttack()

            self.compareStaticMargins()

            self.compareAerodynamicForces()

            self.compareAerodynamicMoments()

            self.compareRailButtonsForces()

            self.compareEnergies()

            self.comparePowers()

            self.compareFluidMechanics()

            # self.comparePressureSignals()

            # self.compareFinFlutterAnalysis()

            self.compareAttitudeFrequencyResponses()

        else:
            raise ValueError("Mode must be 'basic' or 'compare'")

        return None
