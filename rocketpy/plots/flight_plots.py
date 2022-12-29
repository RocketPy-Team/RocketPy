__author__ = "Guilherme Fernandes Alves, Mateus Stano Junqueira"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"


import matplotlib.pyplot as plt
import numpy as np

try:
    from functools import cached_property
except ImportError:
    from ..tools import cached_property


class _FlightPlots:
    """Class that holds plot methods for Flight class.

    Attributes
    ----------
    _FlightPlots.flight : Flight
        Flight object that will be used for the plots.

    _FlightPlots.first_event_time : float
        Time of first event.

    _FlightPlots.first_event_time_index : int
        Time index of first event.
    """

    def __init__(self, flight):
        """Initializes _FlightPlots class.

        Parameters
        ----------
        flight : Flight
            Instance of the Flight class

        Returns
        -------
        None
        """
        self.flight = flight
        return None

    @cached_property
    def first_event_time(self):
        """Time of the first flight event."""
        if len(self.flight.parachuteEvents) > 0:
            return (
                self.flight.parachuteEvents[0][0]
                + self.flight.parachuteEvents[0][1].lag
            )
        else:
            return self.flight.tFinal

    @cached_property
    def first_event_time_index(self):
        """Time index of the first flight event."""
        if len(self.flight.parachuteEvents) > 0:
            return np.nonzero(self.flight.x[:, 0] == self.first_event_time)[0][0]
        else:
            return -1

    def trajectory_3d(self):
        """Plot a 3D graph of the trajectory

        Parameters
        ----------
        None

        Return
        ------
        None
        """

        # Get max and min x and y
        maxZ = max(self.flight.z[:, 1] - self.flight.env.elevation)
        maxX = max(self.flight.x[:, 1])
        minX = min(self.flight.x[:, 1])
        maxY = max(self.flight.y[:, 1])
        minY = min(self.flight.y[:, 1])
        maxXY = max(maxX, maxY)
        minXY = min(minX, minY)

        # Create figure
        fig1 = plt.figure(figsize=(9, 9))
        ax1 = plt.subplot(111, projection="3d")
        ax1.plot(
            self.flight.x[:, 1], self.flight.y[:, 1], zs=0, zdir="z", linestyle="--"
        )
        ax1.plot(
            self.flight.x[:, 1],
            self.flight.z[:, 1] - self.flight.env.elevation,
            zs=minXY,
            zdir="y",
            linestyle="--",
        )
        ax1.plot(
            self.flight.y[:, 1],
            self.flight.z[:, 1] - self.flight.env.elevation,
            zs=minXY,
            zdir="x",
            linestyle="--",
        )
        ax1.plot(
            self.flight.x[:, 1],
            self.flight.y[:, 1],
            self.flight.z[:, 1] - self.flight.env.elevation,
            linewidth="2",
        )
        ax1.scatter(0, 0, 0)
        ax1.set_xlabel("X - East (m)")
        ax1.set_ylabel("Y - North (m)")
        ax1.set_zlabel("Z - Altitude Above Ground Level (m)")
        ax1.set_title("Flight Trajectory")
        ax1.set_zlim3d([0, maxZ])
        ax1.set_ylim3d([minXY, maxXY])
        ax1.set_xlim3d([minXY, maxXY])
        ax1.view_init(15, 45)
        plt.show()

        return None

    def linear_kinematics_data(self):
        """Prints out all Kinematics graphs available about the Flight

        Parameters
        ----------
        None

        Return
        ------
        None
        """

        # Velocity and acceleration plots
        fig2 = plt.figure(figsize=(9, 12))

        ax1 = plt.subplot(414)
        ax1.plot(self.flight.vx[:, 0], self.flight.vx[:, 1], color="#ff7f0e")
        ax1.set_xlim(0, self.flight.tFinal)
        ax1.set_title("Velocity X | Acceleration X")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Velocity X (m/s)", color="#ff7f0e")
        ax1.tick_params("y", colors="#ff7f0e")
        ax1.grid(True)

        ax1up = ax1.twinx()
        ax1up.plot(self.flight.ax[:, 0], self.flight.ax[:, 1], color="#1f77b4")
        ax1up.set_ylabel("Acceleration X (m/s²)", color="#1f77b4")
        ax1up.tick_params("y", colors="#1f77b4")

        ax2 = plt.subplot(413)
        ax2.plot(self.flight.vy[:, 0], self.flight.vy[:, 1], color="#ff7f0e")
        ax2.set_xlim(0, self.flight.tFinal)
        ax2.set_title("Velocity Y | Acceleration Y")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Velocity Y (m/s)", color="#ff7f0e")
        ax2.tick_params("y", colors="#ff7f0e")
        ax2.grid(True)

        ax2up = ax2.twinx()
        ax2up.plot(self.flight.ay[:, 0], self.flight.ay[:, 1], color="#1f77b4")
        ax2up.set_ylabel("Acceleration Y (m/s²)", color="#1f77b4")
        ax2up.tick_params("y", colors="#1f77b4")

        ax3 = plt.subplot(412)
        ax3.plot(self.flight.vz[:, 0], self.flight.vz[:, 1], color="#ff7f0e")
        ax3.set_xlim(0, self.flight.tFinal)
        ax3.set_title("Velocity Z | Acceleration Z")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Velocity Z (m/s)", color="#ff7f0e")
        ax3.tick_params("y", colors="#ff7f0e")
        ax3.grid(True)

        ax3up = ax3.twinx()
        ax3up.plot(self.flight.az[:, 0], self.flight.az[:, 1], color="#1f77b4")
        ax3up.set_ylabel("Acceleration Z (m/s²)", color="#1f77b4")
        ax3up.tick_params("y", colors="#1f77b4")

        ax4 = plt.subplot(411)
        ax4.plot(self.flight.speed[:, 0], self.flight.speed[:, 1], color="#ff7f0e")
        ax4.set_xlim(0, self.flight.tFinal)
        ax4.set_title("Velocity Magnitude | Acceleration Magnitude")
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Velocity (m/s)", color="#ff7f0e")
        ax4.tick_params("y", colors="#ff7f0e")
        ax4.grid(True)

        ax4up = ax4.twinx()
        ax4up.plot(
            self.flight.acceleration[:, 0],
            self.flight.acceleration[:, 1],
            color="#1f77b4",
        )
        ax4up.set_ylabel("Acceleration (m/s²)", color="#1f77b4")
        ax4up.tick_params("y", colors="#1f77b4")

        plt.subplots_adjust(hspace=0.5)
        plt.show()
        return None

    def attitude_data(self):
        """Prints out all Angular position graphs available about the Flight

        Parameters
        ----------
        None

        Return
        ------
        None
        """

        # Angular position plots
        fig3 = plt.figure(figsize=(9, 12))

        ax1 = plt.subplot(411)
        ax1.plot(self.flight.e0[:, 0], self.flight.e0[:, 1], label="$e_0$")
        ax1.plot(self.flight.e1[:, 0], self.flight.e1[:, 1], label="$e_1$")
        ax1.plot(self.flight.e2[:, 0], self.flight.e2[:, 1], label="$e_2$")
        ax1.plot(self.flight.e3[:, 0], self.flight.e3[:, 1], label="$e_3$")
        ax1.set_xlim(0, self.first_event_time)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Euler Parameters")
        ax1.set_title("Euler Parameters")
        ax1.legend()
        ax1.grid(True)

        ax2 = plt.subplot(412)
        ax2.plot(self.flight.psi[:, 0], self.flight.psi[:, 1])
        ax2.set_xlim(0, self.first_event_time)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("ψ (°)")
        ax2.set_title("Euler Precession Angle")
        ax2.grid(True)

        ax3 = plt.subplot(413)
        ax3.plot(self.flight.theta[:, 0], self.flight.theta[:, 1], label="θ - Nutation")
        ax3.set_xlim(0, self.first_event_time)
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("θ (°)")
        ax3.set_title("Euler Nutation Angle")
        ax3.grid(True)

        ax4 = plt.subplot(414)
        ax4.plot(self.flight.phi[:, 0], self.flight.phi[:, 1], label="φ - Spin")
        ax4.set_xlim(0, self.first_event_time)
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("φ (°)")
        ax4.set_title("Euler Spin Angle")
        ax4.grid(True)

        plt.subplots_adjust(hspace=0.5)
        plt.show()

        return None

    def flight_path_angle_data(self):
        """Prints out Flight path and Rocket Attitude angle graphs available
        about the Flight

        Parameters
        ----------
        None

        Return
        ------
        None
        """

        # Path, Attitude and Lateral Attitude Angle
        # Angular position plots
        fig5 = plt.figure(figsize=(9, 6))

        ax1 = plt.subplot(211)
        ax1.plot(
            self.flight.pathAngle[:, 0],
            self.flight.pathAngle[:, 1],
            label="Flight Path Angle",
        )
        ax1.plot(
            self.flight.attitudeAngle[:, 0],
            self.flight.attitudeAngle[:, 1],
            label="Rocket Attitude Angle",
        )
        ax1.set_xlim(0, self.first_event_time)
        ax1.legend()
        ax1.grid(True)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Angle (°)")
        ax1.set_title("Flight Path and Attitude Angle")

        ax2 = plt.subplot(212)
        ax2.plot(
            self.flight.lateralAttitudeAngle[:, 0],
            self.flight.lateralAttitudeAngle[:, 1],
        )
        ax2.set_xlim(0, self.first_event_time)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Lateral Attitude Angle (°)")
        ax2.set_title("Lateral Attitude Angle")
        ax2.grid(True)

        plt.subplots_adjust(hspace=0.5)
        plt.show()

        return None

    def angular_kinematics_data(self):
        """Prints out all Angular velocity and acceleration graphs available
        about the Flight

        Parameters
        ----------
        None

        Return
        ------
        None
        """

        # Angular velocity and acceleration plots
        fig4 = plt.figure(figsize=(9, 9))
        ax1 = plt.subplot(311)
        ax1.plot(self.flight.w1[:, 0], self.flight.w1[:, 1], color="#ff7f0e")
        ax1.set_xlim(0, self.first_event_time)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel(r"Angular Velocity - ${\omega_1}$ (rad/s)", color="#ff7f0e")
        ax1.set_title(
            r"Angular Velocity ${\omega_1}$ | Angular Acceleration ${\alpha_1}$"
        )
        ax1.tick_params("y", colors="#ff7f0e")
        ax1.grid(True)

        ax1up = ax1.twinx()
        ax1up.plot(self.flight.alpha1[:, 0], self.flight.alpha1[:, 1], color="#1f77b4")
        ax1up.set_ylabel(
            r"Angular Acceleration - ${\alpha_1}$ (rad/s²)", color="#1f77b4"
        )
        ax1up.tick_params("y", colors="#1f77b4")

        ax2 = plt.subplot(312)
        ax2.plot(self.flight.w2[:, 0], self.flight.w2[:, 1], color="#ff7f0e")
        ax2.set_xlim(0, self.first_event_time)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel(r"Angular Velocity - ${\omega_2}$ (rad/s)", color="#ff7f0e")
        ax2.set_title(
            r"Angular Velocity ${\omega_2}$ | Angular Acceleration ${\alpha_2}$"
        )
        ax2.tick_params("y", colors="#ff7f0e")
        ax2.grid(True)

        ax2up = ax2.twinx()
        ax2up.plot(self.flight.alpha2[:, 0], self.flight.alpha2[:, 1], color="#1f77b4")
        ax2up.set_ylabel(
            r"Angular Acceleration - ${\alpha_2}$ (rad/s²)", color="#1f77b4"
        )
        ax2up.tick_params("y", colors="#1f77b4")

        ax3 = plt.subplot(313)
        ax3.plot(self.flight.w3[:, 0], self.flight.w3[:, 1], color="#ff7f0e")
        ax3.set_xlim(0, self.first_event_time)
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel(r"Angular Velocity - ${\omega_3}$ (rad/s)", color="#ff7f0e")
        ax3.set_title(
            r"Angular Velocity ${\omega_3}$ | Angular Acceleration ${\alpha_3}$"
        )
        ax3.tick_params("y", colors="#ff7f0e")
        ax3.grid(True)

        ax3up = ax3.twinx()
        ax3up.plot(self.flight.alpha3[:, 0], self.flight.alpha3[:, 1], color="#1f77b4")
        ax3up.set_ylabel(
            r"Angular Acceleration - ${\alpha_3}$ (rad/s²)", color="#1f77b4"
        )
        ax3up.tick_params("y", colors="#1f77b4")

        plt.subplots_adjust(hspace=0.5)
        plt.show()

        return None

    def trajectory_force_data(self):
        """Prints out all Forces and Moments graphs available about the Flight

        Parameters
        ----------
        None

        Return
        ------
        None
        """

        # Rail Button Forces
        fig6 = plt.figure(figsize=(9, 6))

        ax1 = plt.subplot(211)
        ax1.plot(
            self.flight.railButton1NormalForce[: self.flight.outOfRailTimeIndex, 0],
            self.flight.railButton1NormalForce[: self.flight.outOfRailTimeIndex, 1],
            label="Upper Rail Button",
        )
        ax1.plot(
            self.flight.railButton2NormalForce[: self.flight.outOfRailTimeIndex, 0],
            self.flight.railButton2NormalForce[: self.flight.outOfRailTimeIndex, 1],
            label="Lower Rail Button",
        )
        ax1.set_xlim(
            0,
            self.flight.outOfRailTime
            if self.flight.outOfRailTime > 0
            else self.flight.tFinal,
        )
        ax1.legend()
        ax1.grid(True)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Normal Force (N)")
        ax1.set_title("Rail Buttons Normal Force")

        ax2 = plt.subplot(212)
        ax2.plot(
            self.flight.railButton1ShearForce[: self.flight.outOfRailTimeIndex, 0],
            self.flight.railButton1ShearForce[: self.flight.outOfRailTimeIndex, 1],
            label="Upper Rail Button",
        )
        ax2.plot(
            self.flight.railButton2ShearForce[: self.flight.outOfRailTimeIndex, 0],
            self.flight.railButton2ShearForce[: self.flight.outOfRailTimeIndex, 1],
            label="Lower Rail Button",
        )
        ax2.set_xlim(
            0,
            self.flight.outOfRailTime
            if self.flight.outOfRailTime > 0
            else self.flight.tFinal,
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

        ax1 = plt.subplot(411)
        ax1.plot(
            self.flight.aerodynamicLift[: self.first_event_time_index, 0],
            self.flight.aerodynamicLift[: self.first_event_time_index, 1],
            label="Resultant",
        )
        ax1.plot(
            self.flight.R1[: self.first_event_time_index, 0],
            self.flight.R1[: self.first_event_time_index, 1],
            label="R1",
        )
        ax1.plot(
            self.flight.R2[: self.first_event_time_index, 0],
            self.flight.R2[: self.first_event_time_index, 1],
            label="R2",
        )
        ax1.set_xlim(0, self.first_event_time)
        ax1.legend()
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Lift Force (N)")
        ax1.set_title("Aerodynamic Lift Resultant Force")
        ax1.grid()

        ax2 = plt.subplot(412)
        ax2.plot(
            self.flight.aerodynamicDrag[: self.first_event_time_index, 0],
            self.flight.aerodynamicDrag[: self.first_event_time_index, 1],
        )
        ax2.set_xlim(0, self.first_event_time)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Drag Force (N)")
        ax2.set_title("Aerodynamic Drag Force")
        ax2.grid()

        ax3 = plt.subplot(413)
        ax3.plot(
            self.flight.aerodynamicBendingMoment[: self.first_event_time_index, 0],
            self.flight.aerodynamicBendingMoment[: self.first_event_time_index, 1],
            label="Resultant",
        )
        ax3.plot(
            self.flight.M1[: self.first_event_time_index, 0],
            self.flight.M1[: self.first_event_time_index, 1],
            label="M1",
        )
        ax3.plot(
            self.flight.M2[: self.first_event_time_index, 0],
            self.flight.M2[: self.first_event_time_index, 1],
            label="M2",
        )
        ax3.set_xlim(0, self.first_event_time)
        ax3.legend()
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Bending Moment (N m)")
        ax3.set_title("Aerodynamic Bending Resultant Moment")
        ax3.grid()

        ax4 = plt.subplot(414)
        ax4.plot(
            self.flight.aerodynamicSpinMoment[: self.first_event_time_index, 0],
            self.flight.aerodynamicSpinMoment[: self.first_event_time_index, 1],
        )
        ax4.set_xlim(0, self.first_event_time)
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Spin Moment (N m)")
        ax4.set_title("Aerodynamic Spin Moment")
        ax4.grid()

        plt.subplots_adjust(hspace=0.5)
        plt.show()

        return None

    def energy_data(self):
        """Prints out all Energy components graphs available about the Flight

        Returns
        -------
        None
        """

        fig8 = plt.figure(figsize=(9, 9))

        ax1 = plt.subplot(411)
        ax1.plot(
            self.flight.kineticEnergy[:, 0],
            self.flight.kineticEnergy[:, 1],
            label="Kinetic Energy",
        )
        ax1.plot(
            self.flight.rotationalEnergy[:, 0],
            self.flight.rotationalEnergy[:, 1],
            label="Rotational Energy",
        )
        ax1.plot(
            self.flight.translationalEnergy[:, 0],
            self.flight.translationalEnergy[:, 1],
            label="Translational Energy",
        )
        ax1.set_xlim(
            0,
            self.flight.apogeeTime
            if self.flight.apogeeTime != 0.0
            else self.flight.tFinal,
        )
        ax1.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax1.set_title("Kinetic Energy Components")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Energy (J)")

        ax1.legend()
        ax1.grid()

        ax2 = plt.subplot(412)
        ax2.plot(
            self.flight.totalEnergy[:, 0],
            self.flight.totalEnergy[:, 1],
            label="Total Energy",
        )
        ax2.plot(
            self.flight.kineticEnergy[:, 0],
            self.flight.kineticEnergy[:, 1],
            label="Kinetic Energy",
        )
        ax2.plot(
            self.flight.potentialEnergy[:, 0],
            self.flight.potentialEnergy[:, 1],
            label="Potential Energy",
        )
        ax2.set_xlim(
            0,
            self.flight.apogeeTime
            if self.flight.apogeeTime != 0.0
            else self.flight.tFinal,
        )
        ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax2.set_title("Total Mechanical Energy Components")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Energy (J)")
        ax2.legend()
        ax2.grid()

        ax3 = plt.subplot(413)
        ax3.plot(
            self.flight.thrustPower[:, 0],
            self.flight.thrustPower[:, 1],
            label="|Thrust Power|",
        )
        ax3.set_xlim(0, self.flight.rocket.motor.burnOutTime)
        ax3.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax3.set_title("Thrust Absolute Power")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Power (W)")
        ax3.legend()
        ax3.grid()

        ax4 = plt.subplot(414)
        ax4.plot(
            self.flight.dragPower[:, 0],
            -self.flight.dragPower[:, 1],
            label="|Drag Power|",
        )
        ax4.set_xlim(
            0,
            self.flight.apogeeTime
            if self.flight.apogeeTime != 0.0
            else self.flight.tFinal,
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

    def fluid_mechanics_data(self):
        """Prints out a summary of the Fluid Mechanics graphs available about
        the Flight

        Parameters
        ----------
        None

        Return
        ------
        None
        """

        # Trajectory Fluid Mechanics Plots
        fig10 = plt.figure(figsize=(9, 12))

        ax1 = plt.subplot(411)
        ax1.plot(self.flight.MachNumber[:, 0], self.flight.MachNumber[:, 1])
        ax1.set_xlim(0, self.flight.tFinal)
        ax1.set_title("Mach Number")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Mach Number")
        ax1.grid()

        ax2 = plt.subplot(412)
        ax2.plot(self.flight.ReynoldsNumber[:, 0], self.flight.ReynoldsNumber[:, 1])
        ax2.set_xlim(0, self.flight.tFinal)
        ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax2.set_title("Reynolds Number")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Reynolds Number")
        ax2.grid()

        ax3 = plt.subplot(413)
        ax3.plot(
            self.flight.dynamicPressure[:, 0],
            self.flight.dynamicPressure[:, 1],
            label="Dynamic Pressure",
        )
        ax3.plot(
            self.flight.totalPressure[:, 0],
            self.flight.totalPressure[:, 1],
            label="Total Pressure",
        )
        ax3.plot(
            self.flight.pressure[:, 0],
            self.flight.pressure[:, 1],
            label="Static Pressure",
        )
        ax3.set_xlim(0, self.flight.tFinal)
        ax3.legend()
        ax3.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax3.set_title("Total and Dynamic Pressure")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Pressure (Pa)")
        ax3.grid()

        ax4 = plt.subplot(414)
        ax4.plot(self.flight.angleOfAttack[:, 0], self.flight.angleOfAttack[:, 1])
        # Make sure bottom and top limits are different
        if (
            self.flight.outOfRailTime
            * self.flight.angleOfAttack(self.flight.outOfRailTime)
            != 0
        ):
            ax4.set_xlim(self.flight.outOfRailTime, 10 * self.flight.outOfRailTime + 1)
            ax4.set_ylim(0, self.flight.angleOfAttack(self.flight.outOfRailTime))
        ax4.set_title("Angle of Attack")
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Angle of Attack (°)")
        ax4.grid()

        plt.subplots_adjust(hspace=0.5)
        plt.show()

        return None

    def stability_and_control_data(self):
        """Prints out Rocket Stability and Control parameters graphs available
        about the Flight

        Parameters
        ----------
        None

        Return
        ------
        None
        """

        fig9 = plt.figure(figsize=(9, 6))

        ax1 = plt.subplot(211)
        ax1.plot(self.flight.staticMargin[:, 0], self.flight.staticMargin[:, 1])
        ax1.set_xlim(0, self.flight.staticMargin[:, 0][-1])
        ax1.set_title("Static Margin")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Static Margin (c)")
        ax1.grid()

        ax2 = plt.subplot(212)
        maxAttitude = max(self.flight.attitudeFrequencyResponse[:, 1])
        maxAttitude = maxAttitude if maxAttitude != 0 else 1
        ax2.plot(
            self.flight.attitudeFrequencyResponse[:, 0],
            self.flight.attitudeFrequencyResponse[:, 1] / maxAttitude,
            label="Attitude Angle",
        )
        maxOmega1 = max(self.flight.omega1FrequencyResponse[:, 1])
        maxOmega1 = maxOmega1 if maxOmega1 != 0 else 1
        ax2.plot(
            self.flight.omega1FrequencyResponse[:, 0],
            self.flight.omega1FrequencyResponse[:, 1] / maxOmega1,
            label=r"$\omega_1$",
        )
        maxOmega2 = max(self.flight.omega2FrequencyResponse[:, 1])
        maxOmega2 = maxOmega2 if maxOmega2 != 0 else 1
        ax2.plot(
            self.flight.omega2FrequencyResponse[:, 0],
            self.flight.omega2FrequencyResponse[:, 1] / maxOmega2,
            label=r"$\omega_2$",
        )
        maxOmega3 = max(self.flight.omega3FrequencyResponse[:, 1])
        maxOmega3 = maxOmega3 if maxOmega3 != 0 else 1
        ax2.plot(
            self.flight.omega3FrequencyResponse[:, 0],
            self.flight.omega3FrequencyResponse[:, 1] / maxOmega3,
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

    def pressure_rocket_altitude(self):
        """Plots out pressure at rocket's altitude.

        Parameters
        ----------
        None

        Return
        ------
        None
        """

        plt.figure()
        ax1 = plt.subplot(111)
        ax1.plot(self.flight.z[:, 0], self.flight.env.pressure(self.flight.z[:, 1]))
        ax1.set_title("Pressure at Rocket's Altitude")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Pressure (Pa)")
        ax1.set_xlim(0, self.flight.tFinal)
        ax1.grid()

        plt.show()

        return None

    def pressure_signals(self):
        """Plots out all Parachute Trigger Pressure Signals.
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

        if len(self.flight.parachuteEvents) > 0:
            for parachute in self.flight.rocket.parachutes:
                print("\nParachute: ", parachute.name)
                self.flight._calculate_pressure_signal()
                parachute.noiseSignalFunction()
                parachute.noisyPressureSignalFunction()
                parachute.cleanPressureSignalFunction()
        else:
            print("\nRocket has no parachutes. No parachute plots available")

        return None

    def all(self):
        """Prints out all plots available about the Flight.

        Parameters
        ----------
        None

        Return
        ------
        None
        """

        print("\n\nTrajectory 3d Plot\n")
        self.trajectory_3d()

        print("\n\nTrajectory Kinematic Plots\n")
        self.linear_kinematics_data()

        print("\n\nAngular Position Plots\n")
        self.flight_path_angle_data()

        print("\n\nPath, Attitude and Lateral Attitude Angle plots\n")
        self.attitude_data()

        print("\n\nTrajectory Angular Velocity and Acceleration Plots\n")
        self.angular_kinematics_data()

        print("\n\nTrajectory Force Plots\n")
        self.trajectory_force_data()

        print("\n\nTrajectory Energy Plots\n")
        self.energy_data()

        print("\n\nTrajectory Fluid Mechanics Plots\n")
        self.fluid_mechanics_data()

        print("\n\nTrajectory Stability and Control Plots\n")
        self.stability_and_control_data()

        print("\n\nRocket and Parachute Pressure Plots\n")
        self.pressure_rocket_altitude()
        self.pressure_signals()

        return None
