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
        if len(self.flight.parachute_events) > 0:
            return (
                self.flight.parachute_events[0][0]
                + self.flight.parachute_events[0][1].lag
            )
        else:
            return self.flight.t_final

    @cached_property
    def first_event_time_index(self):
        """Time index of the first flight event."""
        if len(self.flight.parachute_events) > 0:
            return np.nonzero(self.flight.x[:, 0] == self.first_event_time)[0][0]
        else:
            return -1

    def trajectory_3d(self):
        """Plot a 3D graph of the trajectory

        Returns
        -------
        None
        """

        # Get max and min x and y
        max_z = max(self.flight.z[:, 1] - self.flight.env.elevation)
        max_x = max(self.flight.x[:, 1])
        min_x = min(self.flight.x[:, 1])
        max_y = max(self.flight.y[:, 1])
        min_y = min(self.flight.y[:, 1])
        max_xy = max(max_x, max_y)
        min_xy = min(min_x, min_y)

        # Create figure
        fig1 = plt.figure(figsize=(9, 9))
        ax1 = plt.subplot(111, projection="3d")
        ax1.plot(
            self.flight.x[:, 1], self.flight.y[:, 1], zs=0, zdir="z", linestyle="--"
        )
        ax1.plot(
            self.flight.x[:, 1],
            self.flight.z[:, 1] - self.flight.env.elevation,
            zs=min_xy,
            zdir="y",
            linestyle="--",
        )
        ax1.plot(
            self.flight.y[:, 1],
            self.flight.z[:, 1] - self.flight.env.elevation,
            zs=min_xy,
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
        ax1.set_zlim3d([0, max_z])
        ax1.set_ylim3d([min_xy, max_xy])
        ax1.set_xlim3d([min_xy, max_xy])
        ax1.view_init(15, 45)
        plt.show()

        return None

    def linear_kinematics_data(self):
        """Prints out all Kinematics graphs available about the Flight

        Returns
        -------
        None
        """

        # Velocity and acceleration plots
        fig2 = plt.figure(figsize=(9, 12))

        ax1 = plt.subplot(414)
        ax1.plot(self.flight.vx[:, 0], self.flight.vx[:, 1], color="#ff7f0e")
        ax1.set_xlim(0, self.flight.t_final)
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
        ax2.set_xlim(0, self.flight.t_final)
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
        ax3.set_xlim(0, self.flight.t_final)
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
        ax4.set_xlim(0, self.flight.t_final)
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

        Returns
        -------
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

        Returns
        -------
        None
        """

        # Path, Attitude and Lateral Attitude Angle
        # Angular position plots
        fig5 = plt.figure(figsize=(9, 6))

        ax1 = plt.subplot(211)
        ax1.plot(
            self.flight.path_angle[:, 0],
            self.flight.path_angle[:, 1],
            label="Flight Path Angle",
        )
        ax1.plot(
            self.flight.attitude_angle[:, 0],
            self.flight.attitude_angle[:, 1],
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
            self.flight.lateral_attitude_angle[:, 0],
            self.flight.lateral_attitude_angle[:, 1],
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

        Returns
        -------
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

    def rail_buttons_forces(self):
        """Prints out all Rail Buttons Forces graphs available about the Flight.

        Returns
        -------
        None
        """
        if len(self.flight.rocket.rail_buttons) == 0:
            print("No rail buttons were defined. Skipping rail button plots.")
        elif self.flight.out_of_rail_time_index == 0:
            print("No rail phase was found. Skipping rail button plots.")
        else:
            fig6 = plt.figure(figsize=(9, 6))

            ax1 = plt.subplot(211)
            ax1.plot(
                self.flight.rail_button1_normal_force[
                    : self.flight.out_of_rail_time_index, 0
                ],
                self.flight.rail_button1_normal_force[
                    : self.flight.out_of_rail_time_index, 1
                ],
                label="Upper Rail Button",
            )
            ax1.plot(
                self.flight.rail_button2_normal_force[
                    : self.flight.out_of_rail_time_index, 0
                ],
                self.flight.rail_button2_normal_force[
                    : self.flight.out_of_rail_time_index, 1
                ],
                label="Lower Rail Button",
            )
            ax1.set_xlim(
                0,
                self.flight.out_of_rail_time
                if self.flight.out_of_rail_time > 0
                else self.flight.tFinal,
            )
            ax1.legend()
            ax1.grid(True)
            ax1.set_xlabel(self.flight.rail_button1_normal_force.get_inputs()[0])
            ax1.set_ylabel(self.flight.rail_button1_normal_force.get_outputs()[0])
            ax1.set_title("Rail Buttons Normal Force")

            ax2 = plt.subplot(212)
            ax2.plot(
                self.flight.rail_button1_shear_force[
                    : self.flight.out_of_rail_time_index, 0
                ],
                self.flight.rail_button1_shear_force[
                    : self.flight.out_of_rail_time_index, 1
                ],
                label="Upper Rail Button",
            )
            ax2.plot(
                self.flight.rail_button2_shear_force[
                    : self.flight.out_of_rail_time_index, 0
                ],
                self.flight.rail_button2_shear_force[
                    : self.flight.out_of_rail_time_index, 1
                ],
                label="Lower Rail Button",
            )
            ax2.set_xlim(
                0,
                self.flight.out_of_rail_time
                if self.flight.out_of_rail_time > 0
                else self.flight.tFinal,
            )
            ax2.legend()
            ax2.grid(True)
            ax2.set_xlabel(self.flight.rail_button1_shear_force.__inputs__[0])
            ax2.set_ylabel(self.flight.rail_button1_shear_force.__outputs__[0])
            ax2.set_title("Rail Buttons Shear Force")

            plt.subplots_adjust(hspace=0.5)
            plt.show()
        return None

    def aerodynamic_forces(self):
        """Prints out all Forces and Moments graphs available about the Flight

        Returns
        -------
        None
        """

        # Aerodynamic force and moment plots
        fig7 = plt.figure(figsize=(9, 12))

        ax1 = plt.subplot(411)
        ax1.plot(
            self.flight.aerodynamic_lift[: self.first_event_time_index, 0],
            self.flight.aerodynamic_lift[: self.first_event_time_index, 1],
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
            self.flight.aerodynamic_drag[: self.first_event_time_index, 0],
            self.flight.aerodynamic_drag[: self.first_event_time_index, 1],
        )
        ax2.set_xlim(0, self.first_event_time)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Drag Force (N)")
        ax2.set_title("Aerodynamic Drag Force")
        ax2.grid()

        ax3 = plt.subplot(413)
        ax3.plot(
            self.flight.aerodynamic_bending_moment[: self.first_event_time_index, 0],
            self.flight.aerodynamic_bending_moment[: self.first_event_time_index, 1],
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
            self.flight.aerodynamic_spin_moment[: self.first_event_time_index, 0],
            self.flight.aerodynamic_spin_moment[: self.first_event_time_index, 1],
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
            self.flight.kinetic_energy[:, 0],
            self.flight.kinetic_energy[:, 1],
            label="Kinetic Energy",
        )
        ax1.plot(
            self.flight.rotational_energy[:, 0],
            self.flight.rotational_energy[:, 1],
            label="Rotational Energy",
        )
        ax1.plot(
            self.flight.translational_energy[:, 0],
            self.flight.translational_energy[:, 1],
            label="Translational Energy",
        )
        ax1.set_xlim(
            0,
            self.flight.apogee_time
            if self.flight.apogee_time != 0.0
            else self.flight.t_final,
        )
        ax1.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax1.set_title("Kinetic Energy Components")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Energy (J)")

        ax1.legend()
        ax1.grid()

        ax2 = plt.subplot(412)
        ax2.plot(
            self.flight.total_energy[:, 0],
            self.flight.total_energy[:, 1],
            label="Total Energy",
        )
        ax2.plot(
            self.flight.kinetic_energy[:, 0],
            self.flight.kinetic_energy[:, 1],
            label="Kinetic Energy",
        )
        ax2.plot(
            self.flight.potential_energy[:, 0],
            self.flight.potential_energy[:, 1],
            label="Potential Energy",
        )
        ax2.set_xlim(
            0,
            self.flight.apogee_time
            if self.flight.apogee_time != 0.0
            else self.flight.t_final,
        )
        ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax2.set_title("Total Mechanical Energy Components")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Energy (J)")
        ax2.legend()
        ax2.grid()

        ax3 = plt.subplot(413)
        ax3.plot(
            self.flight.thrust_power[:, 0],
            self.flight.thrust_power[:, 1],
            label="|Thrust Power|",
        )
        ax3.set_xlim(0, self.flight.rocket.motor.burn_out_time)
        ax3.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax3.set_title("Thrust Absolute Power")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Power (W)")
        ax3.legend()
        ax3.grid()

        ax4 = plt.subplot(414)
        ax4.plot(
            self.flight.drag_power[:, 0],
            -self.flight.drag_power[:, 1],
            label="|Drag Power|",
        )
        ax4.set_xlim(
            0,
            self.flight.apogee_time
            if self.flight.apogee_time != 0.0
            else self.flight.t_final,
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

        Returns
        -------
        None
        """

        # Trajectory Fluid Mechanics Plots
        fig10 = plt.figure(figsize=(9, 12))

        ax1 = plt.subplot(411)
        ax1.plot(self.flight.mach_number[:, 0], self.flight.mach_number[:, 1])
        ax1.set_xlim(0, self.flight.t_final)
        ax1.set_title("Mach Number")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Mach Number")
        ax1.grid()

        ax2 = plt.subplot(412)
        ax2.plot(self.flight.reynolds_number[:, 0], self.flight.reynolds_number[:, 1])
        ax2.set_xlim(0, self.flight.t_final)
        ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax2.set_title("Reynolds Number")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Reynolds Number")
        ax2.grid()

        ax3 = plt.subplot(413)
        ax3.plot(
            self.flight.dynamic_pressure[:, 0],
            self.flight.dynamic_pressure[:, 1],
            label="Dynamic Pressure",
        )
        ax3.plot(
            self.flight.total_pressure[:, 0],
            self.flight.total_pressure[:, 1],
            label="Total Pressure",
        )
        ax3.plot(
            self.flight.pressure[:, 0],
            self.flight.pressure[:, 1],
            label="Static Pressure",
        )
        ax3.set_xlim(0, self.flight.t_final)
        ax3.legend()
        ax3.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax3.set_title("Total and Dynamic Pressure")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Pressure (Pa)")
        ax3.grid()

        ax4 = plt.subplot(414)
        ax4.plot(self.flight.angle_of_attack[:, 0], self.flight.angle_of_attack[:, 1])
        ax4.set_title("Angle of Attack")
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Angle of Attack (°)")
        ax4.set_xlim(self.flight.out_of_rail_time, self.first_event_time)
        ax4.set_ylim(0, self.flight.angle_of_attack(self.flight.out_of_rail_time) + 15)
        ax4.grid()

        plt.subplots_adjust(hspace=0.5)
        plt.show()

        return None

    def stability_and_control_data(self):
        """Prints out Rocket Stability and Control parameters graphs available
        about the Flight

        Returns
        -------
        None
        """

        fig9 = plt.figure(figsize=(9, 6))

        ax1 = plt.subplot(211)
        ax1.plot(self.flight.stability_margin[:, 0], self.flight.stability_margin[:, 1])
        ax1.set_xlim(0, self.flight.stability_margin[:, 0][-1])
        ax1.set_title("Stability Margin")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Stability Margin (c)")
        ax1.set_xlim(0, self.first_event_time)
        ax1.axvline(
            x=self.flight.out_of_rail_time,
            color="r",
            linestyle="--",
            label="Out of Rail Time",
        )
        ax1.axvline(
            x=self.flight.rocket.motor.burn_out_time,
            color="g",
            linestyle="--",
            label="Burn Out Time",
        )

        ax1.axvline(
            x=self.flight.apogee_time,
            color="m",
            linestyle="--",
            label="Apogee Time",
        )
        ax1.legend()
        ax1.grid()

        ax2 = plt.subplot(212)
        max_attitude = max(self.flight.attitude_frequency_response[:, 1])
        max_attitude = max_attitude if max_attitude != 0 else 1
        ax2.plot(
            self.flight.attitude_frequency_response[:, 0],
            self.flight.attitude_frequency_response[:, 1] / max_attitude,
            label="Attitude Angle",
        )
        max_omega1 = max(self.flight.omega1_frequency_response[:, 1])
        max_omega1 = max_omega1 if max_omega1 != 0 else 1
        ax2.plot(
            self.flight.omega1_frequency_response[:, 0],
            self.flight.omega1_frequency_response[:, 1] / max_omega1,
            label=r"$\omega_1$",
        )
        max_omega2 = max(self.flight.omega2_frequency_response[:, 1])
        max_omega2 = max_omega2 if max_omega2 != 0 else 1
        ax2.plot(
            self.flight.omega2_frequency_response[:, 0],
            self.flight.omega2_frequency_response[:, 1] / max_omega2,
            label=r"$\omega_2$",
        )
        max_omega3 = max(self.flight.omega3_frequency_response[:, 1])
        max_omega3 = max_omega3 if max_omega3 != 0 else 1
        ax2.plot(
            self.flight.omega3_frequency_response[:, 0],
            self.flight.omega3_frequency_response[:, 1] / max_omega3,
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

        Returns
        -------
        None
        """

        # self.flight.pressure()

        plt.figure()
        ax1 = plt.subplot(111)
        ax1.plot(self.flight.pressure[:, 0], self.flight.pressure[:, 1])
        ax1.set_title("Pressure at Rocket's Altitude")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Pressure (Pa)")
        ax1.set_xlim(0, self.flight.t_final)
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

        Returns
        -------
        None
        """

        if len(self.flight.parachute_events) > 0:
            for parachute in self.flight.rocket.parachutes:
                print("\nParachute: ", parachute.name)
                self.flight._calculate_pressure_signal()
                parachute.noise_signal_function()
                parachute.noisy_pressure_signal_function()
                parachute.clean_pressure_signal_function()
        else:
            print("\nRocket has no parachutes. No parachute plots available")

        return None

    def all(self):
        """Prints out all plots available about the Flight.

        Returns
        -------
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

        print("\n\nAerodynamic Forces Plots\n")
        self.aerodynamic_forces()

        print("\n\nRail Buttons Forces Plots\n")
        self.rail_buttons_forces()

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
