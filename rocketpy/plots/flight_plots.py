from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np

from .plot_helpers import show_or_save_plot


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

    def trajectory_3d(self, *, filename=None):  # pylint: disable=too-many-statements
        """Plot a 3D graph of the trajectory

        Parameters
        ----------
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved. Supported file endings are:
            eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            and webp (these are the formats supported by matplotlib).

        Returns
        -------
        None
        """
        max_z = max(self.flight.altitude[:, 1])
        min_z = min(self.flight.altitude[:, 1])
        max_x = max(self.flight.x[:, 1])
        min_x = min(self.flight.x[:, 1])
        max_y = max(self.flight.y[:, 1])
        min_y = min(self.flight.y[:, 1])
        min_xy = min(min_x, min_y)
        max_xy = max(max_x, max_y)

        # avoids errors when x_lim and y_lim are the same
        if abs(min_z - max_z) < 1e-5:
            max_z += 1
        if abs(min_xy - max_xy) < 1e-5:
            max_xy += 1

        _ = plt.figure(figsize=(9, 9))
        ax1 = plt.subplot(111, projection="3d")
        ax1.plot(
            self.flight.x[:, 1], self.flight.y[:, 1], zs=min_z, zdir="z", linestyle="--"
        )
        ax1.plot(
            self.flight.x[:, 1],
            self.flight.altitude[:, 1],
            zs=min_y,
            zdir="y",
            linestyle="--",
        )
        ax1.plot(
            self.flight.y[:, 1],
            self.flight.altitude[:, 1],
            zs=min_x,
            zdir="x",
            linestyle="--",
        )
        ax1.plot(
            self.flight.x[:, 1],
            self.flight.y[:, 1],
            self.flight.altitude[:, 1],
            linewidth="2",
        )
        ax1.scatter(
            self.flight.x(0),
            self.flight.y(0),
            self.flight.z(0) - self.flight.env.elevation,
            color="black",
        )
        ax1.scatter(
            self.flight.x(self.flight.t_final),
            self.flight.y(self.flight.t_final),
            self.flight.z(self.flight.t_final) - self.flight.env.elevation,
            color="red",
            marker="X",
        )
        ax1.set_xlabel("X - East (m)")
        ax1.set_ylabel("Y - North (m)")
        ax1.set_zlabel("Z - Altitude Above Ground Level (m)")
        ax1.set_title("Flight Trajectory")
        ax1.set_xlim(min_xy, max_xy)
        ax1.set_ylim(min_xy, max_xy)
        ax1.set_zlim(min_z, max_z)
        ax1.view_init(15, 45)
        ax1.set_box_aspect(None, zoom=0.95)  # 95% for label adjustment
        show_or_save_plot(filename)

    def linear_kinematics_data(self, *, filename=None):  # pylint: disable=too-many-statements
        """Prints out all Kinematics graphs available about the Flight

        Parameters
        ----------
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved. Supported file endings are:
            eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            and webp (these are the formats supported by matplotlib).

        Returns
        -------
        None
        """
        plt.figure(figsize=(9, 12))

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
        show_or_save_plot(filename)

    def attitude_data(self, *, filename=None):  # pylint: disable=too-many-statements
        """Prints out all Angular position graphs available about the Flight

        Parameters
        ----------
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved. Supported file endings are:
            eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            and webp (these are the formats supported by matplotlib).

        Returns
        -------
        None
        """

        # Angular position plots
        _ = plt.figure(figsize=(9, 12))

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
        show_or_save_plot(filename)

    def flight_path_angle_data(self, *, filename=None):
        """Prints out Flight path and Rocket Attitude angle graphs available
        about the Flight

        Parameters
        ----------
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved. Supported file endings are:
            eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            and webp (these are the formats supported by matplotlib).

        Returns
        -------
        None
        """
        plt.figure(figsize=(9, 6))

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
        show_or_save_plot(filename)

    def angular_kinematics_data(self, *, filename=None):  # pylint: disable=too-many-statements
        """Prints out all Angular velocity and acceleration graphs available
        about the Flight

        Parameters
        ----------
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved. Supported file endings are:
            eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            and webp (these are the formats supported by matplotlib).

        Returns
        -------
        None
        """
        plt.figure(figsize=(9, 9))
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
        show_or_save_plot(filename)

    def rail_buttons_bending_moments(self, *, filename=None):
        """Prints out Rail Buttons Bending Moments graphs.

        Parameters
        ----------
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved. Supported file endings are:
            eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            and webp (these are the formats supported by matplotlib).

        Returns
        -------
        None
        """
        if len(self.flight.rocket.rail_buttons) == 0:
            print(
                "No rail buttons were defined. Skipping rail button bending moment plots."
            )
        elif self.flight.out_of_rail_time_index == 0:
            print("No rail phase was found. Skipping rail button bending moment plots.")
        else:
            # Check if button_height is defined
            rail_buttons_tuple = self.flight.rocket.rail_buttons[0]
            if rail_buttons_tuple.component.button_height is None:
                print("Rail button height not defined. Skipping bending moment plots.")
            else:
                plt.figure(figsize=(9, 3))

                ax1 = plt.subplot(111)
                ax1.plot(
                    self.flight.rail_button1_bending_moment[
                        : self.flight.out_of_rail_time_index, 0
                    ],
                    self.flight.rail_button1_bending_moment[
                        : self.flight.out_of_rail_time_index, 1
                    ],
                    label="Upper Rail Button",
                )
                ax1.plot(
                    self.flight.rail_button2_bending_moment[
                        : self.flight.out_of_rail_time_index, 0
                    ],
                    self.flight.rail_button2_bending_moment[
                        : self.flight.out_of_rail_time_index, 1
                    ],
                    label="Lower Rail Button",
                )
                ax1.set_xlim(
                    0,
                    (
                        self.flight.out_of_rail_time
                        if self.flight.out_of_rail_time > 0
                        else self.flight.tFinal
                    ),
                )
                ax1.legend()
                ax1.grid(True)
                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("Bending Moment (N·m)")
                ax1.set_title("Rail Button Bending Moments")

                show_or_save_plot(filename)

    def rail_buttons_forces(self, *, filename=None):  # pylint: disable=too-many-statements
        """Prints out all Rail Buttons Forces graphs available about the Flight.

        Parameters
        ----------
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved. Supported file endings are:
            eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            and webp (these are the formats supported by matplotlib).

        Returns
        -------
        None
        """
        if len(self.flight.rocket.rail_buttons) == 0:
            print("No rail buttons were defined. Skipping rail button plots.")
        elif self.flight.out_of_rail_time_index == 0:
            print("No rail phase was found. Skipping rail button plots.")
        else:
            plt.figure(figsize=(9, 6))

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
                (
                    self.flight.out_of_rail_time
                    if self.flight.out_of_rail_time > 0
                    else self.flight.tFinal
                ),
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
                (
                    self.flight.out_of_rail_time
                    if self.flight.out_of_rail_time > 0
                    else self.flight.tFinal
                ),
            )
            ax2.legend()
            ax2.grid(True)
            ax2.set_xlabel(self.flight.rail_button1_shear_force.__inputs__[0])
            ax2.set_ylabel(self.flight.rail_button1_shear_force.__outputs__[0])
            ax2.set_title("Rail Buttons Shear Force")

            plt.subplots_adjust(hspace=0.5)
            show_or_save_plot(filename)

    def aerodynamic_forces(self, *, filename=None):  # pylint: disable=too-many-statements
        """Prints out all Forces and Moments graphs available about the Flight

        Parameters
        ----------
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved. Supported file endings are:
            eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            and webp (these are the formats supported by matplotlib).

        Returns
        -------
        None
        """
        plt.figure(figsize=(9, 12))

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
        show_or_save_plot(filename)

    def energy_data(self, *, filename=None):  # pylint: disable=too-many-statements
        """Prints out all Energy components graphs available about the Flight

        Parameters
        ----------
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved. Supported file endings are:
            eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            and webp (these are the formats supported by matplotlib).

        Returns
        -------
        None
        """

        plt.figure(figsize=(9, 9))

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
            (
                self.flight.apogee_time
                if self.flight.apogee_time != 0.0
                else self.flight.t_final
            ),
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
            (
                self.flight.apogee_time
                if self.flight.apogee_time != 0.0
                else self.flight.t_final
            ),
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
            (
                self.flight.apogee_time
                if self.flight.apogee_time != 0.0
                else self.flight.t_final
            ),
        )
        ax3.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax4.set_title("Drag Absolute Power")
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Power (W)")
        ax4.legend()
        ax4.grid()

        plt.subplots_adjust(hspace=1)
        show_or_save_plot(filename)

    def fluid_mechanics_data(self, *, filename=None):  # pylint: disable=too-many-statements
        """Prints out a summary of the Fluid Mechanics graphs available about
        the Flight

        Parameters
        ----------
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved. Supported file endings are:
            eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            and webp (these are the formats supported by matplotlib).

        Returns
        -------
        None
        """
        plt.figure(figsize=(9, 16))

        ax1 = plt.subplot(611)
        ax1.plot(self.flight.mach_number[:, 0], self.flight.mach_number[:, 1])
        ax1.set_xlim(0, self.flight.t_final)
        ax1.set_title("Mach Number")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Mach Number")
        ax1.grid()

        ax2 = plt.subplot(612)
        ax2.plot(self.flight.reynolds_number[:, 0], self.flight.reynolds_number[:, 1])
        ax2.set_xlim(0, self.flight.t_final)
        ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax2.set_title("Reynolds Number")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Reynolds Number")
        ax2.grid()

        ax3 = plt.subplot(613)
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

        ax4 = plt.subplot(614)
        ax4.plot(self.flight.angle_of_attack[:, 0], self.flight.angle_of_attack[:, 1])
        ax4.set_title("Angle of Attack")
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Angle of Attack (°)")
        ax4.set_xlim(self.flight.out_of_rail_time, self.first_event_time)
        ax4.set_ylim(0, self.flight.angle_of_attack(self.flight.out_of_rail_time) + 15)
        ax4.grid()

        ax5 = plt.subplot(615)
        ax5.plot(
            self.flight.partial_angle_of_attack[:, 0],
            self.flight.partial_angle_of_attack[:, 1],
        )
        ax5.set_title("Partial Angle of Attack")
        ax5.set_xlabel("Time (s)")
        ax5.set_ylabel("Partial Angle of Attack (°)")
        ax5.set_xlim(self.flight.out_of_rail_time, self.first_event_time)
        ax5.set_ylim(
            0, self.flight.partial_angle_of_attack(self.flight.out_of_rail_time) + 15
        )
        ax5.grid()

        ax6 = plt.subplot(616)
        ax6.plot(
            self.flight.angle_of_sideslip[:, 0], self.flight.angle_of_sideslip[:, 1]
        )
        ax6.set_title("Angle of Sideslip")
        ax6.set_xlabel("Time (s)")
        ax6.set_ylabel("Angle of Sideslip (°)")
        ax6.set_xlim(self.flight.out_of_rail_time, self.first_event_time)
        ax6.set_ylim(
            0, self.flight.angle_of_sideslip(self.flight.out_of_rail_time) + 15
        )
        ax6.grid()

        plt.subplots_adjust(hspace=0.5)
        show_or_save_plot(filename)

    def stability_and_control_data(self, *, filename=None):  # pylint: disable=too-many-statements
        """Prints out Rocket Stability and Control parameters graphs available
        about the Flight

        Parameters
        ----------
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved. Supported file endings are:
            eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            and webp (these are the formats supported by matplotlib).

        Returns
        -------
        None
        """

        plt.figure(figsize=(9, 6))

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
        x_axis = np.arange(0, 5, 0.01)
        max_attitude = self.flight.attitude_frequency_response.max
        max_attitude = max_attitude if max_attitude != 0 else 1
        ax2.plot(
            x_axis,
            self.flight.attitude_frequency_response(x_axis) / max_attitude,
            label="Attitude Angle",
        )
        max_omega1 = self.flight.omega1_frequency_response.max
        max_omega1 = max_omega1 if max_omega1 != 0 else 1
        ax2.plot(
            x_axis,
            self.flight.omega1_frequency_response(x_axis) / max_omega1,
            label=r"$\omega_1$",
        )
        max_omega2 = self.flight.omega2_frequency_response.max
        max_omega2 = max_omega2 if max_omega2 != 0 else 1
        ax2.plot(
            x_axis,
            self.flight.omega2_frequency_response(x_axis) / max_omega2,
            label=r"$\omega_2$",
        )
        max_omega3 = self.flight.omega3_frequency_response.max
        max_omega3 = max_omega3 if max_omega3 != 0 else 1
        ax2.plot(
            x_axis,
            self.flight.omega3_frequency_response(x_axis) / max_omega3,
            label=r"$\omega_3$",
        )
        ax2.set_title("Frequency Response")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Amplitude Magnitude Normalized")
        ax2.set_xlim(0, 5)
        ax2.legend()
        ax2.grid()

        plt.subplots_adjust(hspace=0.5)
        show_or_save_plot(filename)

    def pressure_rocket_altitude(self, *, filename=None):
        """Plots out pressure at rocket's altitude.

        Parameters
        ----------
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved. Supported file endings are:
            eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            and webp (these are the formats supported by matplotlib).

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

        show_or_save_plot(filename)

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
                parachute.noise_signal_function()
                parachute.noisy_pressure_signal_function()
                parachute.clean_pressure_signal_function()
        else:
            print("\nRocket has no parachutes. No parachute plots available")

    def all(self):  # pylint: disable=too-many-statements
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

        print("\n\nRail Buttons Bending Moments Plots\n")
        self.rail_buttons_bending_moments()

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
