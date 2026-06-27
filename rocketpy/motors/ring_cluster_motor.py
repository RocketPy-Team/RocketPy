# pylint: disable=invalid-name
import matplotlib.pyplot as plt
import numpy as np

from ..mathutils.function import Function
from .motor import Motor


class RingClusterMotor(Motor):
    """
    A class representing a cluster of N identical motors arranged symmetrically.

    This class models a ring (annular) cluster configuration where a specific
    number of identical motors (N >= 2) are arranged symmetrically along a
    circular perimeter of a given radius. Note that this model assumes no
    central motor is present along the rocket's longitudinal axis. The total
    inertia tensors (Ixx and Iyy) are computed by explicitly summing the
    contribution of each individual motor based on its angular position,
    ensuring mathematical accuracy for all configurations, including the
    asymmetric transverse inertia case of N=2.

    Attributes
    ----------
    motor : SolidMotor
        The single motor instance used in the cluster.
    number : int
        The number of motors in the cluster.
    radius : float
        The radial distance from the rocket's central axis to the center of each motor.
    """

    def __init__(self, motor, number, radius):
        """
        Initialize the RingClusterMotor.

        Parameters
        ----------
        motor : SolidMotor
            The base motor to be clustered.
        number : int
            Number of motors. Must be >= 2.
        radius : float
            Distance from center of rocket to center of motor (m).
        """
        if not isinstance(number, int):
            raise TypeError(f"number must be an int, got {type(number).__name__}")
        if number < 2:
            raise ValueError("number must be >= 2 for a RingClusterMotor")
        if not isinstance(radius, (int, float)):
            raise TypeError(
                f"radius must be a real number, got {type(radius).__name__}"
            )
        if radius < 0:
            raise ValueError("radius must be non-negative")

        self.motor = motor
        self.number = number
        self.radius = float(radius)
        dry_inertia_cluster = self._calculate_dry_inertia()

        # Use a thrust source scaled by the number of motors so that
        # all thrust-derived quantities computed by the base Motor class
        # correspond to the full cluster rather than a single motor.
        scaled_thrust_source = motor.thrust * number

        super().__init__(
            thrust_source=scaled_thrust_source,
            nozzle_radius=motor.nozzle_radius,
            burn_time=motor.burn_time,
            dry_mass=motor.dry_mass * number,
            dry_inertia=dry_inertia_cluster,
            center_of_dry_mass_position=motor.center_of_dry_mass_position,
            coordinate_system_orientation=motor.coordinate_system_orientation,
            interpolation_method="linear",
        )

        self._setup_grain_properties()
        self._propellant_mass = self.motor.propellant_mass * self.number
        self._propellant_initial_mass = self.number * self.motor.propellant_initial_mass
        self._center_of_propellant_mass = self.motor.center_of_propellant_mass
        self._evaluate_propellant_inertia()

    def _evaluate_propellant_inertia(self):
        """Calculates the dynamic inertia of the propellant using Steiner's theorem."""
        self._propellant_I_11 = self.motor.propellant_I_11 * self.number
        self._propellant_I_22 = self.motor.propellant_I_22 * self.number

        angles = np.linspace(0, 2 * np.pi, self.number, endpoint=False)
        for angle in angles:
            x = self.radius * np.cos(angle)
            y = self.radius * np.sin(angle)

            self._propellant_I_11 += self.motor.propellant_mass * (y**2)
            self._propellant_I_22 += self.motor.propellant_mass * (x**2)

        Izz_term1 = self.motor.propellant_I_33 * self.number
        Izz_term2 = self.motor.propellant_mass * (self.number * self.radius**2)
        self._propellant_I_33 = Izz_term1 + Izz_term2

        zero_func = Function(0)
        self._propellant_I_12 = zero_func
        self._propellant_I_13 = zero_func
        self._propellant_I_23 = zero_func

    def _setup_grain_properties(self):
        """Copies the grain properties from the base motor."""
        self.throat_radius = self.motor.throat_radius
        self.grain_number = self.motor.grain_number
        self.grain_density = self.motor.grain_density
        self.grain_outer_radius = self.motor.grain_outer_radius
        self.grain_initial_inner_radius = self.motor.grain_initial_inner_radius
        self.grain_initial_height = self.motor.grain_initial_height
        self.grains_center_of_mass_position = self.motor.grains_center_of_mass_position

    @property
    def thrust(self):
        return self._thrust

    @thrust.setter
    def thrust(self, value):
        self._thrust = value

    @property
    def propellant_mass(self):
        return self._propellant_mass

    @propellant_mass.setter
    def propellant_mass(self, value):
        self._propellant_mass = value

    @property
    def propellant_initial_mass(self):
        return self._propellant_initial_mass

    @propellant_initial_mass.setter
    def propellant_initial_mass(self, value):
        self._propellant_initial_mass = value

    @property
    def center_of_propellant_mass(self):
        return self._center_of_propellant_mass

    @center_of_propellant_mass.setter
    def center_of_propellant_mass(self, value):
        self._center_of_propellant_mass = value

    @property
    def propellant_I_11(self):
        return self._propellant_I_11

    @propellant_I_11.setter
    def propellant_I_11(self, value):
        self._propellant_I_11 = value

    @property
    def propellant_I_22(self):
        return self._propellant_I_22

    @propellant_I_22.setter
    def propellant_I_22(self, value):
        self._propellant_I_22 = value

    @property
    def propellant_I_33(self):
        return self._propellant_I_33

    @propellant_I_33.setter
    def propellant_I_33(self, value):
        self._propellant_I_33 = value

    @property
    def propellant_I_12(self):
        return self._propellant_I_12

    @propellant_I_12.setter
    def propellant_I_12(self, value):
        self._propellant_I_12 = value

    @property
    def propellant_I_13(self):
        return self._propellant_I_13

    @propellant_I_13.setter
    def propellant_I_13(self, value):
        self._propellant_I_13 = value

    @property
    def propellant_I_23(self):
        return self._propellant_I_23

    @propellant_I_23.setter
    def propellant_I_23(self, value):
        self._propellant_I_23 = value

    @property
    def exhaust_velocity(self):
        return self.motor.exhaust_velocity

    def _calculate_dry_inertia(self):
        Ixx_loc = self.motor.dry_I_11
        Iyy_loc = self.motor.dry_I_22
        Izz_loc = self.motor.dry_I_33
        m_dry = self.motor.dry_mass

        Izz_cluster = self.number * Izz_loc + self.number * m_dry * (self.radius**2)
        Ixx_cluster = self.number * Ixx_loc
        Iyy_cluster = self.number * Iyy_loc

        angles = np.linspace(0, 2 * np.pi, self.number, endpoint=False)
        for angle in angles:
            x = self.radius * np.cos(angle)
            y = self.radius * np.sin(angle)
            Ixx_cluster += m_dry * (y**2)
            Iyy_cluster += m_dry * (x**2)

        return (Ixx_cluster, Iyy_cluster, Izz_cluster)

    def info(self, *, filename=None):  # pylint: disable=unused-argument
        """Prints a summary of the cluster configuration and its aggregated
        (cluster-level) thrust and mass properties."""
        print("Cluster Configuration:")
        print(f" - Motors: {self.number} x {type(self.motor).__name__}")
        print(f" - Radial Distance: {self.radius} m")
        print(f" - Total Dry Mass: {self.dry_mass:.3f} kg")
        print(
            f" - Total Initial Propellant Mass: {self.propellant_initial_mass:.3f} kg"
        )
        print(f" - Total Impulse: {self.total_impulse:.3f} Ns")
        print(f" - Average Thrust: {self.average_thrust:.3f} N")
        print(f" - Burn Duration: {self.burn_duration:.3f} s")

    def draw_cluster_layout(self, rocket_radius=None, show=True):
        """Draw the geometric layout of the clustered motors."""
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(0, 0, "k+", markersize=10, label="Central axis")
        if rocket_radius:
            rocket_tube = plt.Circle(
                (0, 0),
                rocket_radius,
                color="black",
                fill=False,
                linestyle="--",
                linewidth=2,
                label="Rocket",
            )
            ax.add_patch(rocket_tube)
            limit = rocket_radius * 1.2
        else:
            limit = self.radius * 2
        self._draw_engines(ax)
        ax.set_aspect("equal", "box")
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_xlabel("Position X (m)")
        ax.set_ylabel("Position Y (m)")
        ax.set_title(f"Cluster Configuration : {self.number} engines")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right")
        if show:
            plt.show()
        return fig, ax

    def _draw_engines(self, ax):
        """Draws the individual engines of the cluster."""
        motor_outer_radius = self.grain_outer_radius
        angles = np.linspace(0, 2 * np.pi, self.number, endpoint=False)

        for i, angle in enumerate(angles):
            x = self.radius * np.cos(angle)
            y = self.radius * np.sin(angle)
            motor_circle = plt.Circle(
                (x, y),
                motor_outer_radius,
                color="red",
                alpha=0.5,
                label="Engine" if i == 0 else "",
            )
            ax.add_patch(motor_circle)
            ax.text(
                x,
                y,
                str(i + 1),
                color="white",
                ha="center",
                va="center",
                fontweight="bold",
            )
