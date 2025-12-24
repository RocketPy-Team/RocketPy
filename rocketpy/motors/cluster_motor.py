import matplotlib as patches
import matplotlib.pyplot as plt
import numpy as np

from rocketpy.mathutils.function import Function
from rocketpy.mathutils.vector_matrix import Vector
from rocketpy.tools import (
    parallel_axis_theorem_I11,
    parallel_axis_theorem_I12,
    parallel_axis_theorem_I13,
    parallel_axis_theorem_I22,
    parallel_axis_theorem_I23,
    parallel_axis_theorem_I33,
)


class ClusterMotor:
    """
    Manages a cluster of motors.

    This class behaves like a single motor by aggregating the properties
    of several individual motors and implementing the full interface
    expected by the Rocket class. It handles the calculation of
    aggregated thrust, mass, center of mass, and inertia tensor as
    functions of time, considering the 3D position and orientation
    of each motor.
    """

    def __init__(self, motors, positions, orientations=None):
        """Initializes the ClusterMotor, aggregating multiple motors.

        Parameters
        ----------
        motors : list[Motor]
            A list of instantiated RocketPy Motor objects to be part of the cluster.
        positions : list[tuple] or list[list] or list[np.array]
            A list of 3D position vectors [x, y, z] for each motor.
            The position is relative to the rocket's coordinate system origin,
            which is also the cluster's origin. The coordinate system
            orientation is assumed to be 'tail_to_nose'.
        orientations : list[tuple] or list[list] or list[np.array], optional
            A list of 3D unit vectors [x, y, z] specifying the thrust
            direction for each motor in the rocket's coordinate system.
            If None, all motors are assumed to thrust along the rocket's
            positive Z-axis (e.g., [0, 0, 1]). Defaults to None.
        """
        self._validate_inputs(motors, positions, orientations)

        self.coordinate_system_orientation = "tail_to_nose"
        self._csys = 1

        self._initialize_basic_properties()
        self._initialize_thrust_and_mass()
        self._initialize_center_of_mass()
        self._initialize_inertia_properties()

    def _validate_inputs(self, motors, positions, orientations):
        """Validates and stores the primary inputs for the cluster."""
        if not motors:
            raise ValueError("The list of motors cannot be empty.")

        self.motors = motors
        self.positions = [np.array(pos) for pos in positions]

        if orientations is None:
            self.orientations = [np.array([0, 0, 1.0]) for _ in motors]
        else:
            self.orientations = [
                np.array(ori) / np.linalg.norm(ori) for ori in orientations
            ]

        if not len(self.motors) == len(self.positions) == len(self.orientations):
            raise ValueError(
                "The 'motors', 'positions', and 'orientations' lists must have the same length."
            )

    def _initialize_basic_properties(self):
        """Calculates simple aggregated scalar properties."""
        self.propellant_initial_mass = sum(
            m.propellant_initial_mass for m in self.motors
        )
        self.dry_mass = sum(m.dry_mass for m in self.motors)
        self.total_impulse = sum(m.total_impulse for m in self.motors)

        self.burn_start_time = min(motor.burn_start_time for motor in self.motors)
        self.burn_out_time = max(motor.burn_out_time for motor in self.motors)
        self.burn_time = (self.burn_start_time, self.burn_out_time)

        self.nozzle_radius = np.sqrt(sum(m.nozzle_radius**2 for m in self.motors))
        self.throat_radius = np.sqrt(sum(m.throat_radius**2 for m in self.motors))

        self.nozzle_position = np.mean(
            [
                pos[2] + motor.nozzle_position * motor._csys
                for motor, pos in zip(self.motors, self.positions)
            ]
        )

    def _initialize_thrust_and_mass(self):
        """Initializes thrust and mass-related Function objects."""
        self.thrust = Function(
            self._calculate_total_thrust_scalar, inputs="t", outputs="Thrust (N)"
        )
        if self.burn_time[1] > self.burn_time[0]:
            self.thrust.set_discrete(self.burn_start_time, self.burn_out_time, 100)

        if self.propellant_initial_mass > 1e-9:
            average_exhaust_velocity = self.total_impulse / self.propellant_initial_mass
            self.total_mass_flow_rate = self.thrust / -average_exhaust_velocity
        else:
            self.total_mass_flow_rate = Function(0)

        self.mass_flow_rate = self.total_mass_flow_rate

        self.propellant_mass = Function(
            self._calculate_propellant_mass, inputs="t", outputs="Propellant Mass (kg)"
        )
        self.total_mass = Function(
            self._calculate_total_mass, inputs="t", outputs="Total Mass (kg)"
        )

    def _initialize_center_of_mass(self):
        """Initializes center of mass Function objects."""
        self.center_of_mass = Function(
            self._calculate_center_of_mass, inputs="t", outputs="Cluster CoM Vector (m)"
        )
        self.center_of_propellant_mass = Function(
            self._calculate_center_of_propellant_mass,
            inputs="t",
            outputs="Cluster CoPM Vector (m)",
        )

        com_time = (
            self.burn_out_time
            if self.burn_out_time > self.burn_start_time
            else self.burn_start_time
        )
        self.center_of_dry_mass_position = self._calculate_center_of_mass(com_time)

    def _initialize_inertia_properties(self):
        """Initializes dry and propellant inertia properties."""
        (
            self.dry_I_11,
            self.dry_I_22,
            self.dry_I_33,
            self.dry_I_12,
            self.dry_I_13,
            self.dry_I_23,
        ) = self._calculate_total_dry_inertia()

        def propellant_inertia_func(t):
            return self._calculate_total_propellant_inertia(t)

        self.propellant_I_11 = Function(
            lambda t: propellant_inertia_func(t)[0], inputs="t", outputs="I_11 (kg*m^2)"
        )
        self.propellant_I_22 = Function(
            lambda t: propellant_inertia_func(t)[1], inputs="t", outputs="I_22 (kg*m^2)"
        )
        self.propellant_I_33 = Function(
            lambda t: propellant_inertia_func(t)[2], inputs="t", outputs="I_33 (kg*m^2)"
        )
        self.propellant_I_12 = Function(
            lambda t: propellant_inertia_func(t)[3], inputs="t", outputs="I_12 (kg*m^2)"
        )
        self.propellant_I_13 = Function(
            lambda t: propellant_inertia_func(t)[4], inputs="t", outputs="I_13 (kg*m^2)"
        )
        self.propellant_I_23 = Function(
            lambda t: propellant_inertia_func(t)[5], inputs="t", outputs="I_23 (kg*m^2)"
        )

    def _calculate_total_mass(self, t):
        """Calculates total cluster mass at time t."""
        return self.dry_mass + self._calculate_propellant_mass(t)

    def _calculate_propellant_mass(self, t):
        """
        Calculates the total propellant mass at time t by integrating the
        total mass flow rate. This ensures consistency between mass and thrust.
        """
        return self.propellant_initial_mass + self.total_mass_flow_rate.integral(
            self.burn_start_time, t
        )

    def get_total_thrust_vector(self, t):
        """
        Calculates the total thrust vector of the cluster at a given time t.
        This vector is the sum of all individual motor thrust vectors,
        considering their orientation.
        """
        total_thrust = np.zeros(3)
        for motor, orientation in zip(self.motors, self.orientations):
            scalar_thrust = motor.thrust.get_value_opt(t)
            total_thrust += scalar_thrust * orientation
        return Vector(total_thrust)

    def _calculate_total_thrust_scalar(self, t):
        """
        Calculates the magnitude of the total thrust vector.
        This is what is wrapped by the `self.thrust` Function.
        """
        return abs(self.get_total_thrust_vector(t))

    def get_total_moment(self, t, ref_point):
        """
        Calculates the total moment (torque) generated by the cluster
        about a given reference point (e.g., the rocket's CoM).
        """
        total_moment = np.zeros(3, dtype=np.float64)
        ref_point_arr = np.array(ref_point, dtype=np.float64)

        for motor, pos, orientation in zip(
            self.motors, self.positions, self.orientations
        ):
            force_magnitude = motor.thrust.get_value_opt(t)
            force = force_magnitude * orientation
            arm = pos - ref_point_arr
            total_moment += np.cross(arm, force)

            if hasattr(motor, "thrust_eccentricity_y") and hasattr(
                motor, "thrust_eccentricity_x"
            ):
                total_moment[0] += motor.thrust_eccentricity_y * force_magnitude
                total_moment[1] -= motor.thrust_eccentricity_x * force_magnitude

        return Vector(total_moment)

    def pressure_thrust(self, pressure):
        """Calculates the total pressure thrust correction for the cluster."""
        return sum(motor.pressure_thrust(pressure) for motor in self.motors)

    def _calculate_center_of_mass(self, t):
        """Calculates the aggregated center of mass of the cluster at time t."""
        total_mass_val = self._calculate_total_mass(t)
        if total_mass_val < 1e-9:
            return Vector(
                np.mean(self.positions, axis=0) if self.positions else np.zeros(3)
            )

        weighted_sum = np.zeros(3, dtype=np.float64)
        for motor, pos in zip(self.motors, self.positions):
            motor_com_local_z = motor.center_of_mass.get_value_opt(t)
            motor_com_global = pos + np.array([0, 0, motor_com_local_z * motor._csys])
            weighted_sum += motor.total_mass.get_value_opt(t) * motor_com_global

        return Vector(weighted_sum / total_mass_val)

    def _calculate_center_of_propellant_mass(self, t):
        """
        Calculates the aggregated center of mass of the cluster's propellant.
        This calculation is based on the individual motor properties.
        """
        total_prop_mass = 0.0
        weighted_sum = np.zeros(3, dtype=np.float64)

        for motor in self.motors:
            total_prop_mass += motor.propellant_mass.get_value_opt(t)

        if total_prop_mass < 1e-9:
            return self.center_of_dry_mass_position

        for motor, pos in zip(self.motors, self.positions):
            prop_mass_t = motor.propellant_mass.get_value_opt(t)
            if prop_mass_t > 1e-9:
                prop_com_local_z = motor.center_of_propellant_mass.get_value_opt(t)
                prop_com_global = pos + np.array([0, 0, prop_com_local_z * motor._csys])
                weighted_sum += prop_mass_t * prop_com_global

        return Vector(weighted_sum / total_prop_mass)

    def _calculate_total_dry_inertia(self):
        """
        Calculates the cluster's total dry inertia tensor relative to the
        cluster's center of dry mass.
        """
        I_11, I_22, I_33 = 0.0, 0.0, 0.0
        I_12, I_13, I_23 = 0.0, 0.0, 0.0

        ref_point = self.center_of_dry_mass_position

        for motor, pos in zip(self.motors, self.positions):
            m = motor.dry_mass
            motor_cdm_local_z = motor.center_of_dry_mass_position
            motor_cdm_global = pos + np.array([0, 0, motor_cdm_local_z * motor._csys])
            r_vec = Vector(motor_cdm_global) - ref_point

            I_11 += parallel_axis_theorem_I11(motor.dry_I_11, m, r_vec)
            I_22 += parallel_axis_theorem_I22(motor.dry_I_22, m, r_vec)
            I_33 += parallel_axis_theorem_I33(motor.dry_I_33, m, r_vec)
            I_12 += parallel_axis_theorem_I12(motor.dry_I_12, m, r_vec)
            I_13 += parallel_axis_theorem_I13(motor.dry_I_13, m, r_vec)
            I_23 += parallel_axis_theorem_I23(motor.dry_I_23, m, r_vec)

        return I_11, I_22, I_33, I_12, I_13, I_23

    def _calculate_total_propellant_inertia(self, t):
        """
        Calculates the cluster's total propellant inertia tensor relative to the
        cluster's center of propellant mass at time t.
        """
        I_11, I_22, I_33 = 0.0, 0.0, 0.0
        I_12, I_13, I_23 = 0.0, 0.0, 0.0

        ref_point = self._calculate_center_of_propellant_mass(t)

        for motor, pos in zip(self.motors, self.positions):
            m = motor.propellant_mass.get_value_opt(t)
            if m < 1e-9:
                continue

            prop_com_local_z = motor.center_of_propellant_mass.get_value_opt(t)
            prop_com_global = pos + np.array([0, 0, prop_com_local_z * motor._csys])
            r_vec = Vector(prop_com_global) - ref_point

            I_11 += parallel_axis_theorem_I11(
                motor.propellant_I_11.get_value_opt(t), m, r_vec
            )
            I_22 += parallel_axis_theorem_I22(
                motor.propellant_I_22.get_value_opt(t), m, r_vec
            )
            I_33 += parallel_axis_theorem_I33(
                motor.propellant_I_33.get_value_opt(t), m, r_vec
            )
            I_12 += parallel_axis_theorem_I12(
                motor.propellant_I_12.get_value_opt(t), m, r_vec
            )
            I_13 += parallel_axis_theorem_I13(
                motor.propellant_I_13.get_value_opt(t), m, r_vec
            )
            I_23 += parallel_axis_theorem_I23(
                motor.propellant_I_23.get_value_opt(t), m, r_vec
            )

        return I_11, I_22, I_33, I_12, I_13, I_23

    # pylint: disable=too-many-statements
    def draw_rear_view(self, rocket_radius, tail_radius=None, filename=None):
        """
        Plots a 2D rear view of the motor cluster, showing the main
        rocket body and optionally the tail cone diameter.

        Args:
            rocket_radius (float): The main radius of the rocket body.
            tail_radius (float, optional): The rocket's radius at the tail (boattail).
                                           If provided, a second circle will be drawn.
            filename (str, optional): If provided, saves the plot to a file.
                                      Otherwise, shows the plot.
        """
        _, ax = plt.subplots(figsize=(8.5, 8.5))

        rocket_body = patches.Circle(
            (0, 0),
            radius=rocket_radius,
            facecolor="lightgrey",
            edgecolor="black",
            linewidth=2,
            label=f"Main Body ({rocket_radius * 200:.0f} cm)",
        )
        ax.add_patch(rocket_body)

        if tail_radius is not None:
            tail_body = patches.Circle(
                (0, 0),
                radius=tail_radius,
                facecolor="silver",
                edgecolor="black",
                linestyle="--",
                linewidth=1.5,
                label=f"Shrunk ({tail_radius * 200:.0f} cm)",
            )
            ax.add_patch(tail_body)

        for i, (motor, pos) in enumerate(zip(self.motors, self.positions)):
            motor_body = patches.Circle(
                (pos[0], pos[1]),
                radius=getattr(motor, "grain_outer_radius", motor.nozzle_radius / 2),
                facecolor="dimgrey",
                edgecolor="black",
                linewidth=1.5,
            )
            ax.add_patch(motor_body)
            ax.text(
                pos[0],
                pos[1],
                f"M{i + 1}",
                ha="center",
                va="center",
                color="white",
                fontsize=12,
                fontweight="bold",
            )

        ax.set_aspect("equal", adjustable="box")
        plot_limit = rocket_radius * 1.2
        ax.set_xlim(-plot_limit, plot_limit)
        ax.set_ylim(-plot_limit, plot_limit)
        ax.set_title("Detailed Rear View of the Cluster", fontsize=16)
        ax.set_xlabel("Axis X (m)")
        ax.set_ylabel("Axis Y (m)")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.5)

        plt.legend()

        if filename:
            plt.savefig(filename)
            print(f"Rear view plot saved to: {filename}")
        else:
            plt.show()
