import inspect
import math
import warnings
from typing import Iterable

import numpy as np

from rocketpy.control.controller import _Controller
from rocketpy.mathutils.function import Function
from rocketpy.mathutils.vector_matrix import Matrix, Vector
from rocketpy.motors.empty_motor import EmptyMotor
from rocketpy.plots.rocket_plots import _RocketPlots
from rocketpy.prints.rocket_prints import _RocketPrints
from rocketpy.rocket.aero_surface import (
    AirBrakes,
    BodyTube,
    EllipticalFins,
    Fin,
    Fins,
    NoseCone,
    RailButtons,
    Tail,
    TrapezoidalFins,
)
from rocketpy.rocket.aero_surface.aero_coefficient import AeroCoefficient
from rocketpy.rocket.aero_surface.fins.elliptical_fin import EllipticalFin
from rocketpy.rocket.aero_surface.fins.free_form_fin import FreeFormFin
from rocketpy.rocket.aero_surface.fins.free_form_fins import FreeFormFins
from rocketpy.rocket.aero_surface.fins.trapezoidal_fin import TrapezoidalFin
from rocketpy.rocket.aero_surface.generic_surface import GenericSurface
from rocketpy.rocket.aero_surface.linear_generic_surface import LinearGenericSurface
from rocketpy.rocket.components import Components
from rocketpy.rocket.helpers import (
    full_body_coefficients,
    neutral_point_and_slope,
    zero_drag,
)
from rocketpy.rocket.parachute import Parachute
from rocketpy.tools import (
    deprecated,
    find_obj_from_hash,
    parallel_axis_theorem_from_com,
)


# pylint: disable=too-many-instance-attributes, too-many-public-methods, too-many-instance-attributes
class Rocket:
    """Keeps rocket information.

    Attributes
    ----------
    Rocket.radius : float
        Rocket's largest radius in meters.
    Rocket.area : float
        Rocket's circular cross section largest frontal area in squared
        meters.
    Rocket.center_of_dry_mass_position : float
        Position, in m, of the rocket's center of dry mass (i.e. center of
        mass without propellant) relative to the rocket's coordinate system.
        See :doc:`Positions and Coordinate Systems </user/positions>`
        for more information
        regarding the rocket's coordinate system.
    Rocket.center_of_mass_without_motor : int, float
        Position, in m, of the rocket's center of mass without motor
        relative to the rocket's coordinate system. This does not include
        the motor or propellant mass.
    Rocket.motor_center_of_mass_position : Function
        Position, in meters, of the motor's center of mass relative to the user
        defined rocket coordinate system. This is a function of time since the
        propellant mass decreases with time. For more information, see the
        :doc:`Positions and Coordinate Systems </user/positions>`.
    Rocket.motor_center_of_dry_mass_position : float
        Position, in meters, of the motor's center of dry mass (i.e. center of
        mass without propellant) relative to the user defined rocket coordinate
        system. This is constant since the motor dry mass is constant.
    Rocket.coordinate_system_orientation : string
        String defining the orientation of the rocket's coordinate system.
        The coordinate system is defined by the rocket's axis of symmetry.
        The system's origin may be placed anywhere along such axis, such as
        in the nozzle or in the nose cone, and must be kept the same for all
        other positions specified. If "tail_to_nose", the coordinate system
        is defined with the rocket's axis of symmetry pointing from the
        rocket's tail to the rocket's nose cone. If "nose_to_tail", the
        coordinate system is defined with the rocket's axis of symmetry
        pointing from the rocket's nose cone to the rocket's tail.
    Rocket.mass : float
        Rocket's mass without motor and propellant, measured in kg.
    Rocket.dry_mass : float
        Rocket's mass without propellant, measured in kg. It does include the
        motor mass.
    Rocket.center_of_mass : Function
        Position of the rocket's center of mass, including propellant, relative
        to the user defined rocket reference system.
        See :doc:`Positions and Coordinate Systems </user/positions>`
        for more information
        regarding the coordinate system.
        Expressed in meters as a function of time.
    Rocket.com_to_cdm_function : Function
        Function of time expressing the z-coordinate of the center of mass
        relative to the center of dry mass.
    Rocket.reduced_mass : Function
        Function of time expressing the reduced mass of the rocket,
        defined as the product of the propellant mass and the mass
        of the rocket without propellant, divided by the sum of the
        propellant mass and the rocket mass.
    Rocket.total_mass : Function
        Function of time expressing the total mass of the rocket,
        defined as the sum of the propellant mass and the rocket
        mass without propellant.
    Rocket.structural_mass_ratio: float
        Initial ratio between the dry mass and the total mass.
    Rocket.total_mass_flow_rate : Function
        Time derivative of rocket's total mass in kg/s as a function
        of time as obtained by the thrust source of the added motor.
    Rocket.thrust_to_weight : Function
        Function of time expressing the motor thrust force divided by rocket
        weight. The gravitational acceleration is assumed as 9.80665 m/s^2.
    Rocket.cp_eccentricity_x : float
        Center of pressure position relative to center of mass in the x
        axis, perpendicular to axis of cylindrical symmetry, in meters.
    Rocket.cp_eccentricity_y : float
        Center of pressure position relative to center of mass in the y
        axis, perpendicular to axis of cylindrical symmetry, in meters.
    Rocket.thrust_eccentricity_y : float
        Thrust vector position relative to center of mass in the y
        axis, perpendicular to axis of cylindrical symmetry, in meters.
    Rocket.thrust_eccentricity_x : float
        Thrust vector position relative to center of mass in the x
        axis, perpendicular to axis of cylindrical symmetry, in meters.
    Rocket.aerodynamic_surfaces : list
        Collection of aerodynamic surfaces of the rocket. Holds Nose cones,
        Fin sets, and Tails.
    Rocket.surfaces_cp_to_cdm : dict
        Dictionary containing the relative position of each aerodynamic surface
        center of pressure to the rocket's center of mass. The key is the
        aerodynamic surface object and the value is the relative position Vector
        in meters.
    Rocket.parachutes : list
        Collection of parachutes of the rocket.
    Rocket.air_brakes : list
        Collection of air brakes of the rocket.
    Rocket._controllers : list
        Collection of controllers of the rocket.
    Rocket.aerodynamic_center : Function
        Function of Mach number expressing the rocket's aerodynamic center
        (the linearized, small-incidence center of pressure) position relative
        to the user defined rocket reference system. ``Rocket.cp_position`` is an
        alias for this attribute. See :doc:`Positions and Coordinate Systems
        </user/positions>` for more information.
    Rocket.stability_margin : Function
        Stability margin of the rocket, in calibers, as a function of angle of
        attack (radians), mach number and time. Stability margin is defined as
        the distance between the center of pressure and the center of mass,
        divided by the rocket's diameter. The angle-of-attack argument matters
        only when a surface is nonlinear in incidence (see
        ``Rocket.is_incidence_linear``); otherwise it has no effect.
    Rocket.static_margin : Function
        Static margin of the rocket, in calibers, as a function of time. Static
        margin is defined as the distance between the center of pressure and the
        center of mass, divided by the rocket's diameter.
    Rocket.static_margin : float
        Float value corresponding to rocket static margin when
        loaded with propellant in units of rocket diameter or calibers.
    Rocket.power_off_drag : Function
        Rocket's drag coefficient as a function of Mach number when the
        motor is off. Alias for ``power_off_drag_by_mach``.
    Rocket.power_on_drag : Function
        Rocket's drag coefficient as a function of Mach number when the
        motor is on. Alias for ``power_on_drag_by_mach``.
    Rocket.power_off_drag_input : int, float, callable, string, array, Function
        Original user input for rocket's drag coefficient when the motor is
        off. Preserved for reconstruction and Monte Carlo workflows.
    Rocket.power_on_drag_input : int, float, callable, string, array, Function
        Original user input for rocket's drag coefficient when the motor is
        on. Preserved for reconstruction and Monte Carlo workflows.
    Rocket.power_off_drag_7d : AeroCoefficient
        Rocket's drag coefficient with motor off, callable over the seven
        independent variables (alpha, beta, mach, reynolds, pitch_rate,
        yaw_rate, roll_rate) and stored at its intrinsic dimensionality.
    Rocket.power_on_drag_7d : AeroCoefficient
        Rocket's drag coefficient with motor on, callable over the seven
        independent variables (alpha, beta, mach, reynolds, pitch_rate,
        yaw_rate, roll_rate) and stored at its intrinsic dimensionality.
    Rocket.power_off_drag_by_mach : Function
        Rocket's drag coefficient with motor off as a function of Mach number.
    Rocket.power_on_drag_by_mach : Function
        Rocket's drag coefficient with motor on as a function of Mach number.
    Rocket.rail_buttons : RailButtons
        RailButtons object containing the rail buttons information.
    Rocket.motor : Motor
        Rocket's motor. See Motor class for more details.
    Rocket.motor_position : float
        Position, in meters, of the motor's coordinate system origin
        relative to the user defined rocket coordinate system.
        See :doc:`Positions and Coordinate Systems </user/positions>`
        for more information.
        regarding the rocket's coordinate system.
    Rocket.nozzle_position : float
        Position, in meters, of the motor's nozzle exit relative to the user
        defined rocket coordinate system.
        See :doc:`Positions and Coordinate Systems </user/positions>`
        for more information.
    Rocket.nozzle_to_cdm : float
        Distance between the nozzle exit and the rocket's center of dry mass
        position, in meters.
    Rocket.nozzle_gyration_tensor: Matrix
        Matrix representing the nozzle gyration tensor.
    Rocket.center_of_propellant_position : Function
        Position of the propellant's center of mass relative to the user defined
        rocket reference system. See
        :doc:`Positions and Coordinate Systems </user/positions>` for more
        information regarding the rocket's coordinate system. Expressed in
        meters as a function of time.
    Rocket.I_11_without_motor : float
        Rocket's inertia tensor 11 component without any motors, in kg*m^2. This
        is the same value that is passed in the Rocket.__init__() method.
    Rocket.I_22_without_motor : float
        Rocket's inertia tensor 22 component without any motors, in kg*m^2. This
        is the same value that is passed in the Rocket.__init__() method.
    Rocket.I_33_without_motor : float
        Rocket's inertia tensor 33 component without any motors, in kg*m^2. This
        is the same value that is passed in the Rocket.__init__() method.
    Rocket.I_12_without_motor : float
        Rocket's inertia tensor 12 component without any motors, in kg*m^2. This
        is the same value that is passed in the Rocket.__init__() method.
    Rocket.I_13_without_motor : float
        Rocket's inertia tensor 13 component without any motors, in kg*m^2. This
        is the same value that is passed in the Rocket.__init__() method.
    Rocket.I_23_without_motor : float
        Rocket's inertia tensor 23 component without any motors, in kg*m^2. This
        is the same value that is passed in the Rocket.__init__() method.
    Rocket.dry_I_11 : float
        Rocket's inertia tensor 11 component with unloaded motor,in kg*m^2.
    Rocket.dry_I_22 : float
        Rocket's inertia tensor 22 component with unloaded motor,in kg*m^2.
    Rocket.dry_I_33 : float
        Rocket's inertia tensor 33 component with unloaded motor,in kg*m^2.
    Rocket.dry_I_12 : float
        Rocket's inertia tensor 12 component with unloaded motor,in kg*m^2.
    Rocket.dry_I_13 : float
        Rocket's inertia tensor 13 component with unloaded motor,in kg*m^2.
    Rocket.dry_I_23 : float
        Rocket's inertia tensor 23 component with unloaded motor,in kg*m^2.
    """

    def __init__(  # pylint: disable=too-many-statements
        self,
        radius,
        mass,
        inertia,
        power_off_drag,
        power_on_drag,
        center_of_mass_without_motor,
        coordinate_system_orientation="tail_to_nose",
    ):
        """Initializes Rocket class, process inertial, geometrical and
        aerodynamic parameters.

        Parameters
        ----------
        radius : int, float
            Rocket largest outer radius in meters.
        mass : int, float
            Rocket total mass without motor in kg.
        inertia : tuple, list
            Tuple or list containing the rocket's inertia tensor components,
            in kg*m^2. This should be measured without motor and propellant so
            that the inertia reference point is the
            `center_of_mass_without_motor`.
            Assuming e_3 is the rocket's axis of symmetry, e_1 and e_2 are
            orthogonal and form a plane perpendicular to e_3, the inertia tensor
            components must be given in the following order: (I_11, I_22, I_33,
            I_12, I_13, I_23), where I_ij is the component of the inertia tensor
            in the direction of e_i x e_j. Alternatively, the inertia tensor can
            be given as (I_11, I_22, I_33), where I_12 = I_13 = I_23 = 0. This
            can also be called as "rocket dry inertia tensor".
        power_off_drag : int, float, callable, string, array
            Rocket's drag coefficient when the motor is off. Can be given as an
            entry to the Function class. See help(Function) for more
            information. If int or float is given, it is assumed constant. If
            callable, string or array is given, it must be a function of Mach
            number only.
        power_on_drag : int, float, callable, string, array
            Rocket's drag coefficient when the motor is on. Can be given as an
            entry to the Function class. See help(Function) for more
            information. If int or float is given, it is assumed constant. If
            callable, string or array is given, it must be a function of Mach
            number only.
        center_of_mass_without_motor : int, float
            Position, in m, of the rocket's center of mass without motor
            relative to the rocket's coordinate system. Default is 0, which
            means the center of dry mass is chosen as the origin, to comply
            with the legacy behavior of versions 0.X.Y.
            See :doc:`Positions and Coordinate Systems </user/positions>`
            for more information
            regarding the rocket's coordinate system.
        coordinate_system_orientation : string, optional
            String defining the orientation of the rocket's coordinate system.
            The coordinate system is defined by the rocket's axis of symmetry.
            The system's origin may be placed anywhere along such axis, such as
            in the nozzle or in the nose cone, and must be kept the same for all
            other positions specified. The two options available are:
            "tail_to_nose" and "nose_to_tail". The first defines the coordinate
            system with the rocket's axis of symmetry pointing from the rocket's
            tail to the rocket's nose cone. The second option defines the
            coordinate system with the rocket's axis of symmetry pointing from
            the  rocket's nose cone to the rocket's tail. Default is
            "tail_to_nose".

        Returns
        -------
        None
        """
        # Define coordinate system orientation
        self.coordinate_system_orientation = coordinate_system_orientation
        match coordinate_system_orientation:
            case "tail_to_nose":
                self._csys = 1
            case "nose_to_tail":
                self._csys = -1
            case _:  # pragma: no cover
                raise TypeError(
                    "Invalid coordinate system orientation. Please choose between "
                    + '"tail_to_nose" and "nose_to_tail".'
                )

        # Define rocket inertia attributes in SI units
        self.mass = mass
        inertia = (*inertia, 0, 0, 0) if len(inertia) == 3 else inertia
        self.I_11_without_motor = inertia[0]
        self.I_22_without_motor = inertia[1]
        self.I_33_without_motor = inertia[2]
        self.I_12_without_motor = inertia[3]
        self.I_13_without_motor = inertia[4]
        self.I_23_without_motor = inertia[5]

        # Define rocket geometrical parameters in SI units
        self.center_of_mass_without_motor = center_of_mass_without_motor
        self.radius = radius
        self.area = np.pi * self.radius**2
        self._is_point_mass = False

        # Eccentricity data initialization
        self.cm_eccentricity_x = 0
        self.cm_eccentricity_y = 0
        self.cp_eccentricity_x = 0
        self.cp_eccentricity_y = 0
        self.thrust_eccentricity_y = 0
        self.thrust_eccentricity_x = 0

        # Parachute, Aerodynamic, Buttons, Controllers, Sensor data initialization
        self.parachutes = []
        self._controllers = []
        self.air_brakes = []
        self.sensors = Components()
        self.sensors_by_name = {}
        self.aerodynamic_surfaces = Components()
        self.surfaces_cp_to_cdm = {}
        # Set once a full-body model replaces the modeled aerodynamics
        # (add_full_body_aerodynamics(overwrite=True)); warns on later surface adds.
        self._aerodynamics_overwritten = False
        self.rail_buttons = Components()

        self._aerodynamic_center = Function(
            lambda mach: 0,
            inputs="Mach Number",
            outputs="Aerodynamic Center Position (m)",
        )
        self._total_lift_coeff_der = Function(
            lambda mach: 0,
            inputs="Mach Number",
            outputs="Total Lift Coefficient Derivative",
        )
        self._static_margin = Function(
            lambda time: 0, inputs="Time (s)", outputs="Static Margin (c)"
        )
        self._stability_margin = Function(
            lambda alpha, mach, time: 0,
            inputs=["Angle of Attack (rad)", "Mach", "Time (s)"],
            outputs="Stability Margin (c)",
        )
        # Yaw-plane counterparts. The pitch-plane attributes above remain the
        # primary (default) margin; these expose the yaw plane for
        # non-axisymmetric rockets (see ``evaluate_center_of_pressure``).
        self._aerodynamic_center_yaw = Function(
            lambda mach: 0,
            inputs="Mach Number",
            outputs="Aerodynamic Center Position - Yaw (m)",
        )
        self._total_side_coeff_der = Function(
            lambda mach: 0,
            inputs="Mach Number",
            outputs="Total Side Coefficient Derivative",
        )
        self._static_margin_yaw = Function(
            lambda time: 0, inputs="Time (s)", outputs="Static Margin - Yaw (c)"
        )
        self._stability_margin_yaw = Function(
            lambda beta, mach, time: 0,
            inputs=["Sideslip Angle (rad)", "Mach", "Time (s)"],
            outputs="Stability Margin - Yaw (c)",
        )

        # Define aerodynamic drag coefficients used during flight simulation
        self.power_off_drag_7d = AeroCoefficient(
            power_off_drag,
            name="Drag Coefficient with Power Off",
            extrapolation="constant",
            single_var="mach",
        )
        self.power_on_drag_7d = AeroCoefficient(
            power_on_drag,
            name="Drag Coefficient with Power On",
            extrapolation="constant",
            single_var="mach",
        )
        self.power_on_drag_by_mach = Function(
            lambda mach: self.power_on_drag_7d(0, 0, mach, 0, 0, 0, 0),
            inputs="Mach Number",
            outputs="Drag Coefficient with Power On",
            interpolation="linear",
            extrapolation="constant",
        )
        self.power_off_drag_by_mach = Function(
            lambda mach: self.power_off_drag_7d(0, 0, mach, 0, 0, 0, 0),
            inputs="Mach Number",
            outputs="Drag Coefficient with Power Off",
            interpolation="linear",
            extrapolation="constant",
        )
        # Saving raw user input for reconstruction and Monte Carlo
        self._power_off_drag_input = power_off_drag
        self._power_on_drag_input = power_on_drag
        # Public API attributes: keep as Function (Mach-only) for backward compatibility
        self.power_off_drag = self.power_off_drag_by_mach
        self.power_on_drag = self.power_on_drag_by_mach

        # Create a, possibly, temporary empty motor
        # self.motors = Components()  # currently unused, only 1 motor is supported
        self.add_motor(motor=EmptyMotor(), position=0)

        # Important dynamic inertial quantities
        self.center_of_mass = None
        self.reduced_mass = None
        self.total_mass = None
        self.dry_mass = None

        # calculate dynamic inertial quantities
        self.evaluate_dry_mass()
        self.evaluate_structural_mass_ratio()
        self.evaluate_total_mass()
        self.evaluate_center_of_dry_mass()
        self.evaluate_center_of_mass()
        self.evaluate_reduced_mass()
        self.evaluate_thrust_to_weight()

        # The aerodynamic center and the margins are evaluated lazily
        self._cp_outdated = True
        self._margin_outdated = True
        # Whether the neutral point moves with angle of attack; set when the
        # margins are evaluated (see evaluate_stability_margin).
        self._is_incidence_linear = True
        # Flag for rocket non-axisymmetric warning. Used to show warning once.
        self._axisymmetry_warned = False

        # Initialize plots and prints object
        self.prints = _RocketPrints(self)
        self.plots = _RocketPlots(self)

    def _check_missing_components(self):
        """Check if the rocket is missing any essential components and issue a warning.

        This method verifies whether the rocket has the following key components:
        - motor
        - aerodynamic surface(s)

        If any of these components are missing, a single warning message is issued
        listing all missing components. This helps users quickly identify potential
        issues before running simulations or analyses.

        Notes
        -----
        - The warning uses Python's built-in `warnings.warn` function.

        Returns
        -------
        None
        """
        missing_components = []
        if isinstance(self.motor, EmptyMotor):
            missing_components.append("motor")
        if not self.aerodynamic_surfaces:
            missing_components.append("aerodynamic surfaces")

        if missing_components:
            component_list = ", ".join(missing_components)
            warnings.warn(f"Rocket has no {component_list} defined.", UserWarning)

    @property
    def nosecones(self):
        """A list containing all the nose cones currently added to the rocket."""
        return self.aerodynamic_surfaces.get_by_type(NoseCone)

    @property
    def fins(self):
        """A list containing all the fins currently added to the rocket."""
        return self.aerodynamic_surfaces.get_by_type(Fins)

    @property
    def tails(self):
        """A list with all the tails currently added to the rocket"""
        return self.aerodynamic_surfaces.get_by_type(Tail)

    def evaluate_total_mass(self):
        """Calculates and returns the rocket's total mass. The total
        mass is defined as the sum of the motor mass with propellant and the
        rocket mass without propellant. The function returns an object
        of the Function class and is defined as a function of time.

        Returns
        -------
        self.total_mass : Function
            Function of time expressing the total mass of the rocket,
            defined as the sum of the propellant mass and the rocket
            mass without propellant.
        """
        # Make sure there is a motor associated with the rocket
        if self.motor is None:
            print("Please associate this rocket with a motor!")
            return False

        self.total_mass = self.mass + self.motor.total_mass
        self.total_mass.set_outputs("Total Mass (Rocket + Motor + Propellant) (kg)")
        self.total_mass.set_title("Total Mass (Rocket + Motor + Propellant)")
        return self.total_mass

    def evaluate_dry_mass(self):
        """Calculates and returns the rocket's dry mass. The dry
        mass is defined as the sum of the motor's dry mass and the
        rocket mass without motor.

        Returns
        -------
        self.dry_mass : float
            Rocket's dry mass (Rocket + Motor) (kg)
        """
        # Make sure there is a motor associated with the rocket
        if self.motor is None:
            print("Please associate this rocket with a motor!")
            return False

        self.dry_mass = self.mass + self.motor.dry_mass

        return self.dry_mass

    def evaluate_structural_mass_ratio(self):
        """Calculates and returns the rocket's structural mass ratio.
        It is defined as the ratio between of the dry mass
        (Motor + Rocket) and the initial total mass
        (Motor + Propellant + Rocket).

        Returns
        -------
        self.structural_mass_ratio: float
            Initial structural mass ratio dry mass (Rocket + Motor) (kg)
            divided by total mass (Rocket + Motor + Propellant) (kg).
        """
        try:
            self.structural_mass_ratio = self.dry_mass / (
                self.dry_mass + self.motor.propellant_initial_mass
            )
        except ZeroDivisionError as e:
            raise ValueError(
                "Total rocket mass (dry + propellant) cannot be zero"
            ) from e
        return self.structural_mass_ratio

    def evaluate_center_of_mass(self):
        """Evaluates rocket center of mass position relative to user defined
        rocket reference system.

        Returns
        -------
        self.center_of_mass : Function
            Function of time expressing the rocket's center of mass position
            relative to user defined rocket reference system.
            See :doc:`Positions and Coordinate Systems </user/positions>`
            for more information.
        """
        self.center_of_mass = (
            self.center_of_mass_without_motor * self.mass
            + self.motor_center_of_mass_position * self.motor.total_mass
        ) / self.total_mass
        self.center_of_mass.set_inputs("Time (s)")
        self.center_of_mass.set_outputs("Center of Mass Position (m)")
        self.center_of_mass.set_title(
            "Center of Mass Position (Rocket + Motor + Propellant)"
        )
        return self.center_of_mass

    def evaluate_center_of_dry_mass(self):
        """Evaluates the rocket's center of dry mass (i.e. rocket with motor but
        without propellant) position relative to user defined rocket reference
        system.

        Returns
        -------
        self.center_of_dry_mass_position : int, float
            Rocket's center of dry mass position (with unloaded motor)
        """
        self.center_of_dry_mass_position = (
            self.center_of_mass_without_motor * self.mass
            + self.motor_center_of_dry_mass_position * self.motor.dry_mass
        ) / self.dry_mass
        return self.center_of_dry_mass_position

    def evaluate_reduced_mass(self):
        """Calculates and returns the rocket's total reduced mass. The reduced
        mass is defined as the product of the propellant mass and the rocket dry
        mass (i.e. with unloaded motor), divided by the loaded rocket mass.
        The function returns an object of the Function class and is defined as a
        function of time.

        Returns
        -------
        self.reduced_mass : Function
            Function of time expressing the reduced mass of the rocket.
        """
        # TODO: add tests for reduced_mass values
        # Make sure there is a motor associated with the rocket
        if self.motor is None:
            print("Please associate this rocket with a motor!")
            return False

        # Get nicknames
        prop_mass = self.motor.propellant_mass
        dry_mass = self.dry_mass
        # calculate reduced mass and return it
        self.reduced_mass = prop_mass * dry_mass / (prop_mass + dry_mass)
        self.reduced_mass.set_outputs("Reduced Mass (kg)")
        self.reduced_mass.set_title("Reduced Mass")
        return self.reduced_mass

    def evaluate_thrust_to_weight(self):
        """Evaluates thrust to weight as a Function of time. This is defined as
        the motor thrust force divided by rocket weight. The gravitational
        acceleration is assumed constant and equals to 9.80665 m/s^2.

        Returns
        -------
        None
        """
        self.thrust_to_weight = self.motor.thrust / (9.80665 * self.total_mass)
        self.thrust_to_weight.set_inputs("Time (s)")
        self.thrust_to_weight.set_outputs("Thrust/Weight")
        self.thrust_to_weight.set_title("Thrust to Weight ratio")

    # Lazily-evaluated aerodynamic outputs.

    def _ensure_aerodynamic_center(self):
        """Recompute the pitch/yaw aerodynamic centers if a surface changed."""
        if self._cp_outdated:
            self.evaluate_center_of_pressure()  # clears ``_cp_outdated``

    def _ensure_margins(self):
        """Recompute the static/stability margins if a surface or the center of
        mass changed. The underlying aerodynamic center is refreshed lazily by
        the margin source closures."""
        if self._margin_outdated:
            self._margin_outdated = False
            self.evaluate_stability_margin()
            self.evaluate_static_margin()

    @property
    def aerodynamic_center(self):
        """Pitch-plane aerodynamic center vs Mach (lazily evaluated)."""
        self._ensure_aerodynamic_center()
        return self._aerodynamic_center

    @property
    def aerodynamic_center_yaw(self):
        """Yaw-plane aerodynamic center vs Mach (lazily evaluated)."""
        self._ensure_aerodynamic_center()
        return self._aerodynamic_center_yaw

    def neutral_point(self, alpha, mach):
        """Pitch-plane neutral point at a finite angle of attack, in meters.

        The neutral point is the point about which the aerodynamic pitching
        moment does not change for a small change in angle of attack. It is the
        angle-of-attack-aware generalization of the
        :attr:`aerodynamic_center`: evaluated at ``alpha = 0`` the two are equal,
        and for a rocket built only from the linear Barrowman surfaces the
        neutral point does not move with angle of attack at all.

        It moves with angle of attack only when a surface's normal force is
        nonlinear in the angle of attack, for example a Galejs body-lift term
        (growing like ``sin**2(alpha)``) added as a
        :class:`rocketpy.GenericSurface`. In that case the neutral point migrates
        as the angle of attack changes, exactly the behavior OpenRocket models,
        and the flight stability margin follows it.

        Parameters
        ----------
        alpha : float
            Angle of attack, in radians, to evaluate the neutral point at.
        mach : float
            Free-stream Mach number.

        Returns
        -------
        float
            Axial position of the pitch-plane neutral point in the user-defined
            rocket coordinate system, in meters.
        """
        return neutral_point_and_slope(self, alpha, 0.0, mach, "pitch")[0]

    def neutral_point_yaw(self, beta, mach):
        """Yaw-plane neutral point at a finite sideslip angle, in meters.

        Yaw-plane counterpart of :meth:`neutral_point`: the point about which the
        yaw moment does not change for a small change in sideslip angle,
        evaluated at the given sideslip angle. Equal to
        :attr:`aerodynamic_center_yaw` at ``beta = 0``.

        Parameters
        ----------
        beta : float
            Sideslip angle, in radians, to evaluate the neutral point at.
        mach : float
            Free-stream Mach number.

        Returns
        -------
        float
            Axial position of the yaw-plane neutral point in the user-defined
            rocket coordinate system, in meters.
        """
        return neutral_point_and_slope(self, 0.0, beta, mach, "yaw")[0]

    @property
    def total_lift_coeff_der(self):
        """Total normal-force-coefficient derivative vs Mach (lazily evaluated)."""
        self._ensure_aerodynamic_center()
        return self._total_lift_coeff_der

    @property
    def total_side_coeff_der(self):
        """Total side-force-coefficient derivative vs Mach (lazily evaluated)."""
        self._ensure_aerodynamic_center()
        return self._total_side_coeff_der

    @property
    def static_margin(self):
        """Pitch-plane static margin (calibers) vs time (lazily evaluated)."""
        self._ensure_margins()
        return self._static_margin

    @property
    def static_margin_yaw(self):
        """Yaw-plane static margin (calibers) vs time (lazily evaluated)."""
        self._ensure_margins()
        return self._static_margin_yaw

    @property
    def stability_margin(self):
        """Pitch-plane stability margin (calibers) as a function of angle of
        attack (radians), Mach and time (lazily evaluated). The angle-of-attack
        argument matters only for a rocket that is nonlinear in incidence (see
        :attr:`is_incidence_linear`); otherwise it has no effect and the margin
        reduces to the Mach-and-time value."""
        self._ensure_margins()
        return self._stability_margin

    @property
    def stability_margin_yaw(self):
        """Yaw-plane stability margin (calibers) as a function of sideslip angle
        (radians), Mach and time (lazily evaluated). Equal to
        :attr:`stability_margin` for an axisymmetric rocket."""
        self._ensure_margins()
        return self._stability_margin_yaw

    @property
    def length(self):
        """Overall aerodynamic length of the rocket, in meters.

        This is the axial distance from the fore-most point of the rocket (the
        nose cone tip) to the aft-most point of the rocket. It is measured along
        the rocket axis and does not depend on the chosen coordinate-system
        orientation.

        The aft-most point is usually the trailing edge of the last fin set or
        the base of the aft tail, but if the motor nozzle extends past the last
        aerodynamic surface, the nozzle sets the aft end instead. The rocket
        must have at least one aerodynamic surface with a defined axial extent
        (a nose cone, tail or fin set); otherwise a ``ValueError`` is raised.

        This length is what the hobby-rocketry convention of expressing the
        static/stability margin as a *percentage of body length* is measured
        against, as opposed to the caliber (diameter) convention used by
        ``static_margin`` and ``stability_margin``.

        Returns
        -------
        float
            Overall aerodynamic length of the rocket, in meters.
        """
        fore_points = []
        aft_points = []
        for surface, position in self.aerodynamic_surfaces:
            if isinstance(surface, (NoseCone, Tail)):
                axial_extent = surface.length
            elif isinstance(surface, (Fins, Fin)):
                axial_extent = surface.root_chord
            else:
                # Generic/controllable surfaces have no defined axial extent;
                # they contribute a single point at their reference position.
                axial_extent = 0.0
            # The reference point and the point one axial extent toward the tail
            # (the tail direction is -_csys along the z axis). Taking the global
            # extremes makes the result independent of which end is the reference.
            fore_points.append(position.z)
            aft_points.append(position.z - self._csys * axial_extent)

        if not fore_points:
            raise ValueError(
                "The rocket must have at least one aerodynamic surface to have a "
                "defined length."
            )

        all_points = fore_points + aft_points
        # Include the nozzle if a real motor extends past the aerodynamic
        # surfaces. nozzle_position is already in the rocket reference frame.
        if getattr(self, "motor", None) is not None and not isinstance(
            self.motor, EmptyMotor
        ):
            all_points.append(self.nozzle_position)
        return max(all_points) - min(all_points)

    def evaluate_center_of_pressure(self):
        """Evaluates the rocket's aerodynamic center (and cp_position) as a
        function of Mach number, relative to the user-defined rocket reference
        system.

        The aerodynamic center is the linearized (small-incidence, alpha=beta=0)
        center of pressure: the normal-force-slope-weighted average of every
        aerodynamic surface's location.

        It is computed independently for the **pitch** plane
        (``aerodynamic_center``, from the normal-force/pitch-moment slopes) and
        the **yaw** plane (``aerodynamic_center_yaw``, from the
        side-force/yaw-moment slopes). For an axisymmetric rocket the two
        coincide. When they differ (a non-axisymmetric configuration), a warning
        is raised because the scalar ``static_margin``/``stability_margin``
        attributes describe the pitch plane only.

        Returns
        -------
        self.aerodynamic_center : Function
            Function of Mach number expressing the rocket's pitch-plane
            aerodynamic center position relative to the user-defined rocket
            reference system. See :doc:`Positions and Coordinate Systems
            </user/positions>` for more information.
        """
        # Mark the pitch/yaw centers up to date before computing, so that a read
        # of the ``aerodynamic_center`` property during this method (the
        # ``is_axisymmetric`` check below) returns the value being built here
        # rather than recursing back into this method.
        self._cp_outdated = False

        # Re-Initialize total force coefficient derivatives and AC positions
        self._total_lift_coeff_der.set_source(lambda mach: 0)
        self._aerodynamic_center.set_source(lambda mach: 0)
        self._total_side_coeff_der.set_source(lambda mach: 0)
        self._aerodynamic_center_yaw.set_source(lambda mach: 0)

        # Calculate total force coefficient derivatives and aerodynamic center
        if len(self.aerodynamic_surfaces) > 0:
            for aero_surface, position in self.aerodynamic_surfaces:
                # Force-curve slopes as Functions of Mach, from the surface's
                # coefficient derivatives sliced at zero alpha/beta and zero
                # rates. The yaw slope is the sign-flipped ``cY_beta`` so an
                # axisymmetric surface gives the same signed weight as the pitch
                # plane (their margins then coincide when symmetric).
                lift_coeff_der = aero_surface.cN_alpha.slice("mach")
                side_coeff_der = -1.0 * aero_surface.cY_beta.slice("mach")
                cp_z = aero_surface.center_of_pressure_z
                cp_z_yaw = aero_surface.center_of_pressure_z_yaw
                # ref_factor corrects force for different reference areas
                ref_factor = aero_surface.reference_area / self.area
                self._total_lift_coeff_der += ref_factor * lift_coeff_der
                self._aerodynamic_center += (
                    ref_factor * lift_coeff_der * (position.z - self._csys * cp_z)
                )

                # Yaw plane.
                self._total_side_coeff_der += ref_factor * side_coeff_der
                self._aerodynamic_center_yaw += (
                    ref_factor * side_coeff_der * (position.z - self._csys * cp_z_yaw)
                )
            # Avoid errors when only zero-lift surfaces are added
            if self._total_lift_coeff_der.get_value(0) != 0:
                self._aerodynamic_center /= self._total_lift_coeff_der
            if self._total_side_coeff_der.get_value(0) != 0:
                self._aerodynamic_center_yaw /= self._total_side_coeff_der

        # Non-axisymmetry advisory. Latched once per configuration: the flag is
        # re-armed whenever a surface is added (see add_surfaces)
        if not self._axisymmetry_warned and not self.is_axisymmetric:
            self._axisymmetry_warned = True
            max_diff = self._cp_plane_max_difference()
            warnings.warn(
                "Pitch- and yaw-plane aerodynamic centers differ "
                f"(max difference ~{max_diff:.4g} m): the rocket is not "
                "axisymmetric. 'aerodynamic_center', 'static_margin' and "
                "'stability_margin' describe the PITCH plane. Use "
                "'aerodynamic_center_yaw', 'static_margin_yaw' and "
                "'stability_margin_yaw' for the yaw plane.",
                stacklevel=2,
            )

        return self._aerodynamic_center

    def _cp_plane_max_difference(self):
        """Largest pitch- vs yaw-plane aerodynamic center difference, in meters.
        The difference is sampled densely across the subsonic, transonic and"""
        # 0 to 3 in 0.2 steps covers RocketPy's flight regimes (sub/trans/
        # supersonic) with enough resolution that a real asymmetry, which spans a
        # Mach *range*, cannot fall entirely between sample points. This only
        # runs for rockets that contain a generic surface or individual fin (see
        # the by-construction short-circuit in ``is_axisymmetric``).
        sample_machs = np.linspace(0.0, 3.0, 16)
        return max(
            abs(
                self.aerodynamic_center.get_value_opt(mach)
                - self.aerodynamic_center_yaw.get_value_opt(mach)
            )
            for mach in sample_machs
        )

    @property
    def is_axisymmetric(self):
        """``True`` when the rocket's pitch- and yaw-plane aerodynamic centers
        coincide (to caliber-scale tolerance). When ``False`` the rocket is not
        axisymmetric: ``aerodynamic_center``, ``static_margin`` and
        ``stability_margin`` describe the PITCH plane only and differ from their
        ``*_yaw`` counterparts (``aerodynamic_center_yaw``,
        ``static_margin_yaw``, ``stability_margin_yaw``)."""
        # Nose, tail and fin sets contribute identically to both planes.
        if all(
            getattr(surface, "is_axisymmetric", False)
            for surface, _ in self.aerodynamic_surfaces
        ):
            return True
        # Tolerance relative to the rocket diameter (caliber-scale).
        return self._cp_plane_max_difference() <= 1e-6 * (2 * self.radius)

    @property
    def is_incidence_linear(self):
        """``True`` when the rocket's aerodynamics are linear in the angle of
        attack, so the neutral point (and therefore the stability margin) does
        not move as the angle of attack changes. This holds for a rocket built
        only from the linear Barrowman surfaces. It is ``False`` when a surface's
        normal force is nonlinear in incidence, such as a Galejs body-lift term
        added as a :class:`rocketpy.GenericSurface`, in which case
        :attr:`stability_margin` varies with its angle-of-attack argument."""
        self._ensure_margins()
        return self._is_incidence_linear

    @property
    def cp_position(self):
        """Alias for :attr:`aerodynamic_center`. Traditional center of pressure
        position, defined as the linearized (small-incidence) center of pressure,
        is the same as the aerodynamic center."""
        return self.aerodynamic_center

    def evaluate_surfaces_cp_to_cdm(self):
        """Calculates the relative position of each aerodynamic surface center
        of pressure to the rocket's center of dry mass in Body Axes Coordinate
        System.

        Returns
        -------
        self.surfaces_cp_to_cdm : dict
            Dictionary mapping the relative position of each aerodynamic
            surface center of pressure to the rocket's center of mass.
        """
        for surface, position in self.aerodynamic_surfaces:
            self.__evaluate_single_surface_cp_to_cdm(surface, position)
        return self.surfaces_cp_to_cdm

    def __evaluate_single_surface_cp_to_cdm(self, surface, position):
        """Calculates the relative position of each aerodynamic surface
        center of pressure to the rocket's center of dry mass in Body Axes
        Coordinate System."""
        # position of the surfaces coordinate system origin in body frame
        pos_origin = Vector(
            [
                (position.x - self.cm_eccentricity_x) * self._csys,
                (position.y - self.cm_eccentricity_y),
                (position.z - self.center_of_dry_mass_position) * self._csys,
            ]
        )
        # position of the force application point in body frame. Every surface
        # applies its resultant force at its center of pressure and transports
        # the moment geometrically; the surface-local application point is mapped
        # into the body frame by ``_rotation_surface_to_body``
        application_point = getattr(
            surface,
            "force_application_point",
            Vector([surface.cpx, surface.cpy, surface.cpz]),
        )
        pos = (
            surface._rotation_surface_to_body @ application_point + pos_origin
        )  # TODO: this should be recomputed whenever cant angle changes for fin
        self.surfaces_cp_to_cdm[surface] = pos

    def _evaluate_is_incidence_linear(self):
        """Detect whether the rocket's aerodynamics are linear in the angle of
        attack, i.e. whether the neutral point moves as the angle of attack
        changes. Returns ``True`` for a rocket built only from the linear
        Barrowman surfaces and ``False`` when a surface's normal force is
        nonlinear in incidence (for example a Galejs body-lift term added as a
        :class:`rocketpy.GenericSurface`).

        The check central-differences the neutral point at zero and at five
        degrees of incidence, in both the pitch and yaw planes; if either plane
        moves, the rocket is treated as incidence-nonlinear.
        """
        probe_mach = 0.3
        probe_alpha = math.radians(5.0)
        for plane in ("pitch", "yaw"):
            if plane == "yaw":
                point_zero = neutral_point_and_slope(self, 0.0, 0.0, probe_mach, "yaw")
                point_five = neutral_point_and_slope(
                    self, 0.0, probe_alpha, probe_mach, "yaw"
                )
            else:
                point_zero = neutral_point_and_slope(
                    self, 0.0, 0.0, probe_mach, "pitch"
                )
                point_five = neutral_point_and_slope(
                    self, probe_alpha, 0.0, probe_mach, "pitch"
                )
            if abs(point_five[0] - point_zero[0]) > 1e-6:
                return False
        return True

    def _neutral_point_margin_slope(self, incidence, mach, time, plane="pitch"):
        """Stability margin (in calibers) and local normal-force-curve slope at a
        single flow state, the shared computation behind ``stability_margin`` and
        the flight dynamic-stability oscillator.

        For a rocket that is linear in incidence the neutral point is the
        zero-incidence :attr:`aerodynamic_center` and ``incidence`` is ignored,
        keeping the fast analytic path (and byte-for-byte the previous margin
        values). Otherwise the neutral point and slope are found at ``incidence``
        by :func:`rocketpy.rocket.helpers.neutral_point_and_slope`, so a surface
        that is nonlinear in the angle of attack (e.g. a Galejs body-lift
        :class:`rocketpy.GenericSurface`) makes the margin move with incidence.

        Parameters
        ----------
        incidence : float
            Angle of attack (pitch) or sideslip angle (yaw), in radians.
        mach : float
            Free-stream Mach number.
        time : float
            Flight time, in seconds, at which the center of mass is taken.
        plane : str, optional
            ``"pitch"`` or ``"yaw"``. Default ``"pitch"``.

        Returns
        -------
        tuple of float
            ``(margin, slope)``: the stability margin in calibers and the local
            force-curve slope (``dCN/dalpha`` for pitch, ``dCY/dbeta`` for yaw).
        """
        self._ensure_margins()
        if plane == "yaw":
            center = self.aerodynamic_center_yaw
            slope_curve = self.total_side_coeff_der
        else:
            center = self.aerodynamic_center
            slope_curve = self.total_lift_coeff_der

        if self._is_incidence_linear:
            neutral_point = center.get_value_opt(mach)
            slope = slope_curve.get_value_opt(mach)
        elif plane == "yaw":
            neutral_point, slope = neutral_point_and_slope(
                self, 0.0, incidence, mach, "yaw"
            )
        else:
            neutral_point, slope = neutral_point_and_slope(
                self, incidence, 0.0, mach, "pitch"
            )

        margin = (
            self._csys
            * (self.center_of_mass.get_value_opt(time) - neutral_point)
            / (2 * self.radius)
        )
        return margin, slope

    def evaluate_stability_margin(self):
        """Calculates the stability margin of the rocket as a function of angle
        of attack, Mach number and time.

        Returns
        -------
        stability_margin : Function
            Stability margin of the rocket, in calibers, as a function of angle
            of attack (radians), Mach number and time. The stability margin is
            the distance between the center of pressure and the center of mass,
            divided by the rocket's diameter. It depends on the angle of attack
            only when a surface is nonlinear in incidence; for a rocket built
            from the linear Barrowman surfaces the angle-of-attack argument has
            no effect.
        """
        self._is_incidence_linear = self._evaluate_is_incidence_linear()
        self._stability_margin.set_source(
            lambda alpha, mach, time: self._neutral_point_margin_slope(
                alpha, mach, time, "pitch"
            )[0]
        )
        self._stability_margin.set_inputs(["Angle of Attack (rad)", "Mach", "Time (s)"])
        # Yaw-plane stability margin (equal to the pitch plane when axisymmetric)
        self._stability_margin_yaw.set_source(
            lambda beta, mach, time: self._neutral_point_margin_slope(
                beta, mach, time, "yaw"
            )[0]
        )
        self._stability_margin_yaw.set_inputs(
            ["Sideslip Angle (rad)", "Mach", "Time (s)"]
        )
        return self._stability_margin

    def evaluate_static_margin(self):
        """Calculates the static margin of the rocket as a function of time.

        Returns
        -------
        static_margin : Function
            Static margin of the rocket, in calibers, as a function of time.
            Static margin is defined as the distance between the center of
            pressure and the center of mass, divided by the rocket's diameter.
        """
        # Calculate static margin
        self._static_margin.set_source(
            lambda time: (
                (
                    self.center_of_mass.get_value_opt(time)
                    - self.aerodynamic_center.get_value_opt(0)
                )
                / (2 * self.radius)
            )
        )
        # Change sign if coordinate system is upside down
        self._static_margin *= self._csys
        self._static_margin.set_inputs("Time (s)")
        self._static_margin.set_outputs("Static Margin (c)")
        self._static_margin.set_title("Static Margin")
        self._static_margin.set_discrete(
            lower=0, upper=self.motor.burn_out_time, samples=200
        )

        # Yaw-plane static margin (equal to the pitch plane when axisymmetric)
        self._static_margin_yaw.set_source(
            lambda time: (
                (
                    self.center_of_mass.get_value_opt(time)
                    - self.aerodynamic_center_yaw.get_value_opt(0)
                )
                / (2 * self.radius)
            )
        )
        self._static_margin_yaw *= self._csys
        self._static_margin_yaw.set_inputs("Time (s)")
        self._static_margin_yaw.set_outputs("Static Margin - Yaw (c)")
        self._static_margin_yaw.set_title("Static Margin - Yaw")
        self._static_margin_yaw.set_discrete(
            lower=0, upper=self.motor.burn_out_time, samples=200
        )
        return self._static_margin

    def evaluate_dry_inertias(self):
        """Calculates and returns the rocket's dry inertias relative to
        the rocket's center of dry mass. The inertias are saved and returned
        in units of kg*m². This does not consider propellant mass but does take
        into account the motor dry mass.

        Returns
        -------
        self.dry_I_11 : float
            Float value corresponding to rocket inertia tensor 11
            component, which corresponds to the inertia relative to the
            e_1 axis, centered at the center of dry mass.
        self.dry_I_22 : float
            Float value corresponding to rocket inertia tensor 22
            component, which corresponds to the inertia relative to the
            e_2 axis, centered at the center of dry mass.
        self.dry_I_33 : float
            Float value corresponding to rocket inertia tensor 33
            component, which corresponds to the inertia relative to the
            e_3 axis, centered at the center of dry mass.
        self.dry_I_12 : float
            Float value corresponding to rocket inertia tensor 12
            component, which corresponds to the inertia relative to the
            e_1 and e_2 axes, centered at the center of dry mass.
        self.dry_I_13 : float
            Float value corresponding to rocket inertia tensor 13
            component, which corresponds to the inertia relative to the
            e_1 and e_3 axes, centered at the center of dry mass.
        self.dry_I_23 : float
            Float value corresponding to rocket inertia tensor 23
            component, which corresponds to the inertia relative to the
            e_2 and e_3 axes, centered at the center of dry mass.

        Notes
        -----
        #. The ``e_1`` and ``e_2`` directions are assumed to be the directions \
            perpendicular to the rocket axial direction.
        #. The ``e_3`` direction is assumed to be the direction parallel to the \
            axis of symmetry of the rocket.
        #. RocketPy follows the definition of the inertia tensor that includes \
            the minus sign for all products of inertia.

        See Also
        --------
        `Inertia Tensor <https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor>`_
        """
        # Get masses
        motor_dry_mass = self.motor.dry_mass
        mass = self.mass

        # Compute axes distances (CDM: Center of Dry Mass)
        center_of_mass_without_motor_to_CDM = (
            self.center_of_mass_without_motor - self.center_of_dry_mass_position
        )
        motor_center_of_dry_mass_to_CDM = (
            self.motor_center_of_dry_mass_position - self.center_of_dry_mass_position
        )

        # Compute dry inertias
        self.dry_I_11 = parallel_axis_theorem_from_com(
            self.I_11_without_motor, mass, center_of_mass_without_motor_to_CDM
        ) + parallel_axis_theorem_from_com(
            self.motor.dry_I_11, motor_dry_mass, motor_center_of_dry_mass_to_CDM
        )

        self.dry_I_22 = parallel_axis_theorem_from_com(
            self.I_22_without_motor, mass, center_of_mass_without_motor_to_CDM
        ) + parallel_axis_theorem_from_com(
            self.motor.dry_I_22, motor_dry_mass, motor_center_of_dry_mass_to_CDM
        )

        self.dry_I_33 = self.I_33_without_motor + self.motor.dry_I_33
        self.dry_I_12 = self.I_12_without_motor + self.motor.dry_I_12
        self.dry_I_13 = self.I_13_without_motor + self.motor.dry_I_13
        self.dry_I_23 = self.I_23_without_motor + self.motor.dry_I_23

        return (
            self.dry_I_11,
            self.dry_I_22,
            self.dry_I_33,
            self.dry_I_12,
            self.dry_I_13,
            self.dry_I_23,
        )

    def evaluate_inertias(self):
        """Calculates and returns the rocket's inertias relative to
        the rocket's center of dry mass. The inertias are saved and returned
        in units of kg*m².

        Returns
        -------
        self.I_11 : float
            Float value corresponding to rocket inertia tensor 11
            component, which corresponds to the inertia relative to the
            e_1 axis, centered at the center of dry mass.
        self.I_22 : float
            Float value corresponding to rocket inertia tensor 22
            component, which corresponds to the inertia relative to the
            e_2 axis, centered at the center of dry mass.
        self.I_33 : float
            Float value corresponding to rocket inertia tensor 33
            component, which corresponds to the inertia relative to the
            e_3 axis, centered at the center of dry mass.

        Notes
        -----
        #. The ``e_1`` and ``e_2`` directions are assumed to be the directions \
            perpendicular to the rocket axial direction.
        #. The ``e_3`` direction is assumed to be the direction parallel to the \
            axis of symmetry of the rocket.
        #. RocketPy follows the definition of the inertia tensor that includes \
            the minus sign for all products of inertia.

        See Also
        --------
        `Inertia Tensor <https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor>`_
        """
        # Get masses
        prop_mass = self.motor.propellant_mass  # Propellant mass as a function of time

        # Compute axes distances
        CDM_to_CPM = (
            self.center_of_dry_mass_position - self.center_of_propellant_position
        )

        # Compute inertias
        self.I_11 = self.dry_I_11 + parallel_axis_theorem_from_com(
            self.motor.propellant_I_11, prop_mass, CDM_to_CPM
        )

        self.I_22 = self.dry_I_22 + parallel_axis_theorem_from_com(
            self.motor.propellant_I_22, prop_mass, CDM_to_CPM
        )

        self.I_33 = self.dry_I_33 + self.motor.propellant_I_33
        self.I_12 = self.dry_I_12 + self.motor.propellant_I_12
        self.I_13 = self.dry_I_13 + self.motor.propellant_I_13
        self.I_23 = self.dry_I_23 + self.motor.propellant_I_23

        # Return inertias
        return (
            self.I_11,
            self.I_22,
            self.I_33,
            self.I_12,
            self.I_13,
            self.I_23,
        )

    def evaluate_nozzle_to_cdm(self):
        """Evaluates the distance between the nozzle exit and the rocket's
        center of dry mass.

        Returns
        -------
        self.nozzle_to_cdm : float
            Distance between the nozzle exit and the rocket's center of dry
            mass position, in meters.
        """
        self.nozzle_to_cdm = (
            -(self.nozzle_position - self.center_of_dry_mass_position) * self._csys
        )
        return self.nozzle_to_cdm

    def evaluate_nozzle_gyration_tensor(self):
        """Calculates and returns the nozzle gyration tensor relative to the
        rocket's center of dry mass. The gyration tensor is saved and returned
        in units of kg*m².

        Returns
        -------
        self.nozzle_gyration_tensor : Matrix
            Matrix containing the nozzle gyration tensor.
        """
        S_noz_33 = 0.5 * self.motor.nozzle_radius**2
        S_noz_11 = S_noz_22 = 0.5 * S_noz_33 + 0.25 * self.nozzle_to_cdm**2
        S_noz_12, S_noz_13, S_noz_23 = 0, 0, 0  # Due to axis symmetry
        self.nozzle_gyration_tensor = Matrix(
            [
                [S_noz_11, S_noz_12, S_noz_13],
                [S_noz_12, S_noz_22, S_noz_23],
                [S_noz_13, S_noz_23, S_noz_33],
            ]
        )
        return self.nozzle_gyration_tensor

    def evaluate_com_to_cdm_function(self):
        """Evaluates the z-coordinate of the center of mass (COM) relative to
        the center of dry mass (CDM).

        Notes
        -----
        1. The `com_to_cdm_function` plus `center_of_mass` should be equal
        to `center_of_dry_mass_position` at every time step.
        2. The `com_to_cdm_function` is a function of time and will usually
        already be discretized.

        Returns
        -------
        self.com_to_cdm_function : Function
            Function of time expressing the z-coordinate of the center of mass
            relative to the center of dry mass.
        """
        self.com_to_cdm_function = (
            -1
            * (
                (self.center_of_propellant_position - self.center_of_dry_mass_position)
                * self._csys
            )
            * self.motor.propellant_mass
            / self.total_mass
        )
        self.com_to_cdm_function.set_inputs("Time (s)")
        self.com_to_cdm_function.set_outputs("Z Coordinate COM to CDM (m)")
        self.com_to_cdm_function.set_title("Z Coordinate COM to CDM")
        return self.com_to_cdm_function

    def get_inertia_tensor_at_time(self, t):
        """Returns a Matrix representing the inertia tensor of the rocket with
        respect to the rocket's center of dry mass at a given time. It evaluates
        each inertia tensor component at the given time and returns a Matrix
        with the computed values.

        Parameters
        ----------
        t : float
            Time at which the inertia tensor is to be evaluated.

        Returns
        -------
        Matrix
            Inertia tensor of the rocket at time t.
        """
        I_11 = self.I_11.get_value_opt(t)
        I_12 = self.I_12.get_value_opt(t)
        I_13 = self.I_13.get_value_opt(t)
        I_22 = self.I_22.get_value_opt(t)
        I_23 = self.I_23.get_value_opt(t)
        I_33 = self.I_33.get_value_opt(t)
        return Matrix(
            [
                [I_11, I_12, I_13],
                [I_12, I_22, I_23],
                [I_13, I_23, I_33],
            ]
        )

    def get_inertia_tensor_derivative_at_time(self, t):
        """Returns a Matrix representing the time derivative of the inertia
        tensor of the rocket with respect to the rocket's center of dry mass at
        a given time. It evaluates each inertia tensor component's derivative at
        the given time and returns a Matrix with the computed values.

        Parameters
        ----------
        t : float
            Time at which the inertia tensor derivative is to be evaluated.

        Returns
        -------
        Matrix
            Inertia tensor time derivative of the rocket at time t.
        """
        I_11_dot = self.I_11.differentiate_complex_step(t)
        I_12_dot = self.I_12.differentiate_complex_step(t)
        I_13_dot = self.I_13.differentiate_complex_step(t)
        I_22_dot = self.I_22.differentiate_complex_step(t)
        I_23_dot = self.I_23.differentiate_complex_step(t)
        I_33_dot = self.I_33.differentiate_complex_step(t)
        return Matrix(
            [
                [I_11_dot, I_12_dot, I_13_dot],
                [I_12_dot, I_22_dot, I_23_dot],
                [I_13_dot, I_23_dot, I_33_dot],
            ]
        )

    def add_motor(self, motor, position):  # pylint: disable=too-many-statements
        """Adds a motor to the rocket.

        Parameters
        ----------
        motor : Motor, SolidMotor, HybridMotor, LiquidMotor, GenericMotor
            Motor to be added to the rocket.
        position : int, float
            Position, in meters, of the motor's coordinate system origin
            relative to the user defined rocket coordinate system.

        See Also
        --------
        :ref:`addsurface`

        Returns
        -------
        None
        """
        if hasattr(self, "motor"):
            # pylint: disable=access-member-before-definition
            if not isinstance(self.motor, EmptyMotor):
                print(
                    "Only one motor per rocket is currently supported. "
                    + "Overwriting previous motor."
                )
        self.motor = motor
        self.motor_position = position
        _ = self._csys * self.motor._csys
        self.center_of_propellant_position = (
            self.motor.center_of_propellant_mass * _ + self.motor_position
        )
        self.motor_center_of_mass_position = (
            self.motor.center_of_mass * _ + self.motor_position
        )
        self.motor_center_of_dry_mass_position = (
            self.motor.center_of_dry_mass_position * _ + self.motor_position
        )
        self.nozzle_position = self.motor.nozzle_position * _ + self.motor_position
        self.total_mass_flow_rate = self.motor.total_mass_flow_rate
        self.evaluate_dry_mass()
        self.evaluate_structural_mass_ratio()
        self.evaluate_total_mass()
        self.evaluate_center_of_dry_mass()
        self.evaluate_nozzle_to_cdm()
        self.evaluate_center_of_mass()
        self.evaluate_dry_inertias()
        self.evaluate_inertias()
        self.evaluate_reduced_mass()
        self.evaluate_thrust_to_weight()
        self.evaluate_surfaces_cp_to_cdm()
        # The motor changes the CM (and the margins)
        self._margin_outdated = True
        self.evaluate_com_to_cdm_function()
        self.evaluate_nozzle_gyration_tensor()

    def __add_single_surface(self, surface, position):
        """Adds a single aerodynamic surface to the rocket. Makes checks for
        rail buttons case, and position type.
        """
        if isinstance(surface, (TrapezoidalFin, EllipticalFin, FreeFormFin)):
            # TODO: the leading edge position should be recomputed whenever cant
            # angle of the fin changes, but currently it is only computed at the
            # moment the fin is added to the rocket. Detecting when the cant
            # angle changes is hard, because it is a parameter of the fin, while
            # the leading edge position is only defined on the rocket
            position = surface._compute_leading_edge_position(position, self._csys)
        else:
            position = (
                Vector([0, 0, position])
                if not isinstance(position, (Vector, tuple, list))
                else Vector(position)
            )

        if isinstance(surface, RailButtons):
            self.rail_buttons = Components()
            self.rail_buttons.add(surface, position)
        else:
            self.aerodynamic_surfaces.add(surface, position)
        self.__evaluate_single_surface_cp_to_cdm(surface, position)

    def add_surfaces(self, surfaces, positions):
        """Adds one or more aerodynamic surfaces to the rocket. The aerodynamic
        surface must be an instance of a class that inherits from the
        AeroSurface (e.g. NoseCone, TrapezoidalFins, etc.)

        Parameters
        ----------
        surfaces : list[AeroSurface], AeroSurface
            Aerodynamic surface to be added to the rocket. Can be a list of
            AeroSurface if more than one surface is to be added.
        positions : int, float, tuple, list, Vector
            Position(s) of the aerodynamic surface's reference point. Can be:

            - a single number (int or float) giving the z-coordinate along
              the rocket axis.
            - a sequence of three numbers (x, y, z) representing the full
              position in the user-defined coordinate system.

            If passing multiple surfaces, provide a list of positions matching
            each surface in order.
            For NoseCone type, position is the tip coordinate along the axis.
            For Fins type, position refers to the z-coordinate of the root
            chord leading-edge point closest to the nose cone, before any
            cant-angle offset is considered.
            For Tail type, position is relative to the point belonging to the
            tail which is highest in the rocket coordinate system.
            For RailButtons type, position is relative to the lower rail button.

        See Also
        --------
        :ref:`addsurface`

        Returns
        -------
        None
        """
        if self._aerodynamics_overwritten:
            warnings.warn(
                "This rocket's aerodynamics were overwritten by a full-body "
                "model (add_full_body_aerodynamics(overwrite=True)); the surface(s) "
                "you are adding now will be summed on top of that model.",
                UserWarning,
                stacklevel=2,
            )
        if isinstance(surfaces, Iterable):
            if isinstance(positions, Iterable):
                if len(surfaces) != len(positions):
                    raise ValueError(
                        "The number of surfaces and positions must be the same."
                    )
            else:
                positions = [positions] * len(surfaces)

            for surface, position in zip(surfaces, positions):
                self.__add_single_surface(surface, position)
        else:
            self.__add_single_surface(surfaces, positions)

        # Adding a surface changes both the aerodynamic center and the margins
        self._cp_outdated = True
        self._margin_outdated = True
        # Re-arm the non-axisymmetry advisory: the warning is latched once per
        # configuration (see evaluate_center_of_pressure), so a new surface may
        # legitimately warn again about the new configuration.
        self._axisymmetry_warned = False

    def add_full_body_aerodynamics(self, surfaces, position=None, overwrite=False):
        """Add a prebuilt full-body aerodynamic surface: the whole rocket
        modeled as a single surface. Instead of (or in addition to) modeling
        each component, this lets you provide a set of coefficients for the
        whole rocket, which is often easier.

        Parameters
        ----------
        surfaces : GenericSurface or list of GenericSurface
            The prebuilt full-body surface, or a list of them (for example a
            power-on/power-off pair, each carrying its own ``active_during``).
            Any of:

            - a :class:`GenericSurface`;
            - a :class:`LinearGenericSurface`;
            - a :class:`ControllableGenericSurface` for coefficients that
              also depend on control-deflection axes.

            Reference the surface's coefficients to the rocket cross-section
            area and diameter (build it with ``reference_area=rocket.area`` and
            ``reference_length=2 * rocket.radius``) so it sums consistently with
            the rest of the rocket. Because it is just another aerodynamic
            surface, a full-body model can be **mixed** with modeled add-on
            surfaces (e.g. use ``add_full_body_aerodynamics`` together with
            ``add_tail``): they simply add.

            A rocket's aerodynamics usually differ between powered and coasting
            flight. To capture this, build two surfaces, set each one's
            ``active_during`` to ``"power_on"`` and ``"power_off"``, and pass
            them together as a list; each then produces force only during its
            phase.
        position : int, float, optional
            Position along the rocket's center axis (in the user coordinate
            system) where the surface's resultant force is applied and about
            which its moment coefficients are taken. Defaults to the center of
            dry mass position.
        overwrite : bool, optional
            If ``True``, make this the rocket's only aerodynamics: every
            aerodynamic surface already on the rocket is removed first, and both
            built-in drag curves (``power_on_drag`` and ``power_off_drag``) are
            cleared. Default ``False`` (the model is added on top of the
            existing aerodynamics).

        Returns
        -------
        GenericSurface or list of GenericSurface
            The surface(s) added.
        """
        if position is None:
            position = self.center_of_dry_mass_position

        if overwrite:
            self._clear_aerodynamic_surfaces()

        surface_list = (
            list(surfaces) if isinstance(surfaces, (list, tuple)) else [surfaces]
        )
        for surface in surface_list:
            self.add_surfaces(surface, position)

        if overwrite:
            # Re-arm the "added after" guard now that the full-body model is set.
            self._aerodynamics_overwritten = True

        return surfaces

    def _clear_aerodynamic_surfaces(self):
        """Wipe the rocket's aerodynamics so a full-body model can fully replace
        them: remove every aerodynamic surface, clear both built-in drag curves,
        and reset the derived stability caches. Used by
        :meth:`add_full_body_aerodynamics` with ``overwrite=True``.
        """
        self.aerodynamic_surfaces.clear()
        self.surfaces_cp_to_cdm.clear()
        # Clear both built-in drag curves; the supplied surface(s) now provide
        # the complete aerodynamics, including any drag they carry.
        zero_drag(self, "power_on")
        zero_drag(self, "power_off")
        warnings.warn(
            "add_full_body_aerodynamics(overwrite=True): the rocket's existing "
            "aerodynamic surfaces and both built-in drag curves (power_on_drag, "
            "power_off_drag) were cleared; the supplied surface(s) now provide "
            "the complete aerodynamics, including any drag they carry.",
            UserWarning,
            stacklevel=3,
        )
        self._cp_outdated = True
        self._margin_outdated = True
        self._axisymmetry_warned = False
        # New adds are welcome again; the guard is re-armed once the full-body
        # surfaces are in place (see add_full_body_aerodynamics).
        self._aerodynamics_overwritten = False

    def to_coefficients(self, machs=None, force_convention="body"):
        """Return the whole rocket's aerodynamic coefficients, split by motor
        phase.

        Sweeps the rocket's aerodynamic surfaces and lumps them into the
        rocket's complete stability-derivative set about the dry center of mass:
        the normal-force and pitch-moment slopes ``cN_alpha``/``cm_alpha``
        (pitch), the side-force and yaw-moment slopes ``cY_beta``/``cn_beta``
        (yaw), the pitch and yaw rate damping ``cN_q``/``cm_q`` and
        ``cY_r``/``cn_r``, the fin roll damping ``cl_p`` and the drag ``cA_0``.

        The result is returned as two coefficient sets, ``"power_off"``
        (coasting) and ``"power_on"`` (motor burning).

        Important
        ---------
        The resulting coefficients are a **linear summary tabulated only against
        Mach**: the derivatives are taken at zero angle of attack, zero sideslip
        and zero rates, so only their Mach dependence is kept. This leaves out:

        - **Incidence and rate nonlinearity.** Only the slope at zero is
          retained, so any curvature in angle of attack, sideslip or the body
          rates is not represented.
        - **Reynolds dependence.** The derivatives are measured in the
          vanishing-Reynolds limit, so a coefficient that varies with Reynolds
          number is frozen at that value rather than following the flight
          Reynolds number.
        - **Control-surface dependence.** Deflection axes of a
          :class:`ControllableGenericSurface` are not carried into the summary.
        - **Induced drag.** The axial coefficient is constant in incidence, so
          drag does not increase with angle of attack.

        These are exactly the assumptions of RocketPy's built-in Barrowman
        surfaces (:class:`NoseCone`, :class:`Tail` and the fin sets), which are
        already linear, Mach-tabulated and Reynolds-independent. A rocket built
        only from them is therefore reproduced exactly, with nothing lost. The
        limitations matter only when you have added a :class:`GenericSurface` or
        :class:`ControllableGenericSurface` (or a Reynolds-dependent
        :class:`LinearGenericSurface`) whose coefficients truly vary with
        incidence beyond a straight line, with Reynolds number, or with a
        control deflection.

        Parameters
        ----------
        machs : sequence of float, optional
            Mach numbers at which the derivatives are sampled and tabulated.
            Defaults to ``0`` to ``3`` in steps of ``0.02``.
        force_convention : str, optional
            The frame the force coefficients are named in. ``"body"`` (default)
            gives the body-frame set : normal ``cN_*``, side ``cY_*`` and axial
            ``cA_0`` (drag). ``"wind"`` gives the wind-frame set: lift
            ``cL_*``, side ``cQ_*`` and drag ``cD_0``. The moment derivatives
            (``cm_*``, ``cn_*``, ``cl_p``) are the same in both.

        Returns
        -------
        dict
            A dict with keys ``"power_off"`` and ``"power_on"``. Each value is
            itself a dict mapping a coefficient name to its curve over Mach (a
            :class:`rocketpy.Function`). The body-frame set is ``cN_alpha``,
            ``cm_alpha``, ``cN_q``, ``cm_q``, ``cY_beta``, ``cn_beta``,
            ``cY_r``, ``cn_r``, ``cl_p`` and ``cA_0``.
        """
        return full_body_coefficients(self, machs, force_convention)

    def to_surface(
        self,
        machs=None,
        force_convention="body",
        name="Full Body Aerodynamics",
    ):
        """Collapse the whole assembled rocket aerodynamics into two
        :class:`rocketpy.LinearGenericSurface` objects, one for coasting and one
        for powered flight. It reproduces the source rocket's aerodynamics, so a
        bare rocket carrying the same body and motor plus this pair flies the
        same as the fully modeled rocket.

        Important
        ---------
        The resulting coefficients are a **linear summary tabulated only against
        Mach**: the derivatives are taken at zero angle of attack, zero sideslip
        and zero rates, so only their Mach dependence is kept. This leaves out:

        - **Incidence and rate nonlinearity.** Only the slope at zero is
          retained, so any curvature in angle of attack, sideslip or the body
          rates is not represented.
        - **Reynolds dependence.** The derivatives are measured in the
          vanishing-Reynolds limit, so a coefficient that varies with Reynolds
          number is frozen at that value rather than following the flight
          Reynolds number.
        - **Control-surface dependence.** Deflection axes of a
          :class:`ControllableGenericSurface` are not carried into the summary.
        - **Induced drag.** The axial coefficient is constant in incidence, so
          drag does not increase with angle of attack.

        These are exactly the assumptions of RocketPy's built-in Barrowman
        surfaces (:class:`NoseCone`, :class:`Tail` and the fin sets), which are
        already linear, Mach-tabulated and Reynolds-independent. A rocket built
        only from them is therefore reproduced exactly, with nothing lost. The
        limitations matter only when you have added a :class:`GenericSurface` or
        :class:`ControllableGenericSurface` (or a Reynolds-dependent
        :class:`LinearGenericSurface`) whose coefficients truly vary with
        incidence beyond a straight line, with Reynolds number, or with a
        control deflection.

        Parameters
        ----------
        machs : sequence of float, optional
            Mach numbers at which the derivatives are sampled and tabulated.
            Defaults to ``0`` to ``3`` in steps of ``0.02``.
        force_convention : str, optional
            The frame the force coefficients are expressed in. ``"body"``
            (default) gives the body-frame set: normal ``cN``, side ``cY`` and
            axial ``cA`` (drag). ``"wind"`` gives the wind-frame set: lift
            ``cL``, side ``cQ`` and drag ``cD``. The moment coefficients are the
            same in both.
        name : str, optional
            Base name of the returned surfaces.
            Default ``"Full Body Aerodynamics"``.

        Returns
        -------
        list of rocketpy.LinearGenericSurface
            Two surfaces, ``[power_off, power_on]``, each carrying the whole
            rocket's coefficient derivatives referenced to the rocket
            cross-section area and diameter and taken about the center of dry
            mass, and gated to its motor phase.
        """
        coefficients = self.to_coefficients(
            machs=machs,
            force_convention=force_convention,
        )
        return [
            LinearGenericSurface(
                reference_area=self.area,
                reference_length=2 * self.radius,
                coefficients=coefficients[phase],
                force_convention=force_convention,
                name=f"{name} ({phase.replace('_', ' ')})",
                active_during=phase,
            )
            for phase in ("power_off", "power_on")
        ]

    def _add_controllers(self, controllers):
        """Adds a controller to the rocket.

        Parameters
        ----------
        controllers : list of Controller objects
            List of controllers to be added to the rocket. If a single
            Controller object is passed, outside of a list, a try/except block
            will be used to try to append the controller to the list.

        Returns
        -------
        None
        """
        try:
            self._controllers.extend(controllers)
        except TypeError:
            self._controllers.append(controllers)

    def add_tail(
        self, top_radius, bottom_radius, length, position, radius=None, name="Tail"
    ):
        """Create a new tail or rocket diameter change, storing its
        parameters as part of the aerodynamic_surfaces list. Its
        parameters are the axial position along the rocket and its
        derivative of the coefficient of lift in respect to angle of
        attack.

        Parameters
        ----------
        top_radius : int, float
            Tail top radius in meters, considering positive direction
            from center of mass to nose cone.
        bottom_radius : int, float
            Tail bottom radius in meters, considering positive direction
            from center of mass to nose cone.
        length : int, float
            Tail length or height in meters. Must be a positive value.
        position : int, float
            Tail position relative to the rocket's coordinate system.
            By tail position, understand the point belonging to the tail which
            is highest in the rocket coordinate system (i.e. the point
            closest to the nose cone).
        radius : int, float, optional
            Reference radius of the tail. This is used to calculate lift
            coefficient. If None, which is default, the rocket radius will
            be used.
        name : string
            Tail name. Default is "Tail".

        See Also
        --------
        :ref:`addsurface`

        Returns
        -------
        tail : Tail
            Tail object created.
        """
        # Modify reference radius if not provided
        radius = self.radius if radius is None else radius
        # Create tail, adds it to the rocket and returns it
        tail = Tail(top_radius, bottom_radius, length, radius, name)
        self.add_surfaces(tail, position)
        return tail

    def add_body_tube(
        self, length, position, radius=None, rocket_radius=None, name="Body Tube"
    ):
        """Create a new constant-radius body tube, storing it as part of the
        aerodynamic_surfaces list. The tube produces no slender-body normal
        force; its in-flight lift comes entirely from the nonlinear Galejs
        body-lift term applied at the planform centroid.

        Parameters
        ----------
        length : int, float
            Body tube length in meters. Must be a positive value.
        position : int, float
            Tube position relative to the rocket's coordinate system. By tube
            position, understand the point belonging to the tube which is
            highest in the rocket coordinate system (i.e. the point closest to
            the nose cone).
        radius : int, float, optional
            Body tube outer radius in meters. If None, which is default, the
            rocket radius will be used.
        rocket_radius : int, float, optional
            Reference radius used for lift coefficient normalization. If None,
            which is default, the rocket radius will be used.
        name : string
            Body tube name. Default is "Body Tube".

        See Also
        --------
        :ref:`addsurface`

        Returns
        -------
        body_tube : BodyTube
            BodyTube object created.
        """
        radius = self.radius if radius is None else radius
        rocket_radius = self.radius if rocket_radius is None else rocket_radius
        body_tube = BodyTube(length, radius, rocket_radius, name)
        self.add_surfaces(body_tube, position)
        return body_tube

    def add_nose(
        self,
        length,
        kind,
        position,
        bluffness=0,
        power=None,
        name="Nose Cone",
        base_radius=None,
    ):
        """Creates a nose cone, storing its parameters as part of the
        aerodynamic_surfaces list. Its parameters are the axial position
        along the rocket and its derivative of the coefficient of lift
        in respect to angle of attack.

        Parameters
        ----------
        length : int, float
            Nose cone length or height in meters. Must be a positive
            value.
        kind : string
            Nose cone type. Von Karman, conical, ogive, lvhaack and
            powerseries are supported.
        position : int, float
            Nose cone tip coordinate relative to the rocket's coordinate system.
            See `Rocket.coordinate_system_orientation` for more information.
        bluffness : float, optional
            Ratio between the radius of the circle on the tip of the ogive and
            the radius of the base of the ogive.
        power : float, optional
            Factor that controls the bluntness of the nose cone shape when
            using a 'powerseries' nose cone kind.
        name : string
            Nose cone name. Default is "Nose Cone".
        base_radius : int, float, optional
            Nose cone base radius in meters. If not given, the rocket radius
            will be used.

        See Also
        --------
        :ref:`addsurface`

        Returns
        -------
        nose : Nose
            Nose cone object created.
        """
        nose = NoseCone(
            length=length,
            kind=kind,
            base_radius=base_radius or self.radius,
            rocket_radius=base_radius or self.radius,
            bluffness=bluffness,
            power=power,
            name=name,
        )
        self.add_surfaces(nose, position)
        return nose

    @deprecated(
        reason="This method is set to be deprecated in version 1.0.0 and fully "
        "removed by version 1.14.0",
        alternative="Rocket.add_trapezoidal_fins",
    )
    def add_fins(self, *args, **kwargs):  # pragma: no cover
        """See Rocket.add_trapezoidal_fins for documentation.
        This method is set to be deprecated in version 1.0.0 and fully removed
        by version 2.0.0. Use Rocket.add_trapezoidal_fins instead. It keeps the
        same arguments and signature."""
        return self.add_trapezoidal_fins(*args, **kwargs)

    def add_trapezoidal_fins(
        self,
        n,
        root_chord,
        tip_chord,
        span,
        position,
        cant_angle=0.0,
        sweep_length=None,
        sweep_angle=None,
        radius=None,
        airfoil=None,
        name="Fins",
    ):
        """Create a trapezoidal fin set, storing its parameters as part of the
        aerodynamic_surfaces list. Its parameters are the axial position along
        the rocket and its derivative of the coefficient of lift in respect to
        angle of attack.

        Parameters
        ----------
        n : int
            Number of fins, must be greater than 2.
        span : int, float
            Fin span in meters.
        root_chord : int, float
            Fin root chord in meters.
        tip_chord : int, float
            Fin tip chord in meters.
        position : int, float
            Fin set position in the z coordinate of the user defined rocket
            coordinate system. By fin set position, understand the point
            belonging to the root chord which is highest in the rocket
            coordinate system (i.e. the point closest to the nose cone tip).

            See Also
            --------
            :ref:`positions`
        cant_angle : int, float, optional
            Fins cant angle with respect to the rocket centerline. Must
            be given in degrees.
        sweep_length : int, float, optional
            Fins sweep length in meters. By sweep length, understand the axial
            distance between the fin root leading edge and the fin tip leading
            edge measured parallel to the rocket centerline. If not given, the
            sweep length is assumed to be equal the root chord minus the tip
            chord, in which case the fin is a right trapezoid with its base
            perpendicular to the rocket's axis. Cannot be used in conjunction
            with sweep_angle.
        sweep_angle : int, float, optional
            Fins sweep angle with respect to the rocket centerline. Must be
            given in degrees. If not given, the sweep angle is automatically
            calculated, in which case the fin is assumed to be a right trapezoid
            with its base perpendicular to the rocket's axis. Cannot be used in
            conjunction with sweep_length.
        radius : int, float, optional
            Reference fuselage radius where the fins are located. This is used
            to calculate lift coefficient and to draw the rocket. If None,
            which is default, the rocket radius will be used.
        airfoil : tuple, optional
            Default is null, in which case fins will be treated as flat plates.
            Otherwise, if tuple, fins will be considered as airfoils. The
            tuple's first item specifies the airfoil's lift coefficient
            by angle of attack and must be either a .csv, .txt, ndarray
            or callable. The .csv and .txt files can contain a single line
            header and the first column must specify the angle of attack, while
            the second column must specify the lift coefficient. The
            ndarray should be as [(x0, y0), (x1, y1), (x2, y2), ...]
            where x0 is the angle of attack and y0 is the lift coefficient.
            If callable, it should take an angle of attack as input and
            return the lift coefficient at that angle of attack.
            The tuple's second item is the unit of the angle of attack,
            accepting either "radians" or "degrees".

        Returns
        -------
        fin_set : TrapezoidalFins
            Fin set object created.
        """
        if n <= 2:
            raise ValueError(
                "Number of fins must be greater than 2. "
                "For 1 or 2 fins, create a FreeFormFin object "
                "and add it to the rocket using the add_surfaces method."
            )

        # Modify radius if not given, use rocket radius, otherwise use given.
        radius = radius if radius is not None else self.radius

        # Create a fin set as an object of TrapezoidalFins class
        fin_set = TrapezoidalFins(
            n,
            root_chord,
            tip_chord,
            span,
            radius,
            cant_angle,
            sweep_length,
            sweep_angle,
            airfoil,
            name,
        )

        # Add fin set to the list of aerodynamic surfaces
        self.add_surfaces(fin_set, position)
        return fin_set

    def add_elliptical_fins(
        self,
        n,
        root_chord,
        span,
        position,
        cant_angle=0,
        radius=None,
        airfoil=None,
        name="Fins",
    ):
        """Create an elliptical fin set, storing its parameters as part of the
        aerodynamic_surfaces list. Its parameters are the axial position along
        the rocket and its derivative of the coefficient of lift in respect to
        angle of attack.

        Parameters
        ----------
        n : int
            Number of fins, must be greater than 2.
        root_chord : int, float
            Fin root chord in meters.
        span : int, float
            Fin span in meters.
        position : int, float
            Fin set position in the z coordinate of the user defined rocket
            coordinate system. By fin set position, understand the point
            belonging to the root chord which is highest in the rocket
            coordinate system (i.e. the point closest to the nose cone tip).

            See Also
            --------
            :ref:`positions`
        cant_angle : int, float, optional
            Fins cant angle with respect to the rocket centerline. Must be given
            in degrees.
        radius : int, float, optional
            Reference fuselage radius where the fins are located. This is used
            to calculate lift coefficient and to draw the rocket. If None,
            which is default, the rocket radius will be used.
        airfoil : tuple, optional
            Default is null, in which case fins will be treated as flat plates.
            Otherwise, if tuple, fins will be considered as airfoils. The
            tuple's first item specifies the airfoil's lift coefficient
            by angle of attack and must be either a .csv, .txt, ndarray
            or callable. The .csv and .txt files can contain a single line
            header and the first column must specify the angle of attack, while
            the second column must specify the lift coefficient. The
            ndarray should be as [(x0, y0), (x1, y1), (x2, y2), ...]
            where x0 is the angle of attack and y0 is the lift coefficient.
            If callable, it should take an angle of attack as input and
            return the lift coefficient at that angle of attack.
            The tuple's second item is the unit of the angle of attack,
            accepting either "radians" or "degrees".

        See Also
        --------
        :ref:`addsurface`

        Returns
        -------
        fin_set : EllipticalFins
            Fin set object created.
        """
        if n <= 2:
            raise ValueError(
                "Number of fins must be greater than 2. "
                "For 1 or 2 fins, create a FreeFormFin object "
                "and add it to the rocket using the add_surfaces method."
            )

        radius = radius if radius is not None else self.radius
        fin_set = EllipticalFins(n, root_chord, span, radius, cant_angle, airfoil, name)
        self.add_surfaces(fin_set, position)
        return fin_set

    def add_free_form_fins(
        self,
        n,
        shape_points,
        position,
        cant_angle=0.0,
        radius=None,
        airfoil=None,
        name="Fins",
    ):
        """Create a free form fin set, storing its parameters as part of the
        aerodynamic_surfaces list. Its parameters are the axial position along
        the rocket and its derivative of the coefficient of lift in respect to
        angle of attack.

        Parameters
        ----------
        n : int
            Number of fins, must be greater than 2.
        shape_points : list
            List of tuples (x, y) containing the coordinates of the fin's
            geometry defining points. The point (0, 0) is the root leading edge.
            Positive x is rearwards, positive y is upwards (span direction).
            The shape will be interpolated between the points, in the order
            they are given. The last point connects to the first point.
        position : int, float
            Fin set position in the z coordinate of the user defined rocket
            coordinate system. By fin set position, understand the point
            belonging to the root chord which is highest in the rocket
            coordinate system (i.e. the point closest to the nose cone tip).

            See Also
            --------
            :ref:`positions`
        cant_angle : int, float, optional
            Fins cant angle with respect to the rocket centerline. Must
            be given in degrees.
        radius : int, float, optional
            Reference fuselage radius where the fins are located. This is used
            to calculate lift coefficient and to draw the rocket. If None,
            which is default, the rocket radius will be used.
        airfoil : tuple, optional
            Default is null, in which case fins will be treated as flat plates.
            Otherwise, if tuple, fins will be considered as airfoils. The
            tuple's first item specifies the airfoil's lift coefficient
            by angle of attack and must be either a .csv, .txt, ndarray
            or callable. The .csv and .txt files can contain a single line
            header and the first column must specify the angle of attack, while
            the second column must specify the lift coefficient. The
            ndarray should be as [(x0, y0), (x1, y1), (x2, y2), ...]
            where x0 is the angle of attack and y0 is the lift coefficient.
            If callable, it should take an angle of attack as input and
            return the lift coefficient at that angle of attack.
            The tuple's second item is the unit of the angle of attack,
            accepting either "radians" or "degrees".

        Returns
        -------
        fin_set : FreeFormFins
            Fin set object created.
        """
        if n <= 2:
            raise ValueError(
                "Number of fins must be greater than 2. "
                "For 1 or 2 fins, create a FreeFormFin object "
                "and add it to the rocket using the add_surfaces method."
            )

        # Modify radius if not given, use rocket radius, otherwise use given.
        radius = radius if radius is not None else self.radius

        fin_set = FreeFormFins(
            n,
            shape_points,
            radius,
            cant_angle,
            airfoil,
            name,
        )

        # Add fin set to the list of aerodynamic surfaces
        self.add_surfaces(fin_set, position)
        return fin_set

    def add_parachute(
        self,
        name,
        cd_s,
        trigger,
        sampling_rate=100,
        lag=0,
        noise=(0, 0, 0),
        radius=None,
        height=None,
        porosity=0.0432,
        drag_coefficient=1.4,
        trigger_needs=None,
    ):
        """Creates a new parachute, storing its parameters such as
        opening delay, drag coefficients and trigger function.

        Parameters
        ----------
        name : string
            Parachute name, such as drogue and main. Has no impact in
            simulation, as it is only used to display data in a more
            organized matter.
        cd_s : float
            Drag coefficient times reference area for parachute. It is
            used to compute the drag force exerted on the parachute by
            the equation F = ((1/2)*rho*V^2)*cd_s, that is, the drag
            force is the dynamic pressure computed on the parachute
            times its cd_s coefficient. Has units of area and must be
            given in squared meters.
        trigger : callable, float, str
            Defines the trigger condition for the parachute ejection system. It
            can be one of the following:

            - A callable function that takes three arguments: \

                1. Freestream pressure in pascals.
                2. Height in meters above ground level.
                3. The state vector of the simulation, which is defined as: \

                    .. code-block:: python

                        u = [x, y, z, vx, vy, vz, e0, e1, e2, e3, wx, wy, wz]

                .. note::

                    The function should return ``True`` if the parachute \
                    ejection system should be triggered and ``False`` otherwise.
            - A float value, representing an absolute height in meters. In this \
                case, the parachute will be ejected when the rocket reaches this \
                height above ground level.
            - The string "apogee" which triggers the parachute at apogee, i.e., \
                when the rocket reaches its highest point and starts descending.

            .. note::

                The function will be called according to the sampling rate specified.
        sampling_rate : float, optional
            Sampling rate in which the trigger function works. It is used to
            simulate the refresh rate of onboard sensors such as barometers.
            Default value is 100. Value must be given in hertz.
        lag : float, optional
            Time between the parachute ejection system is triggered and the
            parachute is fully opened. During this time, the simulation will
            consider the rocket as flying without a parachute. Default value
            is 0. Must be given in seconds.
        noise : tuple, list, optional
            List in the format (mean, standard deviation, time-correlation).
            The values are used to add noise to the pressure signal which is
            passed to the trigger function. Default value is (0, 0, 0). Units
            are in pascal.
        radius : float, optional
            Length of the non-unique semi-axis (radius) of the inflated
            hemispheroid parachute. If not provided, it is estimated from
            `cd_s` and `drag_coefficient` using:
            `radius = sqrt(cd_s / (drag_coefficient * pi))`.
            Units are in meters.
        height : float, optional
            Length of the unique semi-axis (height) of the inflated hemispheroid
            parachute. Default value is the radius of the parachute.
            Units are in meters.
        porosity : float, optional
            Geometric porosity of the canopy (ratio of open area to total
            canopy area), in [0, 1]. Affects only the added-mass scaling
            during descent; it does not change `cd_s` (drag). The default
            value of 0.0432 yields an `added_mass_coefficient` of
            approximately 1.0 ("neutral" added-mass behavior).
        drag_coefficient : float, optional
            Drag coefficient of the inflated canopy shape, used only when
            `radius` is not provided. Typical values: 1.4 for hemispherical
            canopies (default), 0.75 for flat circular canopies, 1.5 for
            extended-skirt canopies. Has no effect when `radius` is given.
        trigger_needs : list or frozenset of str or None, optional
            Declares which expensive simulation values the trigger function
            accesses. Valid keys: ``'state_dot'``, ``'pressure'``,
            ``'state_history'``. When ``None`` (default), built-in trigger
            types (``'apogee'`` string, numeric height) have their needs set
            automatically. For callable triggers no needs are assumed; pass
            an explicit list if your trigger accesses any of the keys above.

        Returns
        -------
        parachute : Parachute
            Parachute containing trigger, sampling_rate, lag, cd_s, noise,
            radius, drag_coefficient, height, porosity and name. Furthermore,
            it stores clean_pressure_signal, noise_signal and
            noisyPressureSignal which are filled in during Flight simulation.
        """
        parachute = Parachute(
            name,
            cd_s,
            trigger,
            sampling_rate,
            lag,
            noise,
            radius,
            height,
            porosity,
            drag_coefficient,
            trigger_needs,
        )
        self.parachutes.append(parachute)
        return self.parachutes[-1]

    def add_sensor(self, sensor, position):
        """Adds a sensor to the rocket.

        Parameters
        ----------
        sensor : Sensor
            Sensor to be added to the rocket.
        position : int, float, tuple, list, Vector
            Position of the sensor. If a Vector, tuple or list is passed, it
            must be in the format (x, y, z) where x, y, and z are defined in the
            rocket's user defined coordinate system. If a single value is
            passed, it is assumed to be along the z-axis (centerline) of the
            rocket's user defined coordinate system.

        Returns
        -------
        None
        """
        if isinstance(position, (float, int)):
            position = (0, 0, position)
        position = Vector(position)
        self.sensors.add(sensor, position)

        # Update sensors_by_name property
        if sensor.name in self.sensors_by_name:
            existing = self.sensors_by_name[sensor.name]
            if isinstance(existing, list):
                existing.append(sensor)
            else:
                self.sensors_by_name[sensor.name] = [existing, sensor]
        else:
            self.sensors_by_name[sensor.name] = sensor

        # Keep track of how many times the sensor is attached to the rocket
        try:
            sensor._attached_rockets[self] += 1
        except KeyError:
            sensor._attached_rockets[self] = 1

        # Create and store a position-specific event for this sensor-position
        # This allows several objects of the same sensor type to be added to the
        # rocket in different positions.
        if not hasattr(self, "_sensor_events"):
            self._sensor_events = []

        event = sensor.to_event(position)
        self._sensor_events.append(event)

    def add_air_brakes(
        self,
        drag_coefficient_curve,
        controller_function,
        sampling_rate,
        clamp=True,
        reference_area=None,
        initial_observed_variables=None,
        context=None,
        override_rocket_drag=False,
        return_controller=False,
        name="AirBrakes",
        controller_name="AirBrakes Controller",
        controller_needs=None,
    ):
        """Creates a new air brakes system, storing its parameters such as
        drag coefficient curve, controller function, sampling rate, and
        reference area.

        Parameters
        ----------
        drag_coefficient_curve : int, float, callable, array, string, Function
            This parameter represents the drag coefficient associated with the
            air brakes and/or the entire rocket, depending on the value of
            ``override_rocket_drag``.

            - If a constant, it should be an integer or a float representing a
              fixed drag coefficient value.
            - If a function, it must take two parameters: deployment level and
              Mach number, and return the drag coefficient. This function allows
              for dynamic computation based on deployment and Mach number.
            - If an array, it should be a 2D array with three columns: the first
              column for deployment level, the second for Mach number, and the
              third for the corresponding drag coefficient.
            - If a string, it should be the path to a .csv or .txt file. The
              file must contain three columns: the first for deployment level,
              the second for Mach number, and the third for the drag
              coefficient.
            - If a Function, it must take two parameters: deployment level and
              Mach number, and return the drag coefficient.

            .. note:: For ``override_rocket_drag = False``, at
                deployment level 0, the drag coefficient is assumed to be 0,
                independent of the input drag coefficient curve. This means that
                the simulation always considers that at a deployment level of 0,
                the air brakes are completely retracted and do not contribute to
                the drag of the rocket.

          controller_function : callable
                Function that executes the control logic, with signature
                ``controller_function(**kwargs) -> dict or None``. Invoked
                once per sample; its return value is appended to the
                controller log. Set ``air_brakes.deployment_level`` to apply
                the control action.
                The following keys are always available in ``kwargs``:
                ``time`` (float, s),
                ``state`` (list ``[x, y, z, vx, vy, vz, e0, e1, e2, e3, wx, wy, wz]``),
                ``sensors`` (list of sensor objects),
                ``sensors_by_name`` (dict of sensor objects),
                ``environment`` (:class:`rocketpy.Environment`),
                ``rocket`` (:class:`rocketpy.Rocket`),
                ``flight`` (:class:`rocketpy.Flight`),
                ``phase`` (current flight phase),
                ``step_size`` (float, s),
                ``height_agl`` (float, m),
                ``event`` (:class:`Event` wrapping this controller),
                ``sampling_rate`` (float, Hz),
                ``controller`` (this :class:`_Controller` instance),
                ``controlled_objects`` (same as ``air_brakes``),
                ``air_brakes`` (:class:`AirBrakes`).
                The following keys are only injected when declared via
                ``controller_needs``:
                ``pressure`` (float, Pa),
                ``state_dot`` (list, time derivative of ``state``),
                ``state_history`` (list of past state vectors).

        sampling_rate : float
            The sampling rate of the controller function in Hertz (Hz). This
            means that the controller function will be called every
            `1/sampling_rate` seconds.
        clamp : bool, optional
            If True, the simulation will clamp the deployment level to 0 or 1 if
            the deployment level is out of bounds. If False, the simulation will
            not clamp the deployment level and will instead raise a warning if
            the deployment level is out of bounds. Default is True.
        reference_area : float, optional
            Reference area used to calculate the drag force of the air brakes
            from the drag coefficient curve. If None, which is default, use
            rocket section area. Must be given in squared meters.
        initial_observed_variables : list, optional
            A list of the initial values of the variables that the controller
            function returns. This list is used to initialize the
            `observed_variables` argument of the controller function. The
            default value is None, which initializes the list as an empty list.

            .. deprecated:: 1.13
                Passing `initial_observed_variables` directly to
                ``add_air_brakes`` is deprecated. Provide initial observed
                variables via the ``context`` parameter as
                ``context={'observed_variables': [...]}`` instead. Support
                for the positional argument will be removed in v1.13.
        override_rocket_drag : bool, optional
            If False, the air brakes drag coefficient will be added to the
            rocket's power off drag coefficient curve. If True, during the
            simulation, the rocket's power off drag will be ignored and the air
            brakes drag coefficient will be used for the entire rocket instead.
            Default is False.
        return_controller : bool, optional
            If True, the function will return the controller object created.
            Default is False.
        name : string, optional
            AirBrakes name, such as drogue and main. Has no impact in
            simulation, as it is only used to display data in a more
            organized matter.
        controller_name : string, optional
            Controller name. Has no impact in simulation, as it is only used to
            display data in a more organized matter.
        controller_needs : list or frozenset of str or None, optional
            Declares which expensive simulation values the controller function
            accesses. Valid keys:
            ``'state_dot'``, ``'pressure'``, ``'state_history'``.
            ``None`` (default) assumes no needs; pass an explicit list if your
            controller accesses any of the keys above.

        Returns
        -------
        air_brakes : AirBrakes
            AirBrakes object created.
        controller : Controller
            Controller object created.
        """
        reference_area = reference_area if reference_area is not None else self.area
        air_brakes = AirBrakes(
            drag_coefficient_curve=drag_coefficient_curve,
            reference_area=reference_area,
            clamp=clamp,
            override_rocket_drag=override_rocket_drag,
            deployment_level=0,
            name=name,
        )
        # Prepare controller context and compatibility wrapper for
        # controller_function. New recommended signature is
        # `controller_function(**kwargs)`. To avoid breaking existing user
        # code, wrap legacy functions that accept positional args.
        # Normalize context dict
        controller_context = context.copy() if context is not None else {}

        # Map initial_observed_variables into controller context for the
        # new API while emitting a deprecation warning for the positional
        # argument usage.
        if initial_observed_variables is not None:
            warnings.warn(
                "Passing `initial_observed_variables` to `add_air_brakes` is "
                "deprecated; supply them via `context={'observed_variables': ...}` "
                "instead. Support for this argument will be removed in v1.13.",
                DeprecationWarning,
            )
            controller_context["observed_variables"] = initial_observed_variables

        orig_controller = controller_function
        signature = inspect.signature(orig_controller)
        parameters = tuple(signature.parameters.values())
        accepts_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters
        )
        accepts_var_args = any(
            p.kind == inspect.Parameter.VAR_POSITIONAL for p in parameters
        )
        positional_parameter_count = sum(
            p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            for p in parameters
        )

        if not accepts_var_kwargs and positional_parameter_count > 0:
            # A legacy positional controller must accept one of the supported
            # signatures (6, 7 or 8 arguments). Reject any other count early so
            # the user gets a clear error instead of a runtime failure mid-flight.
            if not accepts_var_args and positional_parameter_count not in (6, 7, 8):
                raise ValueError(
                    "A positional controller_function must have 6, 7, or 8 "
                    f"arguments, but {positional_parameter_count} were given. "
                    "Alternatively, define the controller function to accept "
                    "`**kwargs` only."
                )
            warnings.warn(
                "Calling controller_function with positional arguments is "
                "deprecated; use controller_function(**kwargs) instead. "
                "Support for positional controller arguments will be removed "
                "in v1.13.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Legacy positional controllers historically always received
        # ``state_history`` as their 4th argument. The event system only
        # computes it when declared in ``needs``, so request it here when the
        # legacy signature will actually consume it.
        if not accepts_var_kwargs and (
            accepts_var_args or positional_parameter_count >= 4
        ):
            needs = set(controller_needs) if controller_needs else set()
            needs.add("state_history")
            controller_needs = frozenset(needs)

        def controller_wrapper(**kwargs):
            if accepts_var_kwargs:
                return orig_controller(**kwargs)

            # Legacy positional signature expected. Build positional args in
            # the historical order described in docs. Provide sensible fallbacks
            # from kwargs when available.
            time = kwargs.get("time")
            sampling = sampling_rate
            state = kwargs.get("state")
            state_history = kwargs.get("state_history")
            observed_variables = controller_context.get("observed_variables", [])
            # Prefer the controller's live ``controlled_objects`` (set by the
            # callback) over the captured ``air_brakes``: after a flight is
            # loaded from a file the two differ, and only the former is the air
            # brakes the rocket actually uses for drag. For a normal flight they
            # are the same object, so this is unchanged behavior.
            interactive_objects = kwargs.get(
                "interactive_objects", kwargs.get("controlled_objects", air_brakes)
            )
            sensors = kwargs.get("sensors")
            environment = kwargs.get("environment")

            pos_args = [
                time,
                sampling,
                state,
                state_history,
                observed_variables,
                interactive_objects,
                sensors,
                environment,
            ]

            if accepts_var_args:
                legacy_args = pos_args
            else:
                legacy_args = pos_args[:positional_parameter_count]

            return orig_controller(*legacy_args)

        # TODO: should this be in the airbrakes object instead?
        _controller = _Controller(
            controller_function=controller_wrapper,
            controlled_objects=air_brakes,
            controlled_objects_name="air_brakes",
            sampling_rate=sampling_rate,
            context=controller_context,
            name=controller_name,
            controller_needs=controller_needs,
        )
        self.air_brakes.append(air_brakes)
        self._add_controllers(_controller)
        if return_controller:
            return air_brakes, _controller
        else:
            return air_brakes

    def set_rail_buttons(
        self,
        upper_button_position,
        lower_button_position,
        angular_position=45,
        radius=None,
    ):
        """Adds rail buttons to the rocket, allowing for the calculation of
        forces exerted by them when the rocket is sliding in the launch rail.
        For the simulation, only two buttons are needed, which are the two
        closest to the nozzle.

        Parameters
        ----------
        upper_button_position : int, float
            Position of the rail button furthest from the nozzle relative to
            the rocket's coordinate system, in meters.
            See :doc:`Positions and Coordinate Systems </user/positions>`
            for more information.
        lower_button_position : int, float
            Position of the rail button closest to the nozzle relative to
            the rocket's coordinate system, in meters.
            See :doc:`Positions and Coordinate Systems </user/positions>`
            for more information.
        angular_position : float, optional
            Angular position of the rail buttons in degrees measured
            as the rotation around the symmetry axis of the rocket
            relative to one of the other principal axis.
            Default value is 45 degrees, generally used in rockets with
            4 fins. See :ref:`Angular Position Inputs <angular_position>`
        radius : int, float, optional
            Fuselage radius where the rail buttons are located.

        See Also
        --------
        :ref:`addsurface`

        Returns
        -------
        rail_buttons : RailButtons
            RailButtons object created
        """
        radius = radius or self.radius
        buttons_distance = abs(upper_button_position - lower_button_position)
        rail_buttons = RailButtons(
            buttons_distance=buttons_distance,
            angular_position=angular_position,
            rocket_radius=radius,
        )
        self.rail_buttons = Components()
        position = Vector(
            [
                radius * -math.sin(math.radians(angular_position)),
                radius * math.cos(math.radians(angular_position)),
                lower_button_position,
            ]
        )
        self.rail_buttons.add(rail_buttons, position)
        return rail_buttons

    def add_cm_eccentricity(self, x, y):
        """Moves line of action of aerodynamic and thrust forces by
        equal translation amount to simulate an eccentricity in the
        position of the center of dry mass of the rocket relative to
        its geometrical center line.

        Parameters
        ----------
        x : float
            Distance in meters by which the CM is to be translated in
            the x direction relative to geometrical center line. The x axis
            is defined according to the body axes coordinate system.
        y : float
            Distance in meters by which the CM is to be translated in
            the y direction relative to geometrical center line. The y axis
            is defined according to the body axes coordinate system.

        Returns
        -------
        self : Rocket
            Object of the Rocket class.

        See Also
        --------
        :ref:`rocket_axes`

        Notes
        -----
        Should not be used together with add_cp_eccentricity and
        add_thrust_eccentricity.
        """
        self.cm_eccentricity_x = x
        self.cm_eccentricity_y = y
        self.add_cp_eccentricity(-x, -y)
        self.add_thrust_eccentricity(-x, -y)
        return self

    def add_cp_eccentricity(self, x, y):
        """Moves line of action of aerodynamic forces to simulate an
        eccentricity in the position of the center of pressure relative
        to the center of dry mass of the rocket.

        Parameters
        ----------
        x : float
            Distance in meters by which the CP is to be translated in
            the x direction relative to the center of dry mass axial line.
            The x axis is defined according to the body axes coordinate system.
        y : float
            Distance in meters by which the CP is to be translated in
            the y direction relative to the center of dry mass axial line.
            The y axis is defined according to the body axes coordinate system.

        Returns
        -------
        self : Rocket
            Object of the Rocket class.

        See Also
        --------
        :ref:`rocket_axes`
        """
        self.cp_eccentricity_x = x
        self.cp_eccentricity_y = y
        return self

    def add_thrust_eccentricity(self, x, y):
        """Moves line of action of thrust forces to simulate a
        misalignment of the thrust vector and the center of dry mass.

        Parameters
        ----------
        x : float
            Distance in meters by which the line of action of the
            thrust force is to be translated in the x direction
            relative to the center of dry mass axial line. The x axis
            is defined according to the body axes coordinate system.
        y : float
            Distance in meters by which the line of action of the
            thrust force is to be translated in the y direction
            relative to the center of dry mass axial line. The y axis
            is defined according to the body axes coordinate system.

        Returns
        -------
        self : Rocket
            Object of the Rocket class.

        See Also
        --------
        :ref:`rocket_axes`
        """
        self.thrust_eccentricity_x = x
        self.thrust_eccentricity_y = y
        return self

    def draw(self, vis_args=None, plane="xz", *, filename=None):
        """Draws the rocket in a matplotlib figure.

        Parameters
        ----------
        vis_args : dict, optional
            Determines the visual aspects when drawing the rocket. If None,
            default values are used. Default values are:

            .. code-block:: python

                {
                    "background": "#EEEEEE",
                    "tail": "black",
                    "nose": "black",
                    "body": "dimgrey",
                    "fins": "black",
                    "motor": "black",
                    "buttons": "black",
                    "line_width": 2.0,
                }

            A full list of color names can be found at:
            https://matplotlib.org/stable/gallery/color/named_colors
        plane : str, optional
            Plane in which the rocket will be drawn. Default is 'xz'. Other
            options is 'yz'. Used only for sensors representation.
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which case
            the plot will be shown instead of saved. Supported file endings are:
            eps, jpg, jpeg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            and webp (these are the formats supported by matplotlib).
        """
        self.plots.draw(vis_args, plane, filename=filename)

    def info(self):
        """Prints out a summary of the data and graphs available about
        the Rocket.

        Returns
        -------
        None
        """
        self.prints.all()

    def all_info(self):
        """Prints out all data and graphs available about the Rocket.

        Returns
        -------
        None
        """
        self.info()
        self.plots.all()

    # pylint: disable=too-many-statements
    def to_dict(self, **kwargs):
        discretize = kwargs.get("discretize", False)

        power_off_drag = self.power_off_drag_7d
        power_on_drag = self.power_on_drag_7d

        rocket_dict = {
            "radius": self.radius,
            "mass": self.mass,
            "I_11_without_motor": self.I_11_without_motor,
            "I_22_without_motor": self.I_22_without_motor,
            "I_33_without_motor": self.I_33_without_motor,
            "I_12_without_motor": self.I_12_without_motor,
            "I_13_without_motor": self.I_13_without_motor,
            "I_23_without_motor": self.I_23_without_motor,
            "power_off_drag": power_off_drag,
            "power_on_drag": power_on_drag,
            "center_of_mass_without_motor": self.center_of_mass_without_motor,
            "coordinate_system_orientation": self.coordinate_system_orientation,
            "motor": self.motor,
            "motor_position": self.motor_position,
            "aerodynamic_surfaces": self.aerodynamic_surfaces,
            "rail_buttons": self.rail_buttons,
            "parachutes": self.parachutes,
            "air_brakes": self.air_brakes,
            "_controllers": self._controllers,
            "sensors": self.sensors,
        }

        if kwargs.get("include_outputs", False):
            thrust_to_weight = self.thrust_to_weight
            aerodynamic_center = self.aerodynamic_center
            # The zero-incidence design surface (Mach, time), a 2-D slice of the
            # angle-of-attack-aware stability_margin, for output inspection.
            stability_margin = Function(
                lambda mach, time: self.stability_margin.get_value_opt(0.0, mach, time),
                inputs=["Mach", "Time (s)"],
                outputs="Stability Margin (c)",
            )
            center_of_mass = self.center_of_mass
            motor_center_of_mass_position = self.motor_center_of_mass_position
            reduced_mass = self.reduced_mass
            total_mass = self.total_mass
            total_mass_flow_rate = self.total_mass_flow_rate
            center_of_propellant_position = self.center_of_propellant_position

            if discretize:
                thrust_to_weight = thrust_to_weight.set_discrete_based_on_model(
                    self.motor.thrust, mutate_self=False
                )
                aerodynamic_center = aerodynamic_center.set_discrete(
                    0, 4, 25, mutate_self=False
                )
                stability_margin = stability_margin.set_discrete(
                    (0, self.motor.burn_time[0]),
                    (2, self.motor.burn_time[1]),
                    (10, 10),
                    mutate_self=False,
                )
                center_of_mass = center_of_mass.set_discrete_based_on_model(
                    self.motor.thrust, mutate_self=False
                )
                motor_center_of_mass_position = (
                    motor_center_of_mass_position.set_discrete_based_on_model(
                        self.motor.thrust, mutate_self=False
                    )
                )
                reduced_mass = reduced_mass.set_discrete_based_on_model(
                    self.motor.thrust, mutate_self=False
                )
                total_mass = total_mass.set_discrete_based_on_model(
                    self.motor.thrust, mutate_self=False
                )
                total_mass_flow_rate = total_mass_flow_rate.set_discrete_based_on_model(
                    self.motor.thrust, mutate_self=False
                )
                center_of_propellant_position = (
                    center_of_propellant_position.set_discrete_based_on_model(
                        self.motor.thrust, mutate_self=False
                    )
                )

            rocket_dict["area"] = self.area
            rocket_dict["center_of_dry_mass_position"] = (
                self.center_of_dry_mass_position
            )
            rocket_dict["center_of_mass_without_motor"] = (
                self.center_of_mass_without_motor
            )
            rocket_dict["motor_center_of_mass_position"] = motor_center_of_mass_position
            rocket_dict["motor_center_of_dry_mass_position"] = (
                self.motor_center_of_dry_mass_position
            )
            rocket_dict["center_of_mass"] = center_of_mass
            rocket_dict["reduced_mass"] = reduced_mass
            rocket_dict["total_mass"] = total_mass
            rocket_dict["total_mass_flow_rate"] = total_mass_flow_rate
            rocket_dict["thrust_to_weight"] = thrust_to_weight
            rocket_dict["cp_eccentricity_x"] = self.cp_eccentricity_x
            rocket_dict["cp_eccentricity_y"] = self.cp_eccentricity_y
            rocket_dict["thrust_eccentricity_x"] = self.thrust_eccentricity_x
            rocket_dict["thrust_eccentricity_y"] = self.thrust_eccentricity_y
            rocket_dict["aerodynamic_center"] = aerodynamic_center
            rocket_dict["stability_margin"] = stability_margin
            rocket_dict["static_margin"] = self.static_margin
            rocket_dict["nozzle_position"] = self.nozzle_position
            rocket_dict["nozzle_to_cdm"] = self.nozzle_to_cdm
            rocket_dict["nozzle_gyration_tensor"] = self.nozzle_gyration_tensor
            rocket_dict["center_of_propellant_position"] = center_of_propellant_position

        return rocket_dict

    @classmethod
    def from_dict(cls, data):
        rocket = cls(
            radius=data["radius"],
            mass=data["mass"],
            inertia=(
                data["I_11_without_motor"],
                data["I_22_without_motor"],
                data["I_33_without_motor"],
                data["I_12_without_motor"],
                data["I_13_without_motor"],
                data["I_23_without_motor"],
            ),
            power_off_drag=data["power_off_drag"],
            power_on_drag=data["power_on_drag"],
            center_of_mass_without_motor=data["center_of_mass_without_motor"],
            coordinate_system_orientation=data["coordinate_system_orientation"],
        )

        if (motor := data["motor"]) is not None:
            rocket.add_motor(
                motor=motor,
                position=data["motor_position"],
            )

        for surface, position in data["aerodynamic_surfaces"]:
            rocket.add_surfaces(surfaces=surface, positions=position)

        for button, position in data["rail_buttons"]:
            rocket.set_rail_buttons(
                upper_button_position=position[2] + button.buttons_distance,
                lower_button_position=position[2],
                angular_position=button.angular_position,
                radius=button.rocket_radius,
            )

        for parachute in data["parachutes"]:
            rocket.parachutes.append(parachute)

        for sensor, position in data["sensors"]:
            rocket.add_sensor(sensor, position)

        for air_brake in data["air_brakes"]:
            rocket.air_brakes.append(air_brake)

        for controller in data["_controllers"]:
            # Reconnect the controller to the rocket's own reconstructed objects
            # by matching the hash(es) it stored for its controlled objects
            # against the reconstructed objects (see _Controller.to_dict).
            controlled_objects_hash = getattr(
                controller, "_serialized_controlled_objects_hash", None
            )
            if controlled_objects_hash is not None:
                is_iterable = isinstance(controlled_objects_hash, Iterable)
                hashes = (
                    controlled_objects_hash
                    if is_iterable
                    else [controlled_objects_hash]
                )
                found = []
                for hash_ in hashes:
                    if hash_ is None:  # unhashable controlled object; cannot match
                        continue
                    if (hashed_obj := find_obj_from_hash(data, hash_)) is not None:
                        found.append(hashed_obj)
                    else:
                        warnings.warn(
                            "Could not find controller controlled objects. "
                            "Deserialization will proceed, results may not be accurate."
                        )
                if found:
                    controller.rebind_controlled_objects(
                        found if is_iterable else found[0]
                    )
            rocket._add_controllers(controller)

        return rocket
