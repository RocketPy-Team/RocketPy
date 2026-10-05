import inspect
import math
import warnings
from collections.abc import Iterable

import numpy as np

from rocketpy.control.controller import _Controller
from rocketpy.mathutils.function import Function
from rocketpy.mathutils.vector_matrix import Matrix, Vector
from rocketpy.motors.empty_motor import EmptyMotor
from rocketpy.plots.rocket_plots import _RocketPlots
from rocketpy.prints.rocket_prints import _RocketPrints
from rocketpy.rocket._helpers import (
    center_of_pressure_position,
    corrective_and_damping_moments,
    disturbance_response,
    full_body_coefficients,
    is_axisymmetric,
    is_incidence_linear,
    lateral_inertia_and_rate,
    lumped_control_names,
    lumped_surface_coefficients,
    moment_slopes_left_out,
    neutral_point_and_slope,
    stability_margin_and_slope,
    stability_surfaces,
    uses_rate_coefficients,
)
from rocketpy.rocket.aero_surface import (
    AirBrakes,
    ControllableGenericSurface,
    EllipticalFins,
    Fin,
    Fins,
    GenericSurface,
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
from rocketpy.rocket.aero_surface.linear_generic_surface import LinearGenericSurface
from rocketpy.rocket.components import Components, position_vector
from rocketpy.rocket.parachute import Parachute
from rocketpy.tools import (
    deprecated,
    find_obj_from_hash,
    parallel_axis_theorem_from_com,
)


# pylint: disable=too-many-instance-attributes, too-many-public-methods
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
        Position of the rocket's aerodynamic center, in meters, as a function
        of Mach number, in the user defined rocket reference system. It is the
        point the static margin is measured from. For a rocket built from nose
        cones, fins and tails it is the same point as the center of pressure.
        See :doc:`Positions and Coordinate Systems </user/positions>` for more
        information.
    Rocket.cp_position : Function
        Position of the rocket's center of pressure, in meters, as a function
        of Mach number, in the user defined rocket reference system. Same
        Function as ``Rocket.aerodynamic_center``.
    Rocket.stability_margin : Function
        Stability margin of the rocket, in calibers, as a function of Mach
        number and time, with the rocket flying straight into the air (zero
        angle of attack). It is the distance from the center of mass to the
        center of pressure, divided by the rocket's diameter.
    Rocket.static_margin : Function
        Static margin of the rocket, in calibers, as a function of time. It is
        the distance from the center of mass to the center of pressure at zero
        airspeed (``cp_position`` at Mach 0), divided by the rocket's diameter.
    Rocket.stability_phase : str
        Which motor phase the aerodynamic center, the margins and the lumped
        coefficients (``to_coefficients``) describe when a surface is only
        active during one phase (its ``active_during``): ``"power_off"``
        (default, the rocket after burnout) or ``"power_on"`` (while the motor
        burns). Has no effect on a rocket whose surfaces are all always active,
        nor on the flight itself, which switches surfaces on and off by time.
    Rocket.power_off_drag : Function
        Rocket's drag coefficient as a function of Mach number when the
        motor is off. Alias for ``power_off_drag_by_mach``. Assign a new drag
        curve to it to replace the drag used in the simulation.
    Rocket.power_on_drag : Function
        Rocket's drag coefficient as a function of Mach number when the
        motor is on. Alias for ``power_on_drag_by_mach``. Assign a new drag
        curve to it to replace the drag used in the simulation.
    Rocket.power_off_drag_7d : AeroCoefficient
        Rocket's drag coefficient with motor off, callable over the seven
        independent variables (alpha, beta, mach, reynolds, pitch_rate,
        yaw_rate, roll_rate), with the three rates non-dimensional
        (``rate * diameter / (2 * airspeed)``). It is the rocket's axial force
        coefficient: the force acts along the rocket's centerline.
    Rocket.power_on_drag_7d : AeroCoefficient
        Rocket's drag coefficient with motor on, callable over the seven
        independent variables (alpha, beta, mach, reynolds, pitch_rate,
        yaw_rate, roll_rate), with the three rates non-dimensional
        (``rate * diameter / (2 * airspeed)``). It is the rocket's axial force
        coefficient: the force acts along the rocket's centerline.
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
        length=None,
    ):
        """Initialize the rocket from its inertial, geometrical and aerodynamic
        parameters.

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
        power_off_drag : int, float, callable, string, array, Function
            Rocket's drag coefficient when the motor is off, based on the
            rocket's cross-section area (``pi * radius**2``). It can be a number,
            a ``.csv`` file or list of points with the Mach number in the first
            column and the drag coefficient in the second, a function such as
            ``lambda mach: ...``, or a :class:`Function`.

            The coefficient may also depend on ``alpha`` and ``beta`` (angle of
            attack and sideslip, rad), or ``alpha_total`` in their place,
            ``mach``, ``reynolds`` (based on the rocket's diameter) and
            ``pitch_rate``, ``yaw_rate`` and ``roll_rate`` (rate in rad/s times
            the diameter, divided by twice the airspeed). Name the arguments of
            the function, or the columns of the ``.csv`` file, after the ones
            used, for example ``lambda alpha, mach: ...``.

            Outside the range of its data, a table or a ``.csv`` file holds the
            value at its nearest end; a function is evaluated as given.

            For the coefficients of the whole rocket (lift, drag and moment
            against angle of attack), see :meth:`add_full_body_aerodynamics`.
        power_on_drag : int, float, callable, string, array, Function
            Rocket's drag coefficient when the motor is on. Given in the same
            way as ``power_off_drag``. If you only have one drag curve, use it
            for both.
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
        length : int, float, optional
            Overall length of the rocket, from the nose tip to the aft end, in
            meters. It is only used to report the static and stability margins
            as a percentage of the rocket's length (prints and plots). When not
            given, the length is measured from the nose cone to the aft-most
            tail, fin set or motor nozzle (see :attr:`length`). Give it when
            the rocket has no nose cone, such as a rocket described only by a
            :class:`rocketpy.GenericSurface`; otherwise the percentage is left
            out. Default is ``None``.

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
        self._length = length
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
        # What the values derived from the surfaces were last built for (see
        # _refresh_aerodynamics, _refresh_aerodynamic_center, _refresh_margins)
        self._lever_arms_stamp = None
        self._aerodynamic_center_stamp = None
        self._margins_stamp = None
        # Set once a full-body model replaces the modeled aerodynamics
        # (add_full_body_aerodynamics(overwrite=True)); warns on later surface adds.
        self._aerodynamics_overwritten = False
        # Which motor phase the stability analysis describes when a surface is
        # only active during one of them (see ``stability_phase``).
        self.stability_phase = "power_off"
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
            lambda mach, time: 0,
            inputs=["Mach", "Time (s)"],
            outputs="Stability Margin (c)",
        )
        # Yaw-plane counterparts
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
            lambda mach, time: 0,
            inputs=["Mach", "Time (s)"],
            outputs="Stability Margin - Yaw (c)",
        )

        # Define aerodynamic drag coefficients used during flight simulation
        self._set_drag("power_off", power_off_drag)
        self._set_drag("power_on", power_on_drag)

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

        # Attributes for lazy evaluation of aerodynamic centers and margins
        self._is_incidence_linear = True
        self._uses_rate_coefficients = False
        # Whether the rocket behaves the same in every plane
        self._is_axisymmetric = True

        # Initialize plots and prints object
        self.prints = _RocketPrints(self)
        self.plots = _RocketPlots(self)

    def _set_drag(self, which, source):
        """Set one of the rocket's drag curves from a user input.

        Builds the two attributes of that curve: ``<which>_drag_7d``, the
        coefficient used in the simulation, and ``<which>_drag_by_mach``, its
        view against Mach number alone (also read as ``<which>_drag``).

        Parameters
        ----------
        which : str
            ``"power_off"`` or ``"power_on"``.
        source : int, float, callable, string, array, Function
            The drag coefficient, as accepted by ``power_off_drag`` in
            :meth:`__init__`.
        """
        label = "Power On" if which == "power_on" else "Power Off"
        setattr(
            self,
            f"{which}_drag_7d",
            AeroCoefficient(
                source,
                name=f"Drag Coefficient with {label}",
                extrapolation="constant",
                single_var="mach",
            ),
        )
        # Reads the coefficient on each call, so it follows a later change of it
        # (for example the Monte Carlo drag factor).
        by_mach = Function(
            lambda mach: getattr(self, f"{which}_drag_7d")(0, 0, mach, 0, 0, 0, 0),
            inputs="Mach Number",
            outputs=f"Drag Coefficient with {label}",
            interpolation="linear",
            extrapolation="constant",
        )
        setattr(self, f"{which}_drag_by_mach", by_mach)

    @property
    def power_off_drag(self):
        """Drag coefficient with the motor off, as a Function of Mach number.

        It is read at zero angle of attack and zero rates. Assign a new drag
        curve to it, in any of the forms accepted by :meth:`__init__`, to
        replace the drag used in the simulation.
        """
        return self.power_off_drag_by_mach

    @power_off_drag.setter
    def power_off_drag(self, source):
        self._set_drag("power_off", source)

    @property
    def power_on_drag(self):
        """Drag coefficient with the motor on, as a Function of Mach number.

        It is read at zero angle of attack and zero rates. Assign a new drag
        curve to it, in any of the forms accepted by :meth:`__init__`, to
        replace the drag used in the simulation.
        """
        return self.power_on_drag_by_mach

    @power_on_drag.setter
    def power_on_drag(self, source):
        self._set_drag("power_on", source)

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

    # The values the rocket derives from its surfaces are rebuilt when read, and
    # only if something they depend on changed.

    def _surfaces_stamp(self):
        """What the aerodynamic center depends on: each surface, its version
        (counted up when its geometry changes) and where it was placed."""
        return (
            self._csys,
            self.radius,
            self.stability_phase,
            tuple(
                (surface, surface._version, position)
                for surface, position in self.aerodynamic_surfaces
            ),
        )

    def _refresh_aerodynamics(self):
        """Bring up to date what the simulation reads from the surfaces: each
        surface's center of pressure relative to the center of dry mass (for an
        individual fin, its leading edge moves with the cant angle). Returns the
        surfaces stamp."""
        surfaces_stamp = self._surfaces_stamp()
        stamp = (
            surfaces_stamp,
            self.center_of_dry_mass_position,
            self.cm_eccentricity_x,
            self.cm_eccentricity_y,
        )
        if stamp != self._lever_arms_stamp:
            self._lever_arms_stamp = stamp
            self.evaluate_surfaces_cp_to_cdm()
        return surfaces_stamp

    def _refresh_aerodynamic_center(self):
        """Rebuild the pitch/yaw aerodynamic centers if a surface changed.
        Returns the surfaces stamp."""
        stamp = self._refresh_aerodynamics()
        if stamp != self._aerodynamic_center_stamp:
            # Set before computing: evaluate_center_of_pressure reads the
            # aerodynamic center it is building (the axisymmetry check).
            self._aerodynamic_center_stamp = stamp
            self.evaluate_center_of_pressure()
        return stamp

    def _refresh_margins(self):
        """Rebuild the static/stability margins if a surface or the center of
        mass changed."""
        stamp = (self._refresh_aerodynamic_center(), self.center_of_mass)
        if stamp != self._margins_stamp:
            self.evaluate_stability_margin()
            self.evaluate_static_margin()
            # Set after computing, so a rebuild that fails is tried again
            self._margins_stamp = stamp

    @property
    def aerodynamic_center(self):
        """Position of the rocket's aerodynamic center, in meters, as a function
        of Mach number, in the user-defined rocket coordinate system.

        The aerodynamic center is the point where the extra aerodynamic force
        appears when the rocket tilts a little away from the airflow. The rocket
        is stable when it is behind the center of mass. For a rocket built from
        nose cones, fins and tails it is the center of pressure.
        """
        self._refresh_aerodynamic_center()
        return self._aerodynamic_center

    @property
    def aerodynamic_center_yaw(self):
        """Position of the rocket's aerodynamic center in the yaw plane, in
        meters, as a function of Mach number. Equals :attr:`aerodynamic_center`
        for an axisymmetric rocket.
        """
        self._refresh_aerodynamic_center()
        return self._aerodynamic_center_yaw

    def neutral_point(self, alpha, mach, beta=0.0):
        """Position of the rocket's neutral point at an angle of attack, in
        meters.

        The neutral point is the point where the extra aerodynamic force
        appears when the angle of attack changes a little from the given one.
        The stability margin is measured from it. At zero angle of attack it is
        the :attr:`aerodynamic_center`.

        Parameters
        ----------
        alpha : float
            Angle of attack, in radians, to evaluate the neutral point at.
        mach : float
            Free-stream Mach number.
        beta : float, optional
            Sideslip angle, in radians, the rocket is at. Default 0.

        Returns
        -------
        float
            Position of the neutral point along the rocket's axis, in meters,
            in the user-defined rocket coordinate system.
        """
        return neutral_point_and_slope(self, alpha, beta, mach, "pitch")[0]

    def neutral_point_yaw(self, beta, mach, alpha=0.0):
        """Position of the rocket's neutral point in the yaw plane at a sideslip
        angle, in meters.

        Same as :meth:`neutral_point`, for the sideslip angle instead of the
        angle of attack. At zero angle it is :attr:`aerodynamic_center_yaw`.

        Parameters
        ----------
        beta : float
            Sideslip angle, in radians, to evaluate the neutral point at.
        mach : float
            Free-stream Mach number.
        alpha : float, optional
            Angle of attack, in radians, the rocket is at. Default 0.

        Returns
        -------
        float
            Position of the yaw-plane neutral point along the rocket's axis, in
            meters, in the user-defined rocket coordinate system.
        """
        return neutral_point_and_slope(self, alpha, beta, mach, "yaw")[0]

    def center_of_pressure(self, alpha, mach, beta=0.0):
        """Position of the rocket's center of pressure at an angle of attack, in
        meters.

        The center of pressure is the point where the whole aerodynamic force
        on the rocket acts. For a rocket built from nose cones, fins and tails
        it does not move with the angle of attack and equals
        :attr:`cp_position`.

        Parameters
        ----------
        alpha : float
            Angle of attack, in radians. For an axisymmetric rocket, pass the
            total angle of attack and leave ``beta`` at 0.
        mach : float
            Free-stream Mach number.
        beta : float, optional
            Sideslip angle, in radians, the rocket is at. Default 0.

        Returns
        -------
        float
            Position of the center of pressure along the rocket's axis, in
            meters, in the user-defined rocket coordinate system. At zero angle
            of attack there is no sideways force, and :attr:`cp_position` is
            returned.
        """
        return center_of_pressure_position(self, alpha, beta, mach, "pitch")

    def center_of_pressure_yaw(self, beta, mach, alpha=0.0):
        """Position of the rocket's center of pressure in the yaw plane at a
        sideslip angle, in meters.

        Same as :meth:`center_of_pressure`, for the side force and the sideslip
        angle. Only needed for a rocket that is not axisymmetric.

        Parameters
        ----------
        beta : float
            Sideslip angle, in radians.
        mach : float
            Free-stream Mach number.
        alpha : float, optional
            Angle of attack, in radians, the rocket is at. Default 0.

        Returns
        -------
        float
            Position of the yaw-plane center of pressure along the rocket's
            axis, in meters, in the user-defined rocket coordinate system. At
            zero sideslip angle there is no side force, and
            :attr:`aerodynamic_center_yaw` is returned.
        """
        return center_of_pressure_position(self, alpha, beta, mach, "yaw")

    @property
    def total_lift_coeff_der(self):
        """How fast the rocket's normal force coefficient grows with the angle of
        attack, in 1/rad, as a function of Mach number: the sum over all
        aerodynamic surfaces, referenced to the rocket's cross-sectional area.
        """
        self._refresh_aerodynamic_center()
        return self._total_lift_coeff_der

    @property
    def total_side_coeff_der(self):
        """How fast the rocket's side force coefficient grows with the sideslip
        angle, in 1/rad, as a function of Mach number. Equals
        :attr:`total_lift_coeff_der` for an axisymmetric rocket.
        """
        self._refresh_aerodynamic_center()
        return self._total_side_coeff_der

    @property
    def static_margin(self):
        """Static margin of the rocket, in calibers, as a function of time (s).

        It is the distance from the center of mass to the center of pressure at
        zero airspeed (:attr:`cp_position` at Mach 0), divided by the rocket's
        diameter. It is positive when the rocket is stable.
        """
        self._refresh_margins()
        return self._static_margin

    @property
    def static_margin_yaw(self):
        """Static margin of the rocket in the yaw plane, in calibers, as a
        function of time (s). Equals :attr:`static_margin` for an axisymmetric
        rocket.
        """
        self._refresh_margins()
        return self._static_margin_yaw

    @property
    def stability_margin(self):
        """Stability margin of the rocket, in calibers, as a function of Mach
        number and time (s), with the rocket flying straight into the air (zero
        angle of attack).

        It is the distance from the center of mass to the center of pressure,
        divided by the rocket's diameter. It is positive when the rocket is
        stable. For the margin at an angle of attack, see
        :meth:`neutral_point`; for the margin along a flight, see
        :attr:`rocketpy.Flight.stability_margin`.
        """
        self._refresh_margins()
        return self._stability_margin

    @property
    def stability_margin_yaw(self):
        """Stability margin of the rocket in the yaw plane, in calibers, as a
        function of Mach number and time (s), with zero sideslip angle. Equals
        :attr:`stability_margin` for an axisymmetric rocket.
        """
        self._refresh_margins()
        return self._stability_margin_yaw

    @property
    def length(self):
        """Overall length of the rocket, from the nose tip to the aft end, in
        meters, or ``None`` when it cannot be measured.

        The ``length`` given when the rocket was built, if any. Otherwise it is
        measured from the rocket's parts: from the tip of the nose cone to the
        aft-most point among the tails, the fins and the motor nozzle. This
        needs a nose cone and at least one of a tail, a fin set or a motor.
        Without them (for example a rocket described only by a
        :class:`rocketpy.GenericSurface`) the two ends are not known and the
        length is ``None``; give ``length`` when building the rocket instead.

        The length is only used to show the static and stability margins as a
        percentage of the rocket's length in the prints and plots. They leave
        that percentage out when the length is ``None``.

        Returns
        -------
        float or None
            Overall length of the rocket, in meters. It does not depend on the
            coordinate system orientation.
        """
        if self._length is not None:
            return self._length
        has_nose = False
        has_aft_end = False
        points = []
        for surface, position in self.aerodynamic_surfaces:
            if isinstance(surface, NoseCone):
                has_nose = True
                axial_extent = surface.length
            elif isinstance(surface, Tail):
                has_aft_end = True
                axial_extent = surface.length
            elif isinstance(surface, (Fins, Fin)):
                has_aft_end = True
                axial_extent = surface.root_chord
            else:
                # Generic/controllable surfaces have no defined axial extent
                # and say nothing about where the rocket ends.
                continue
            # The reference point and the point one axial extent toward the tail
            # (the tail direction is -_csys along the z axis). Taking the global
            # extremes makes the result independent of which end is the reference.
            points.append(position.z)
            points.append(position.z - self._csys * axial_extent)
        # nozzle_position is already in the rocket reference frame.
        if getattr(self, "motor", None) is not None and not isinstance(
            self.motor, EmptyMotor
        ):
            has_aft_end = True
            points.append(self.nozzle_position)
        if not (has_nose and has_aft_end):
            return None
        return max(points) - min(points)

    def evaluate_center_of_pressure(self):
        """Compute the rocket's center of pressure as a function of Mach number.

        The result is stored in ``aerodynamic_center`` (also read as
        ``cp_position``): the average position of the aerodynamic surfaces,
        each weighted by how fast its normal force grows with the angle of
        attack. This is the center of pressure at a small angle of attack.

        The same is done for the yaw plane (``aerodynamic_center_yaw``), with
        the side force and the sideslip angle. For an axisymmetric rocket the
        two are equal. When they differ a warning is shown, because
        ``static_margin`` and ``stability_margin`` then describe the pitch
        plane only.

        A surface active during only one motor phase (its ``active_during``)
        is counted only when that phase is the rocket's ``stability_phase``
        (``"power_off"`` by default), and a surface that starts the flight
        switched off (``active=False``) is not counted. A warning says so when
        one is left out.

        Returns
        -------
        self.aerodynamic_center : Function
            Position of the rocket's pitch-plane aerodynamic center, in meters,
            as a function of Mach number, in the user-defined rocket coordinate
            system. See :doc:`Positions and Coordinate Systems
            </user/positions>` for more information.
        """
        # Re-Initialize total force coefficient derivatives and AC positions
        self._total_lift_coeff_der.set_source(lambda mach: 0)
        self._aerodynamic_center.set_source(lambda mach: 0)
        self._total_side_coeff_der.set_source(lambda mach: 0)
        self._aerodynamic_center_yaw.set_source(lambda mach: 0)

        # Surfaces active only in the other motor phase, or switched off, are
        # left out. This method runs once per configuration, so the notice is
        # shown once.
        surfaces = stability_surfaces(self)
        if len(surfaces) != len(self.aerodynamic_surfaces):
            warnings.warn(
                "The aerodynamic center, the margins and the lumped coefficients "
                f"describe the rocket during '{self.stability_phase}': surfaces "
                "active only in the other motor phase, or that start the flight "
                "switched off (`active=False`), are left out. Set "
                "`rocket.stability_phase` to 'power_on' or 'power_off' to choose "
                "the phase.",
                stacklevel=2,
            )

        # Calculate total force coefficient derivatives and aerodynamic center.
        # The parts of the surfaces' moments the weighted average leaves out
        # are kept apart and added before dividing.
        self.evaluate_surfaces_cp_to_cdm()
        pitch_left_out, yaw_left_out = [], []
        for aero_surface, position in surfaces:
            # Force-curve slopes as Functions of Mach, from the surface's
            # coefficient derivatives sliced at zero alpha/beta and zero
            # rates. The yaw slope is the sign-flipped ``cY_beta`` so an
            # axisymmetric surface gives the same signed weight as the pitch
            # plane (their margins then coincide when symmetric).
            lift_coeff_der = aero_surface.cN_alpha.slice("mach")
            side_coeff_der = -1.0 * aero_surface.cY_beta.slice("mach")
            cp_z = aero_surface.aerodynamic_center
            cp_z_yaw = aero_surface.aerodynamic_center_yaw
            # ref_factor corrects force for different reference areas
            ref_factor = aero_surface.reference_area / self.area
            self._total_lift_coeff_der += ref_factor * lift_coeff_der
            self._aerodynamic_center += (
                ref_factor * lift_coeff_der * (position.z + self._csys * cp_z)
            )

            # Yaw plane.
            self._total_side_coeff_der += ref_factor * side_coeff_der
            self._aerodynamic_center_yaw += (
                ref_factor * side_coeff_der * (position.z + self._csys * cp_z_yaw)
            )
            pitch, yaw = moment_slopes_left_out(
                self, aero_surface, position, lift_coeff_der, side_coeff_der
            )
            if pitch is not None:
                pitch_left_out.append(pitch)
            if yaw is not None:
                yaw_left_out.append(yaw)
        # Avoid errors when only zero-lift surfaces are added
        if self._total_lift_coeff_der.get_value(0) != 0:
            for moment_slope in pitch_left_out:
                self._aerodynamic_center += moment_slope
            self._aerodynamic_center /= self._total_lift_coeff_der
        if self._total_side_coeff_der.get_value(0) != 0:
            for moment_slope in yaw_left_out:
                self._aerodynamic_center_yaw += moment_slope
            self._aerodynamic_center_yaw /= self._total_side_coeff_der

        # Non-axisymmetry advisory. This method runs once per configuration of
        # the surfaces, so the warning is shown once per configuration.
        # Nose, tail and fin sets contribute identically to both planes; with
        # any other surface the rocket's own derivatives decide.
        self._is_axisymmetric = all(
            surface.is_axisymmetric for surface, _ in surfaces
        ) or is_axisymmetric(self)
        if not self._is_axisymmetric:
            # Largest difference of the two aerodynamic centers over Mach 0 to 3
            max_diff = max(
                abs(
                    self.aerodynamic_center.get_value_opt(mach)
                    - self.aerodynamic_center_yaw.get_value_opt(mach)
                )
                for mach in np.linspace(0.0, 3.0, 16)
            )
            if max_diff > 1e-6 * (2 * self.radius):
                reason = (
                    "Pitch- and yaw-plane aerodynamic centers differ "
                    f"(max difference ~{max_diff:.4g} m)"
                )
            else:
                reason = (
                    "The pitch and yaw planes differ in strength, or couple "
                    "differently seen from each"
                )
            warnings.warn(
                f"{reason}: the rocket is not axisymmetric. "
                "'aerodynamic_center', 'static_margin' and 'stability_margin' "
                "describe the PITCH plane. Use 'aerodynamic_center_yaw', "
                "'static_margin_yaw' and 'stability_margin_yaw' for the yaw "
                "plane.",
                stacklevel=2,
            )

        # The margins are built from the center of pressure
        self._margins_stamp = None
        return self._aerodynamic_center

    def disturbance_response(
        self,
        speed,
        time=0.0,
        disturbance=5.0,
        density=1.225,
        speed_of_sound=340.29,
        plane="pitch",
        duration=None,
    ):
        """Compute how the rocket would swing back after a sudden disturbance,
        such as a gust, at a flight condition you choose.

        The rocket is tilted by ``disturbance`` away from its flight direction
        and released. The result shows how its angle swings back: how fast it
        oscillates and how quickly the oscillation dies out. No flight
        simulation is needed, so it is a quick way to compare designs (fin
        size, ballast, inertia) at a condition such as the rail exit.

        The airspeed, air density and rocket mass are held fixed, so this is a
        snapshot of one instant: during the motor burn the real conditions
        change while the rocket swings. It is valid for small angles. For the
        response at an instant of a simulated flight, see
        :meth:`rocketpy.Flight.disturbance_response`.

        Parameters
        ----------
        speed : float
            Airspeed of the rocket, in m/s. For example the speed at which it
            leaves the rail.
        time : float, optional
            Time since motor ignition, in seconds. It sets how much propellant
            is left, and so the rocket's mass, center of mass and inertia, and
            whether the motor is still burning (a burning motor adds damping).
            Default is 0.
        disturbance : float, optional
            Angle the rocket is tilted by, in degrees. Default is 5.
        density : float, optional
            Air density, in kg/m³. Default is 1.225 (sea level).
        speed_of_sound : float, optional
            Speed of sound, in m/s, used to find the Mach number. Default is
            340.29 (sea level).
        plane : str, optional
            ``"pitch"`` or ``"yaw"``. The two only differ for a rocket that is
            not axisymmetric. Default is ``"pitch"``.
        duration : float, optional
            How long to follow the response, in seconds. By default, long
            enough for the oscillation to settle.

        Returns
        -------
        rocketpy.Function
            Angle of the rocket, in degrees, as a function of the time since
            the disturbance, in seconds. Call ``.plot()`` on it to see the
            curve; its title shows the natural frequency and damping ratio.

        Raises
        ------
        ValueError
            For a point mass rocket, which has no attitude to disturb.

        Examples
        --------
        >>> response = rocket.disturbance_response(speed=30, time=0.4)  # doctest: +SKIP
        >>> response.plot()  # doctest: +SKIP
        """
        inertia_about_cdm = self.I_22 if plane == "yaw" else self.I_11
        if not isinstance(inertia_about_cdm, Function):
            inertia, inertia_rate = 0.0, 0.0  # a point mass rocket
        else:
            inertia, inertia_rate = lateral_inertia_and_rate(
                self, inertia_about_cdm, time
            )
        corrective, damping = corrective_and_damping_moments(
            self,
            0.0,
            0.0,
            speed / speed_of_sound,
            time,
            speed,
            density,
            0.5 * density * speed**2,
            inertia_rate,
            plane,
        )
        return disturbance_response(corrective, damping, inertia, disturbance, duration)

    @property
    def is_axisymmetric(self):
        """Whether the rocket behaves the same in every plane through its axis,
        at small angles of attack.

        ``True`` for a rocket with evenly spaced fins. ``False`` for example
        with canards on a single axis or a single fin; the pitch and yaw planes
        then have separate values (:attr:`static_margin` and
        :attr:`static_margin_yaw`, and so on).
        """
        self._refresh_aerodynamic_center()
        return self._is_axisymmetric

    @property
    def is_incidence_linear(self):
        """Whether every aerodynamic force on the rocket grows in proportion to
        the angle of attack.

        ``True`` for a rocket built from nose cones, fins and tails. ``False``
        when a :class:`rocketpy.GenericSurface` has a force that does not, such
        as a body-lift term; the stability margin then varies with the angle of
        attack (see :meth:`neutral_point`). Checked up to 15 degrees, at Mach 0.3, 0.9 and 2.
        """
        self._refresh_margins()
        return self._is_incidence_linear

    @property
    def cp_position(self):
        """Position of the rocket's center of pressure, in meters, as a function
        of Mach number, in the user-defined rocket coordinate system.

        It is the point where the aerodynamic force acts at a small angle of
        attack. The rocket is stable when it is behind the center of mass. Same
        Function as :attr:`aerodynamic_center`.
        """
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

    def _surface_origin(self, surface, position):
        """Where a surface's own frame sits, in the user's coordinate system:
        the position it was added at, except for an individual fin, whose frame
        sits at its root leading edge once the cant angle is applied. A fin
        added at a position on the rocket axis (x = y = 0) sits on the body
        surface at its angular position."""
        if isinstance(surface, Fin):
            on_axis = position.x == 0 and position.y == 0
            return surface._compute_leading_edge_position(
                position.z if on_axis else position, self._csys
            )
        return position

    def __evaluate_single_surface_cp_to_cdm(self, surface, position):
        """Store where one surface applies its force, relative to the rocket's
        center of dry mass, in the body axes."""
        position = self._surface_origin(surface, position)
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
        pos = (
            surface._rotation_surface_to_body @ surface.force_application_point
            + pos_origin
        )
        self.surfaces_cp_to_cdm[surface] = pos

    def evaluate_stability_margin(self):
        """Compute the stability margin of the rocket as a function of Mach
        number and time.

        Returns
        -------
        stability_margin : Function
            Stability margin of the rocket, in calibers, as a function of Mach
            number and time (s), at zero angle of attack: the distance from the
            center of mass to the center of pressure, divided by the rocket's
            diameter.
        """
        self._is_incidence_linear = is_incidence_linear(self)
        self._uses_rate_coefficients = uses_rate_coefficients(self)
        self._stability_margin.set_source(
            lambda mach, time: stability_margin_and_slope(
                self, 0.0, 0.0, mach, time, "pitch"
            )[0]
        )
        # Yaw-plane stability margin (equal to the pitch plane when axisymmetric)
        self._stability_margin_yaw.set_source(
            lambda mach, time: stability_margin_and_slope(
                self, 0.0, 0.0, mach, time, "yaw"
            )[0]
        )
        return self._stability_margin

    def evaluate_static_margin(self):
        """Compute the static margin of the rocket as a function of time.

        Returns
        -------
        static_margin : Function
            Static margin of the rocket, in calibers, as a function of time
            (s): the distance from the center of mass to the center of pressure
            at zero airspeed (``cp_position`` at Mach 0), divided by the
            rocket's diameter.
        """

        # The sign flip for an upside-down coordinate system is part of the
        # formula, so each Function is updated in place and a reference kept by
        # the user stays current.
        def margin_about(center):
            return lambda time: (
                (
                    self.center_of_mass.get_value_opt(time)
                    - getattr(self, center).get_value_opt(0)
                )
                / (2 * self.radius)
                * self._csys
            )

        for margin, center, label in (
            (self._static_margin, "aerodynamic_center", "Static Margin"),
            (self._static_margin_yaw, "aerodynamic_center_yaw", "Static Margin - Yaw"),
        ):
            margin.set_source(margin_about(center))
            margin.set_inputs("Time (s)")
            margin.set_outputs(f"{label} (c)")
            margin.set_title(label)
            margin.set_discrete(lower=0, upper=self.motor.burn_out_time, samples=200)
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
        S_noz_11 = S_noz_22 = 0.5 * S_noz_33 + self.nozzle_to_cdm**2
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
        self.evaluate_com_to_cdm_function()
        self.evaluate_nozzle_gyration_tensor()

    def __add_single_surface(self, surface, position):
        """Adds a single aerodynamic surface to the rocket. Makes checks for
        rail buttons case, and position type.
        """
        position = position_vector(position)
        if isinstance(surface, RailButtons):
            self.rail_buttons = Components()
            self.rail_buttons.add(surface, position)
        else:
            self.aerodynamic_surfaces.add(surface, position)
        self.__evaluate_single_surface_cp_to_cdm(surface, position)

    def add_surfaces(self, surfaces, positions):
        """Adds one or more aerodynamic surfaces to the rocket. The aerodynamic
        surface must be an instance of a class that inherits from
        GenericSurface (e.g. NoseCone, TrapezoidalFins, etc.)

        Parameters
        ----------
        surfaces : list[GenericSurface], GenericSurface
            Aerodynamic surface to be added to the rocket. Can be a list of
            surfaces if more than one surface is to be added.
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
            cant-angle offset is considered. For an individual fin
            (TrapezoidalFin, EllipticalFin, FreeFormFin), a single number
            places that point on the body surface at the fin's angular
            position; a full (x, y, z) is used as given.
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
            which its moment coefficients are taken. Defaults to the rocket's
            center of dry mass at the time of the call, so add the motor first
            (before that, it is the center of mass without the motor).
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
        added = []
        for surface in surface_list:
            # pylint: disable-next=protected-access
            yaw_is_zero = all(
                getattr(surface, name).is_zero
                for name in surface._get_default_coefficients()
                if name.partition("_")[0] in ("cY", "cn")
            )
            if yaw_is_zero:
                warnings.warn(
                    f"'{surface.name}' has no yaw-plane coefficients (cY and cn "
                    "are zero): the rocket will have no side force or yaw moment "
                    "in flight. If the data describes an axisymmetric rocket, "
                    "build a LinearGenericSurface with axisymmetric=True, or "
                    "give the cN and cm of a GenericSurface against alpha_total.",
                    UserWarning,
                    stacklevel=2,
                )
            self.add_surfaces(surface, position)
            added.append(surface)

        if overwrite:
            # Re-arm the "added after" guard now that the full-body model is set.
            self._aerodynamics_overwritten = True

        return added if isinstance(surfaces, (list, tuple)) else added[0]

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
        self._set_drag("power_on", 0)
        self._set_drag("power_off", 0)
        warnings.warn(
            "add_full_body_aerodynamics(overwrite=True): the rocket's existing "
            "aerodynamic surfaces and both built-in drag curves (power_on_drag, "
            "power_off_drag) were cleared; the supplied surface(s) now provide "
            "the complete aerodynamics, including any drag they carry.",
            UserWarning,
            stacklevel=3,
        )
        # New adds are welcome again; the guard is re-armed once the full-body
        # surfaces are in place (see add_full_body_aerodynamics).
        self._aerodynamics_overwritten = False

    def to_coefficients(
        self,
        machs=None,
        force_convention="body",
        model="linear",
        angles=None,
        rates=True,
        reynolds=None,
        controls=None,
    ):
        """Return the whole rocket's aerodynamic coefficients, split by motor
        phase.

        Sweeps the rocket's aerodynamic surfaces and lumps them into one set
        of coefficients about the dry center of mass, in one of two forms:

        - ``model="linear"`` (default): the 36 derivatives a
          :class:`LinearGenericSurface` takes. For each of the normal force
          ``cN``, side force ``cY``, axial force ``cA``, pitch moment ``cm``,
          yaw moment ``cn`` and roll moment ``cl``, its value at zero angle
          and zero rates (``_0``) and its slopes with the angle of attack
          (``_alpha``), the sideslip angle (``_beta``) and the reduced roll,
          pitch and yaw rates (``_p``, ``_q``, ``_r``), each a curve over
          Mach. Terms that are zero at every Mach number are left out, so an
          ordinary rocket gets the familiar ``cN_alpha``, ``cm_alpha``,
          ``cY_beta``, ``cn_beta``, ``cN_q``, ``cm_q``, ``cY_r``, ``cn_r``,
          ``cl_p`` and ``cA_0``; canted fins add their roll forcing ``cl_0``,
          and a rocket that is not axisymmetric keeps its cross terms
          (``cN_beta``, ``cm_beta``, ...) and any force or moment it has at
          zero angle.
        - ``model="table"``: the six coefficients ``cN``, ``cY``, ``cA``,
          ``cm``, ``cn`` and ``cl`` as tables over the angle of attack, the
          sideslip angle and Mach, read at zero rates, which keep any curve
          in the angles (a stall, a body-lift term, the cross coupling of a
          lopsided rocket). For an axisymmetric rocket the sweep is over the
          total angle of attack and Mach only: ``cN``, ``cA``, ``cm`` and
          ``cl`` are then tables of ``alpha_total`` and ``mach``, and ``cY``
          and ``cn`` are left out since they follow from ``cN`` and ``cm``
          by the split along the crossflow (see
          :class:`GenericSurface`).
          The rate terms are added as in the linear model, ``cN_q``,
          ``cm_q``, ``cY_r``, ``cn_r``, ``cl_p`` and any other non-zero one,
          each a curve over Mach.

        The result is returned as two coefficient sets, ``"power_off"``
        (coasting) and ``"power_on"`` (motor burning). Each is built from the
        surfaces active during its phase (a surface's ``active_during``) and
        that phase's drag curve. The axial coefficient is the rocket's drag
        coefficient of the phase plus the axial force of the surfaces.

        Important
        ---------
        By default the coefficients depend on Mach number (and, for the table
        model, on the angles) only. Three things are then held fixed, and each
        can be included with an optional argument:

        - **Reynolds number.** Held at zero unless ``reynolds`` is given.
        - **Control deflections.** Each control of a
          :class:`ControllableGenericSurface` is held at its current value
          unless it is listed in ``controls``.
        - **How the rate terms change with the angles.** The rate terms are
          read at zero angle unless ``rates="at_each_angle"`` (table model).
          In every case the effect of a rate is a straight line: the rate term
          times the rate.

        Each value added to ``reynolds`` or ``controls`` multiplies the number
        of points computed, so keep those lists short.

        The linear model also keeps only the slope at zero angle, so any
        curve in the angle of attack or sideslip is lost, the drag rise with
        angle of attack (induced drag) among them. The table model keeps those
        curves within the range of ``angles`` and holds the edge value beyond
        it.

        RocketPy's built-in Barrowman surfaces (:class:`NoseCone`,
        :class:`Tail`, the fin sets and the individual fins, canted or not)
        are linear, Mach-tabulated and Reynolds-independent, so a rocket built
        only from them is reproduced exactly by the linear model to first
        order in the angles, and by the table model at every angle of an
        axisymmetric rocket. The optional arguments matter when you have added
        a :class:`GenericSurface` or :class:`ControllableGenericSurface` (or a
        Reynolds-dependent :class:`LinearGenericSurface`) whose coefficients
        vary with Reynolds number, with a control deflection or, for the
        linear model, with the angles beyond a straight line.

        Parameters
        ----------
        machs : sequence of float, optional
            Mach numbers at which the coefficients are read and tabulated.
            Defaults to ``0`` to ``3`` in steps of ``0.02`` for the linear
            model and of ``0.05`` for the table model.
        force_convention : str, optional
            The frame the force coefficients are named in. ``"body"`` (default)
            gives the body-frame set: normal ``cN_*``, side ``cY_*`` and axial
            ``cA_*``. ``"wind"`` gives the wind-frame set: lift ``cL_*``, side
            ``cQ_*`` and drag ``cD_*``. The moment derivatives (``cm_*``,
            ``cn_*``, ``cl_*``) are the same in both. The table model gives
            the body frame only.
        model : str, optional
            ``"linear"`` (default) for the derivatives, ``"table"`` for the
            tables over the angles. See above.
        angles : sequence of float, optional
            Angles, in radians, the table model is read at, used for both the
            angle of attack and the sideslip angle (for an axisymmetric
            rocket, their absolute values give the total angles of attack).
            Defaults to -30 to 30 degrees every 2 degrees, that is
            ``np.radians(np.arange(-30, 31, 2))``. Widen it for a surface
            that stalls beyond that. Ignored by the linear model.
        rates : bool or str, optional
            How the rate terms are included:

            - ``True`` (default): each rate term is read at zero angle of
              attack and sideslip.
            - ``"at_each_angle"``: table model only. Each rate term is read at
              every angle of the table, so damping that changes with the
              angle of attack is kept. The rate terms are then tables like the
              coefficients, and the rocket is swept over both the angle of
              attack and the sideslip angle even when it is axisymmetric. It
              takes about seven times as long as ``True``.
            - ``False``: no rate terms. The linear model then has no ``_p``,
              ``_q`` or ``_r`` terms and the table model holds the static
              tables only, which describe the rocket as a wind tunnel does,
              held still, with no aerodynamic damping at all.
        reynolds : float or sequence of float, optional
            Reynolds number of the rocket, based on its diameter
            (``air density * airspeed * 2 * radius / air viscosity``):

            - ``None`` (default): the coefficients are read at zero Reynolds
              number.
            - a single number: the coefficients are read at that Reynolds
              number, for example ``1e6``.
            - a list: the coefficients are read at each value and gain
              ``reynolds`` as an input, for example ``[1e5, 1e6, 1e7]``.

            It only has an effect when a surface or a drag curve of the rocket
            depends on the Reynolds number.
        controls : dict, optional
            Controls to keep as inputs of the coefficients, as a dict from the
            name of a control of a :class:`ControllableGenericSurface` to the
            values it is read at (at least two), for example
            ``{"deflection": np.radians([-10, 0, 10])}``. Each control listed
            becomes an input with that name. A control that is not listed is
            held at its current value. The controls of the rocket's surfaces
            are left as they were.

            When several surfaces use the same control name, each is kept as
            a separate input named ``<surface name>_<control name>`` (in lower
            case, with ``_`` for spaces). Giving the shared name reads all of
            them at the same values and shows a warning with the names; give
            those names to choose the values of each. With a control listed,
            the table model sweeps both the angle of attack and the sideslip
            angle. Default ``None``.

        Returns
        -------
        dict
            A dict with keys ``"power_off"`` and ``"power_on"``. Each value is
            itself a dict mapping a coefficient name to a
            :class:`rocketpy.Function`. Its inputs are, in this order: the
            angles (table model only: ``alpha_total``, or ``alpha`` and
            ``beta``), ``mach``, ``reynolds`` (when a list was given) and one
            per control listed. A derivative of the linear model and a rate
            term read at zero angle have no angle inputs.

        Examples
        --------
        Coefficients as tables over the angles, Mach and three Reynolds
        numbers:

        >>> coefficients = rocket.to_coefficients(  # doctest: +SKIP
        ...     model="table", reynolds=[1e5, 1e6, 1e7]
        ... )
        >>> cN = coefficients["power_off"]["cN"]  # doctest: +SKIP
        >>> cN(0.05, 0.6, 1e6)  # alpha_total, mach, reynolds  # doctest: +SKIP
        """
        return full_body_coefficients(
            self, machs, force_convention, model, angles, rates, reynolds, controls
        )

    def to_surface(
        self,
        machs=None,
        force_convention="body",
        name="Full Body Aerodynamics",
        model="linear",
        angles=None,
        rates=True,
        reynolds=None,
        controls=None,
    ):
        """Collapse the whole assembled rocket aerodynamics into two surfaces,
        one for coasting and one for powered flight. It reproduces the source
        rocket's aerodynamics, so a bare rocket carrying the same body and
        motor plus this pair flies the same as the fully modeled rocket.

        Important
        ---------
        With ``model="linear"`` (default) the surfaces are
        :class:`rocketpy.LinearGenericSurface` objects holding a linear summary
        of the rocket, tabulated against Mach. With ``model="table"`` they are
        :class:`rocketpy.GenericSurface` objects holding the coefficients as
        tables over the angles and Mach, which keep any curve in the angles.
        See :meth:`to_coefficients` for what each model keeps and leaves out.

        Parameters
        ----------
        machs : sequence of float, optional
            Mach numbers at which the coefficients are read and tabulated.
            Defaults to ``0`` to ``3`` in steps of ``0.02`` for the linear
            model and of ``0.05`` for the table model.
        force_convention : str, optional
            The frame the force coefficients are expressed in. ``"body"``
            (default) gives the body-frame set: normal ``cN``, side ``cY`` and
            axial ``cA`` (drag). ``"wind"`` gives the wind-frame set: lift
            ``cL``, side ``cQ`` and drag ``cD``. The moment coefficients are the
            same in both. The table model gives the body frame only.
        name : str, optional
            Base name of the returned surfaces.
            Default ``"Full Body Aerodynamics"``.
        model : str, optional
            ``"linear"`` (default) or ``"table"``. See :meth:`to_coefficients`.
        angles : sequence of float, optional
            Angles, in radians, the table model is read at. See
            :meth:`to_coefficients`.
        rates : bool or str, optional
            ``True`` (default), ``False`` or ``"at_each_angle"``: whether and
            how the surfaces carry the rate (damping) terms. See
            :meth:`to_coefficients`.
        reynolds : float or sequence of float, optional
            Reynolds number of the rocket, based on its diameter: a single
            value to read the coefficients at, or a list to make the surfaces
            follow the Reynolds number of the flight. Default ``None`` (zero).
            See :meth:`to_coefficients`.
        controls : dict, optional
            Controls the surfaces keep, from control name to the values it is
            read at, for example ``{"deflection": np.radians([-10, 0, 10])}``.
            Needs ``model="table"``; the surfaces returned are then
            :class:`rocketpy.ControllableGenericSurface` objects with those
            controls, each starting at 0. Default ``None``. See
            :meth:`to_coefficients`.

        Returns
        -------
        list of rocketpy.LinearGenericSurface, rocketpy.GenericSurface or \
rocketpy.ControllableGenericSurface
            Two surfaces, ``[power_off, power_on]``, each carrying the whole
            rocket's coefficients referenced to the rocket cross-section area
            and diameter and taken about the center of dry mass, and gated to
            its motor phase.
        """
        if controls and model != "table":
            raise ValueError(
                "A surface that keeps controls needs model='table'; the linear "
                "model has no control inputs."
            )
        coefficients = self.to_coefficients(
            machs=machs,
            force_convention=force_convention,
            model=model,
            angles=angles,
            rates=rates,
            reynolds=reynolds,
            controls=controls,
        )
        surfaces = []
        for phase in ("power_off", "power_on"):
            label = f"{name} ({phase.replace('_', ' ')})"
            if model == "table" and controls:
                surface = ControllableGenericSurface(
                    reference_area=self.area,
                    reference_length=2 * self.radius,
                    coefficients=lumped_surface_coefficients(coefficients[phase]),
                    controls=lumped_control_names(coefficients[phase]),
                    force_convention="body",
                    name=label,
                    active_during=phase,
                )
            elif model == "table":
                surface = GenericSurface(
                    reference_area=self.area,
                    reference_length=2 * self.radius,
                    coefficients=lumped_surface_coefficients(coefficients[phase]),
                    force_convention="body",
                    name=label,
                    active_during=phase,
                )
            else:
                surface = LinearGenericSurface(
                    reference_area=self.area,
                    reference_length=2 * self.radius,
                    coefficients=coefficients[phase],
                    force_convention=force_convention,
                    name=label,
                    active_during=phase,
                )
            surfaces.append(surface)
        return surfaces

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
        "removed by version 1.16.0",
        alternative="Rocket.add_trapezoidal_fins",
    )
    def add_fins(self, *args, **kwargs):  # pragma: no cover
        """See Rocket.add_trapezoidal_fins for documentation.
        This method is set to be deprecated in version 1.0.0 and fully removed
        by version 2.0.0. Use Rocket.add_trapezoidal_fins instead. It keeps the
        same arguments and signature."""
        return self.add_trapezoidal_fins(*args, **kwargs)

    @staticmethod
    def _check_fin_set_count(n):
        """Raise a ``ValueError`` unless a fin set has more than 2 fins."""
        if n <= 2:
            raise ValueError(
                "Number of fins must be greater than 2. "
                "For 1 or 2 fins, create each fin as a TrapezoidalFin, "
                "EllipticalFin or FreeFormFin object and add it to the rocket "
                "using the add_surfaces method."
            )

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
        self._check_fin_set_count(n)

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
        self._check_fin_set_count(n)

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
        self._check_fin_set_count(n)

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

            - A callable function ``trigger(context)`` that returns ``True`` \
                if the parachute ejection system should be triggered and \
                ``False`` otherwise. ``context`` holds the simulation values \
                at the moment of the check, such as ``context.pressure`` \
                (Pa), ``context.height_agl`` (m), ``context.state`` (the \
                rocket's states, read by name, such as ``context.state.vz``) \
                and ``context.previous_state``. See :ref:`triggerdetails` \
                for how to write a trigger and the values it can read. The \
                legacy form ``trigger(p, h, y)`` is deprecated.
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
        memory=None,
        override_rocket_drag=False,
        return_controller=False,
        name="AirBrakes",
        controller_name="AirBrakes Controller",
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
            ``controller_function(context) -> dict or None``. Invoked once per
            sample; its return value is appended to the controller log. Set
            ``air_brakes.deployment_level`` to apply the control action.
            ``context`` holds the values listed in :class:`rocketpy.Event`
            (``context.time``, ``context.state``, ``context.height_agl``,
            ``context.sensors``, ``context.environment``, ``context.rocket``,
            ``context.flight``, ``context.state_dot``, ``context.pressure``,
            ``context.previous_state`` and so on) plus ``context.controller``
            (the :class:`_Controller`) and ``context.controlled.air_brakes``
            (the :class:`AirBrakes` being controlled).
            The legacy positional form
            ``controller_function(time, sampling_rate, state, state_history,
            observed_variables, interactive_objects[, sensors[, environment]])``
            is deprecated.
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
                variables via the ``memory`` parameter as
                ``memory={'observed_variables': [...]}`` instead. Support
                for the positional argument will be removed in v1.14.
        memory : dict, optional
            The controller's own dictionary, kept from one run to the next.
            Read and write it as ``context.controller.memory`` inside
            ``controller_function``. Defaults to an empty dict.
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
        # Prepare controller memory and compatibility wrapper for
        # controller_function. The signature is `controller_function(context)`.
        # To avoid breaking existing user code, wrap legacy functions that
        # accept positional args.
        controller_memory = memory.copy() if memory is not None else {}

        # Map initial_observed_variables into controller memory for the
        # new API while emitting a deprecation warning for the positional
        # argument usage.
        if initial_observed_variables is not None:
            warnings.warn(
                "Passing `initial_observed_variables` to `add_air_brakes` is "
                "deprecated; supply them via `memory={'observed_variables': ...}` "
                "instead. Support for this argument will be removed in v1.14.",
                DeprecationWarning,
            )
            controller_memory["observed_variables"] = initial_observed_variables

        orig_controller = controller_function
        signature = inspect.signature(orig_controller)
        parameters = tuple(signature.parameters.values())
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

        if positional_parameter_count == 1 and not accepts_var_args:
            controller_wrapper = orig_controller
        else:
            # A legacy positional controller must accept one of the supported
            # signatures (5, 6 or 7 arguments). Reject any other count early so
            # the user gets a clear error instead of a runtime failure mid-flight.
            if not accepts_var_args and positional_parameter_count not in (5, 6, 7):
                raise ValueError(
                    "controller_function must take the event context as its "
                    "single argument, like `def controller(context): ...`. A "
                    "legacy positional controller_function must have 5, 6, or 7 "
                    f"arguments, but {positional_parameter_count} were given."
                )
            warnings.warn(
                "Calling controller_function with positional arguments is "
                "deprecated; use controller_function(context) instead. "
                "Support for positional controller arguments will be removed "
                "in v1.16.",
                DeprecationWarning,
                stacklevel=2,
            )

            def controller_wrapper(context):
                # Build positional args in the historical order described in
                # the docs.
                pos_args = [
                    context["time"],
                    sampling_rate,
                    context["canonical_state"],
                    controller_memory.get("observed_variables", []),
                    context["controlled"][0],
                    context["sensors"],
                    context["environment"],
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
            memory=controller_memory,
            name=controller_name,
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
        # The center of mass moved sideways: so did every surface relative to it
        self._refresh_aerodynamics()
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
        """Return the rocket as a dictionary, to save it and rebuild it later.

        Parameters
        ----------
        **kwargs
            ``include_outputs`` (bool, default False) also saves the values the
            rocket computes, such as its mass and margins over time, and
            ``discretize`` (bool, default False) saves those as tables sampled
            over the motor's burn instead of as functions.

        Returns
        -------
        dict
            The arguments needed to rebuild the rocket with :meth:`from_dict`.
        """
        discretize = kwargs.get("discretize", False)

        power_off_drag = self.power_off_drag_7d
        power_on_drag = self.power_on_drag_7d

        rocket_dict = {
            "radius": self.radius,
            "length": self._length,
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
            "stability_phase": self.stability_phase,
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
            aerodynamic_center = self.aerodynamic_center
            stability_margin = self.stability_margin
            # Functions of time while the motor burns
            timed = {
                name: getattr(self, name)
                for name in (
                    "thrust_to_weight",
                    "center_of_mass",
                    "motor_center_of_mass_position",
                    "reduced_mass",
                    "total_mass",
                    "total_mass_flow_rate",
                    "center_of_propellant_position",
                )
            }
            if discretize:
                timed = {
                    name: function.set_discrete_based_on_model(
                        self.motor.thrust, mutate_self=False
                    )
                    for name, function in timed.items()
                }
                aerodynamic_center = aerodynamic_center.set_discrete(
                    0, 4, 25, mutate_self=False
                )
                stability_margin = stability_margin.set_discrete(
                    (0, self.motor.burn_time[0]),
                    (2, self.motor.burn_time[1]),
                    (10, 10),
                    mutate_self=False,
                )

            rocket_dict.update(timed)
            rocket_dict["area"] = self.area
            rocket_dict["center_of_dry_mass_position"] = (
                self.center_of_dry_mass_position
            )
            rocket_dict["motor_center_of_dry_mass_position"] = (
                self.motor_center_of_dry_mass_position
            )
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

        return rocket_dict

    @classmethod
    def from_dict(cls, data):
        """Rebuild a rocket saved with :meth:`to_dict`.

        Parameters
        ----------
        data : dict
            The dictionary returned by :meth:`to_dict`.

        Returns
        -------
        Rocket
            The rocket, with its motor, surfaces, rail buttons, parachutes,
            sensors, air brakes and controllers.
        """
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
            length=data.get("length"),
        )
        rocket.stability_phase = data.get("stability_phase", "power_off")

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
