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
    EllipticalFins,
    Fins,
    NoseCone,
    RailButtons,
    Tail,
    TrapezoidalFins,
)
from rocketpy.rocket.aero_surface.fins.free_form_fins import FreeFormFins
from rocketpy.rocket.aero_surface.generic_surface import GenericSurface
from rocketpy.rocket.components import Components
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
    Rocket.cp_position : Function
        Function of Mach number expressing the rocket's center of pressure
        position relative to user defined rocket reference system.
        See :doc:`Positions and Coordinate Systems </user/positions>`
        for more information.
    Rocket.stability_margin : Function
        Stability margin of the rocket, in calibers, as a function of mach
        number and time. Stability margin is defined as the distance between
        the center of pressure and the center of mass, divided by the
        rocket's diameter.
    Rocket.static_margin : Function
        Static margin of the rocket, in calibers, as a function of time. Static
        margin is defined as the distance between the center of pressure and the
        center of mass, divided by the rocket's diameter.
    Rocket.static_margin : float
        Float value corresponding to rocket static margin when
        loaded with propellant in units of rocket diameter or calibers.
    Rocket.power_off_drag : Function
        Rocket's drag coefficient as a function of Mach number when the
        motor is off.
    Rocket.power_on_drag : Function
        Rocket's drag coefficient as a function of Mach number when the
        motor is on.
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
        self.aerodynamic_surfaces = Components()
        self.surfaces_cp_to_cdm = {}
        self.rail_buttons = Components()

        self.cp_position = Function(
            lambda mach: 0,
            inputs="Mach Number",
            outputs="Center of Pressure Position (m)",
        )
        self.total_lift_coeff_der = Function(
            lambda mach: 0,
            inputs="Mach Number",
            outputs="Total Lift Coefficient Derivative",
        )
        self.static_margin = Function(
            lambda time: 0, inputs="Time (s)", outputs="Static Margin (c)"
        )
        self.stability_margin = Function(
            lambda mach, time: 0,
            inputs=["Mach", "Time (s)"],
            outputs="Stability Margin (c)",
        )

        # Define aerodynamic drag coefficients
        # If already a Function, use it directly (preserves multi-dimensional drag)
        if isinstance(power_off_drag, Function):
            self.power_off_drag = power_off_drag
        else:
            self.power_off_drag = Function(
                power_off_drag,
                "Mach Number",
                "Drag Coefficient with Power Off",
                "linear",
                "constant",
            )

        if isinstance(power_on_drag, Function):
            self.power_on_drag = power_on_drag
        else:
            self.power_on_drag = Function(
                power_on_drag,
                "Mach Number",
                "Drag Coefficient with Power On",
                "linear",
                "constant",
            )

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

        # Evaluate stability (even though no aerodynamic surfaces are present yet)
        self.evaluate_center_of_pressure()
        self.evaluate_stability_margin()
        self.evaluate_static_margin()

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

    def evaluate_center_of_pressure(self):
        """Evaluates rocket center of pressure position relative to user defined
        rocket reference system. It can be called as many times as needed, as it
        will update the center of pressure function every time it is called. The
        code will iterate through all aerodynamic surfaces and consider each of
        their center of pressure position and derivative of the coefficient of
        lift as a function of Mach number.

        Returns
        -------
        self.cp_position : Function
            Function of Mach number expressing the rocket's center of pressure
            position relative to user defined rocket reference system.
            See :doc:`Positions and Coordinate Systems </user/positions>`
            for more information.
        """
        # Re-Initialize total lift coefficient derivative and center of pressure position
        self.total_lift_coeff_der.set_source(lambda mach: 0)
        self.cp_position.set_source(lambda mach: 0)

        # Calculate total lift coefficient derivative and center of pressure
        if len(self.aerodynamic_surfaces) > 0:
            for aero_surface, position in self.aerodynamic_surfaces:
                if isinstance(aero_surface, GenericSurface):
                    continue
                # ref_factor corrects lift for different reference areas
                ref_factor = (aero_surface.rocket_radius / self.radius) ** 2
                self.total_lift_coeff_der += ref_factor * aero_surface.clalpha
                self.cp_position += (
                    ref_factor
                    * aero_surface.clalpha
                    * (position.z - self._csys * aero_surface.cpz)
                )
            # Avoid errors when only generic surfaces are added
            if self.total_lift_coeff_der.get_value(0) != 0:
                self.cp_position /= self.total_lift_coeff_der
        return self.cp_position

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
        pos = Vector(
            [
                (position.x - self.cm_eccentricity_x) * self._csys - surface.cpx,
                (position.y - self.cm_eccentricity_y) - surface.cpy,
                (position.z - self.center_of_dry_mass_position) * self._csys
                - surface.cpz,
            ]
        )
        self.surfaces_cp_to_cdm[surface] = pos

    def evaluate_stability_margin(self):
        """Calculates the stability margin of the rocket as a function of mach
        number and time.

        Returns
        -------
        stability_margin : Function
            Stability margin of the rocket, in calibers, as a function of mach
            number and time. Stability margin is defined as the distance between
            the center of pressure and the center of mass, divided by the
            rocket's diameter.
        """
        self.stability_margin.set_source(
            lambda mach, time: (
                (
                    self.center_of_mass.get_value_opt(time)
                    - self.cp_position.get_value_opt(mach)
                )
                / (2 * self.radius)
            )
            * self._csys
        )
        return self.stability_margin

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
        self.static_margin.set_source(
            lambda time: (
                self.center_of_mass.get_value_opt(time)
                - self.cp_position.get_value_opt(0)
            )
            / (2 * self.radius)
        )
        # Change sign if coordinate system is upside down
        self.static_margin *= self._csys
        self.static_margin.set_inputs("Time (s)")
        self.static_margin.set_outputs("Static Margin (c)")
        self.static_margin.set_title("Static Margin")
        self.static_margin.set_discrete(
            lower=0, upper=self.motor.burn_out_time, samples=200
        )
        return self.static_margin

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
        self.evaluate_center_of_pressure()
        self.evaluate_surfaces_cp_to_cdm()
        self.evaluate_stability_margin()
        self.evaluate_static_margin()
        self.evaluate_com_to_cdm_function()
        self.evaluate_nozzle_gyration_tensor()

    def __add_single_surface(self, surface, position):
        """Adds a single aerodynamic surface to the rocket. Makes checks for
        rail buttons case, and position type.
        """
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
        surfaces : list, AeroSurface, NoseCone, TrapezoidalFins, EllipticalFins, Tail, RailButtons
            Aerodynamic surface to be added to the rocket. Can be a list of
            AeroSurface if more than one surface is to be added.
        positions : int, float, list, tuple, Vector
            Position, in m, of the aerodynamic surface's center of pressure
            relative to the user defined rocket coordinate system.
            If a list is passed, it will correspond to the position of each item
            in the surfaces list.
            For NoseCone type, position is relative to the nose cone tip.
            For Fins type, position is relative to the point belonging to
            the root chord which is highest in the rocket coordinate system.
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
        try:
            for surface, position in zip(surfaces, positions):
                self.__add_single_surface(surface, position)
        except TypeError:
            self.__add_single_surface(surfaces, positions)

        self.evaluate_center_of_pressure()
        self.evaluate_stability_margin()
        self.evaluate_static_margin()

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
        "removed by version 2.0.0",
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
            Fin set position relative to the rocket's coordinate system.
            By fin set position, understand the point belonging to the root
            chord which is highest in the rocket coordinate system (i.e.
            the point closest to the nose cone tip).

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
            Fin set position relative to the rocket's coordinate system. By fin
            set position, understand the point belonging to the root chord which
            is highest in the rocket coordinate system (i.e. the point
            closest to the nose cone tip).

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
            Fin set position relative to the rocket's coordinate system.
            By fin set position, understand the point belonging to the root
            chord which is highest in the rocket coordinate system (i.e.
            the point closest to the nose cone tip).

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
        radius=1.5,
        height=None,
        porosity=0.0432,
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
            Length of the non-unique semi-axis (radius) of the inflated hemispheroid
            parachute. Default value is 1.5.
            Units are in meters.
        height : float, optional
            Length of the unique semi-axis (height) of the inflated hemispheroid
            parachute. Default value is the radius of the parachute.
            Units are in meters.
        porosity : float, optional
            Geometric porosity of the canopy (ratio of open area to total canopy area),
            in [0, 1]. Affects only the added-mass scaling during descent; it does
            not change ``cd_s`` (drag). The default, 0.0432, yields an added-mass
            of 1.0 (“neutral” behavior).

        Returns
        -------
        parachute : Parachute
            Parachute containing trigger, sampling_rate, lag, cd_s, noise, radius,
            height, porosity and name. Furthermore, it stores clean_pressure_signal,
            noise_signal and noisyPressureSignal which are filled in during
            Flight simulation.
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
            rocket's user defined coordinate system and angular_position and
            radius must be given.

        Returns
        -------
        None
        """
        if isinstance(position, (float, int)):
            position = (0, 0, position)
        position = Vector(position)
        self.sensors.add(sensor, position)
        try:
            sensor._attached_rockets[self] += 1
        except KeyError:
            sensor._attached_rockets[self] = 1

    def add_air_brakes(
        self,
        drag_coefficient_curve,
        controller_function,
        sampling_rate,
        clamp=True,
        reference_area=None,
        initial_observed_variables=None,
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

        controller_function : function, callable
            An user-defined function responsible for controlling the simulation.
            This function is expected to take the following arguments, in order:

            1. `time` (float): The current simulation time in seconds.
            2. `sampling_rate` (float): The rate at which the controller
               function is called, measured in Hertz (Hz).
            3. `state` (list): The state vector of the simulation, structured as
               `[x, y, z, vx, vy, vz, e0, e1, e2, e3, wx, wy, wz]`.
            4. `state_history` (list): A record of the rocket's state at each
               step throughout the simulation. The state_history is organized as a
               list of lists, with each sublist containing a state vector. The last
               item in the list always corresponds to the previous state vector,
               providing a chronological sequence of the rocket's evolving states.
            5. `observed_variables` (list): A list containing the variables that
               the controller function returns. The initial value in the first
               step of the simulation of this list is provided by the
               `initial_observed_variables` argument.
            6. `interactive_objects` (list): A list containing the objects that
               the controller function can interact with. The objects are
               listed in the same order as they are provided in the
               `interactive_objects` argument.
            7. `sensors` (list): A list of sensors that are attached to the
               rocket. The most recent measurements of the sensors are provided
               with the ``sensor.measurement`` attribute. The sensors are
               listed in the same order as they are added to the rocket.

            This function will be called during the simulation at the specified
            sampling rate. The function should evaluate and change the observed
            objects as needed. The function should return None.

            .. note::

                The function will be called according to the sampling rate specified.

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
        _controller = _Controller(
            interactive_objects=air_brakes,
            controller_function=controller_function,
            sampling_rate=sampling_rate,
            initial_observed_variables=initial_observed_variables,
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
            4 fins.
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

        power_off_drag = self.power_off_drag
        power_on_drag = self.power_on_drag
        if discretize:
            power_off_drag = power_off_drag.set_discrete(0, 4, 50, mutate_self=False)
            power_on_drag = power_on_drag.set_discrete(0, 4, 50, mutate_self=False)

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
            cp_position = self.cp_position
            stability_margin = self.stability_margin
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
                cp_position = cp_position.set_discrete(0, 4, 25, mutate_self=False)
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
            rocket_dict["cp_position"] = cp_position
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
            interactive_objects_hash = getattr(controller, "_interactive_objects_hash")
            if interactive_objects_hash is not None:
                is_iterable = isinstance(interactive_objects_hash, Iterable)
                if not is_iterable:
                    interactive_objects_hash = [interactive_objects_hash]
                for hash_ in interactive_objects_hash:
                    if (hashed_obj := find_obj_from_hash(data, hash_)) is not None:
                        if not is_iterable:
                            controller.interactive_objects = hashed_obj
                        else:
                            controller.interactive_objects.append(hashed_obj)
                    else:
                        warnings.warn(
                            "Could not find controller interactive objects."
                            "Deserialization will proceed, results may not be accurate."
                        )
            rocket._add_controllers(controller)

        return rocket
