# -*- coding: utf-8 -*-

__author__ = "Giovani Hidalgo Ceotto, Franz Masatoshi Yuri, Mateus Stano Junqueira, Kaleb Ramos Wanderley, Calebe Gomes Teles, Matheus Doretto"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

import warnings

import numpy as np

from .AeroSurface import (
    EllipticalFins,
    Fins,
    NoseCone,
    RailButtons,
    Tail,
    TrapezoidalFins,
)
from .Components import Components
from .Function import Function, funcify_method
from .motors.Motor import EmptyMotor
from .Parachute import Parachute
from .plots.rocket_plots import _RocketPlots
from .prints.rocket_prints import _RocketPrints


class Rocket:

    """Keeps rocket information.

    Attributes
    ----------
        Geometrical attributes:
        Rocket.radius : float
            Rocket's largest radius in meters.
        Rocket.area : float
            Rocket's circular cross section largest frontal area in squared
            meters.
        Rocket.center_of_dry_mass_position : float
            Position, in m, of the rocket's center of dry mass (i.e. center of
            mass without propellant) relative to the rocket's coordinate system.
            See `Rocket.coordinate_system_orientation` for more information
            regarding the rocket's coordinate system.
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

        Mass and Inertia attributes:
        Rocket.mass : float
            Rocket's mass without propellant in kg.
        Rocket.center_of_mass : Function
            Position of the rocket's center of mass, including propellant, relative
            to the user defined rocket reference system.
            See `Rocket.coordinate_system_orientation` for more information regarding the
            coordinate system.
            Expressed in meters as a function of time.
        Rocket.reduced_mass : Function
            Function of time expressing the reduced mass of the rocket,
            defined as the product of the propellant mass and the mass
            of the rocket without propellant, divided by the sum of the
            propellant mass and the rocket mass.
        Rocket.total_mass : Function
            Function of time expressing the total mass of the rocket,
            defined as the sum of the propellant mass and the rocket
            mass without propellant.
        Rocket.thrust_to_weight : Function
            Function of time expressing the motor thrust force divided by rocket
            weight. The gravitational acceleration is assumed as 9.80665 m/s^2.

        Eccentricity attributes:
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

        Aerodynamic attributes
        Rocket.aerodynamic_surfaces : list
            Collection of aerodynamic surfaces of the rocket. Holds Nose cones,
            Fin sets, and Tails.
        Rocket.cp_position : float
            Rocket's center of pressure position relative to the user defined rocket
            reference system. See `Rocket.coordinate_system_orientation` for more information
            regarding the reference system.
            Expressed in meters.
        Rocket.static_margin : float
            Float value corresponding to rocket static margin when
            loaded with propellant in units of rocket diameter or
            calibers.
        Rocket.power_off_drag : Function
            Rocket's drag coefficient as a function of Mach number when the
            motor is off.
        Rocket.power_on_drag : Function
            Rocket's drag coefficient as a function of Mach number when the
            motor is on.
        Rocket.rail_buttons : RailButtons
            RailButtons object containing the rail buttons information.

        Motor attributes:
        Rocket.motor : Motor
            Rocket's motor. See Motor class for more details.
        Rocket.motor_position : float
            Position, in m, of the motor's nozzle exit area relative to the user defined
            rocket coordinate system. See `Rocket.coordinate_system_orientation` for more
            information regarding the rocket's coordinate system.
        Rocket.center_of_propellant_position : Function
            Position of the propellant's center of mass relative to the user defined
            rocket reference system. See `Rocket.coordinate_system_orientation` for more
            information regarding the rocket's coordinate system.
            Expressed in meters as a function of time.
    """

    def __init__(
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
            Tuple or list containing the rocket's dry mass inertia tensor
            components, in kg*m^2.
            Assuming e_3 is the rocket's axis of symmetry, e_1 and e_2 are
            orthogonal and form a plane perpendicular to e_3, the dry mass
            inertia tensor components must be given in the following order:
            (I_11, I_22, I_33, I_12, I_13, I_23), where I_ij is the
            component of the inertia tensor in the direction of e_i x e_j.
            Alternatively, the inertia tensor can be given as (I_11, I_22, I_33),
            where I_12 = I_13 = I_23 = 0.
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
            See `Rocket.coordinate_system_orientation` for more information
            regarding the rocket's coordinate system.
        coordinate_system_orientation : string, optional
            String defining the orientation of the rocket's coordinate system. The
            coordinate system is defined by the rocket's axis of symmetry. The system's
            origin may be placed anywhere along such axis, such as in the nozzle or in
            the nose cone, and must be kept the same for all other positions specified.
            The two options available are: "tail_to_nose" and "nose_to_tail". The first
            defines the coordinate system with the rocket's axis of symmetry pointing
            from the rocket's tail to the rocket's nose cone. The second option defines
            the coordinate system with the rocket's axis of symmetry pointing from the
            rocket's nose cone to the rocket's tail. Default is "tail_to_nose".

        Returns
        -------
        None
        """
        # Define coordinate system orientation
        self.coordinate_system_orientation = coordinate_system_orientation
        if coordinate_system_orientation == "tail_to_nose":
            self._csys = 1
        elif coordinate_system_orientation == "nose_to_tail":
            self._csys = -1
        else:
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
        self.cp_eccentricity_x = 0
        self.cp_eccentricity_y = 0
        self.thrust_eccentricity_y = 0
        self.thrust_eccentricity_x = 0

        # Parachute data initialization
        self.parachutes = []

        # Aerodynamic data initialization
        self.aerodynamic_surfaces = Components()

        # Rail buttons data initialization
        self.rail_buttons = Components()

        self.cp_position = 0
        self.static_margin = Function(
            lambda x: 0, inputs="Time (s)", outputs="Static Margin (c)"
        )

        # Define aerodynamic drag coefficients
        self.power_off_drag = Function(
            power_off_drag,
            "Mach Number",
            "Drag Coefficient with Power Off",
            "linear",
            "constant",
        )
        self.power_on_drag = Function(
            power_on_drag,
            "Mach Number",
            "Drag Coefficient with Power On",
            "linear",
            "constant",
        )
        self.cp_position = 0  # Set by self.evaluate_static_margin()

        # Create a, possibly, temporary empty motor
        # self.motors = Components()  # currently unused since only one motor is supported
        self.add_motor(motor=EmptyMotor(), position=0)

        # Important dynamic inertial quantities
        self.center_of_mass = None
        self.reduced_mass = None
        self.total_mass = None
        self.dry_mass = None

        # calculate dynamic inertial quantities
        self.evaluate_dry_mass()
        self.evaluate_total_mass()
        self.evaluate_center_of_dry_mass()
        self.evaluate_center_of_mass()
        self.evaluate_reduced_mass()
        self.evaluate_thrust_to_weight()

        # Evaluate static margin (even though no aerodynamic surfaces are present yet)
        self.evaluate_static_margin()

        # Initialize plots and prints object
        self.prints = _RocketPrints(self)
        self.plots = _RocketPlots(self)

        return None

    @property
    def nosecones(self):
        return self.aerodynamic_surfaces.get_by_type(NoseCone)

    @property
    def fins(self):
        return self.aerodynamic_surfaces.get_by_type(Fins)

    @property
    def tails(self):
        return self.aerodynamic_surfaces.get_by_type(Tail)

    def evaluate_total_mass(self):
        """Calculates and returns the rocket's total mass. The total
        mass is defined as the sum of the motor mass with propellant and the
        rocket mass without propellant. The function returns an object
        of the Function class and is defined as a function of time.

        Parameters
        ----------
        None

        Returns
        -------
        self.total_mass : rocketpy.Function
            Function of time expressing the total mass of the rocket,
            defined as the sum of the propellant mass and the rocket
            mass without propellant.
        """
        # Make sure there is a motor associated with the rocket
        if self.motor is None:
            print("Please associate this rocket with a motor!")
            return False

        # Calculate total mass by summing up propellant and dry mass
        self.total_mass = self.mass + self.motor.total_mass
        self.total_mass.set_outputs("Total Mass (Rocket + Propellant) (kg)")

        # Return total mass
        return self.total_mass

    def evaluate_dry_mass(self):
        """Calculates and returns the rocket's dry mass. The dry
        mass is defined as the sum of the motor's dry mass and the
        rocket mass without motor. The function returns an object
        of the Function class and is defined as a function of time.

        Parameters
        ----------
        None

        Returns
        -------
        self.total_mass : rocketpy.Function
            Function of time expressing the total mass of the rocket,
            defined as the sum of the propellant mass and the rocket
            mass without propellant.
        """
        # Make sure there is a motor associated with the rocket
        if self.motor is None:
            print("Please associate this rocket with a motor!")
            return False

        # Calculate total dry mass: motor (without propellant) + rocket
        self.dry_mass = self.mass + self.motor.dry_mass

        # Return total mass
        return self.dry_mass

    def evaluate_center_of_mass(self):
        """Evaluates rocket center of mass position relative to user defined rocket
        reference system.

        Parameters
        ----------
        None

        Returns
        -------
        self.center_of_mass : rocketpy.Function
            Function of time expressing the rocket's center of mass position relative to
            user defined rocket reference system.
            See `Rocket.coordinate_system_orientation` for more information.
        """
        # Compute center of mass position
        self.center_of_mass = (
            self.center_of_mass_without_motor * self.mass
            + self.motor_center_of_mass_position * self.motor.total_mass
        ) / self.total_mass
        self.center_of_mass.set_inputs("Time (s)")
        self.center_of_mass.set_outputs("Center of Mass Position (m)")

        return self.center_of_mass

    def evaluate_center_of_dry_mass(self):
        """Evaluates rocket center dry of mass (i.e. without propellant)
        position relative to user defined rocket reference system.

        Parameters
        ----------
        None

        Returns
        -------
        self.center_of_dry_mass : int, float
            Rocket's center of dry mass position relative to user defined rocket
            reference system. See `Rocket.coordinate_system_orientation` for more
            information.
        """
        # Compute center of mass position
        self.center_of_dry_mass_position = (
            self.center_of_mass_without_motor * self.mass
            + self.motor_center_of_dry_mass_position * self.motor.dry_mass
        ) / self.dry_mass

        return self.center_of_dry_mass_position

    def evaluate_reduced_mass(self):
        """Calculates and returns the rocket's total reduced mass. The
        reduced mass is defined as the product of the propellant mass
        and the mass of the rocket without propellant, divided by the
        sum of the propellant mass and the rocket mass. The function
        returns an object of the Function class and is defined as a
        function of time.

        Parameters
        ----------
        None

        Returns
        -------
        self.reduced_mass : Function
            Function of time expressing the reduced mass of the rocket,
            defined as the product of the propellant mass and the mass
            of the rocket without propellant, divided by the sum of the
            propellant mass and the rocket mass.
        """
        # Make sure there is a motor associated with the rocket
        if self.motor is None:
            print("Please associate this rocket with a motor!")
            return False

        # Retrieve propellant mass as a function of time
        motor_mass = self.motor.propellant_mass

        # retrieve constant rocket mass without propellant
        mass = self.dry_mass

        # calculate reduced mass
        self.reduced_mass = motor_mass * mass / (motor_mass + mass)
        self.reduced_mass.set_outputs("Reduced Mass (kg)")

        # Return reduced mass
        return self.reduced_mass

    def evaluate_thrust_to_weight(self):
        """Evaluates thrust to weight as a Function of time.

        Uses g = 9.80665 m/s² as nominal gravity for weight calculation.

        Returns
        -------
        None
        """
        self.thrust_to_weight = self.motor.thrust / (9.80665 * self.total_mass)
        self.thrust_to_weight.set_inputs("Time (s)")
        self.thrust_to_weight.set_outputs("Thrust/Weight")

    def evaluate_static_margin(self):
        """Calculates and returns the rocket's static margin when
        loaded with propellant. The static margin is saved and returned
        in units of rocket diameter or calibers. This function also calculates
        the rocket center of pressure and total lift coefficients.

        Parameters
        ----------
        None

        Returns
        -------
        self.static_margin : float
            Float value corresponding to rocket static margin when
            loaded with propellant in units of rocket diameter or
            calibers.
        """
        # Initialize total lift coefficient derivative and center of pressure position
        self.total_lift_coeff_der = 0
        self.cp_position = 0

        # Calculate total lift coefficient derivative and center of pressure
        if len(self.aerodynamic_surfaces) > 0:
            for aero_surface, position in self.aerodynamic_surfaces:
                self.total_lift_coeff_der += aero_surface.clalpha(0)
                self.cp_position += aero_surface.clalpha(0) * (
                    position - self._csys * aero_surface.cpz
                )
            self.cp_position /= self.total_lift_coeff_der

        # Calculate static margin
        self.static_margin = (self.center_of_mass - self.cp_position) / (
            2 * self.radius
        )
        self.static_margin *= (
            self._csys
        )  # Change sign if coordinate system is upside down
        self.static_margin.set_inputs("Time (s)")
        self.static_margin.set_outputs("Static Margin (c)")
        self.static_margin.set_discrete(
            lower=0, upper=self.motor.burn_out_time, samples=200
        )
        return None

    def evaluate_dry_inertias(self):
        """Calculates and returns the rocket's dry inertias relative to
        the rocket's center of mass. The inertias are saved and returned
        in units of kg*m².

        Parameters
        ----------
        None

        Returns
        -------
        self.dry_I_11 : float
            Float value corresponding to rocket inertia tensor 11
            component, which corresponds to the inertia relative to the
            e_1 axis, centered at the instantaneous center of mass.
        self.dry_I_22 : float
            Float value corresponding to rocket inertia tensor 22
            component, which corresponds to the inertia relative to the
            e_2 axis, centered at the instantaneous center of mass.
        self.dry_I_33 : float
            Float value corresponding to rocket inertia tensor 33
            component, which corresponds to the inertia relative to the
            e_3 axis, centered at the instantaneous center of mass.
        self.dry_I_12 : float
            Float value corresponding to rocket inertia tensor 12
            component, which corresponds to the inertia relative to the
            e_1 and e_2 axes, centered at the instantaneous center of mass.
        self.dry_I_13 : float
            Float value corresponding to rocket inertia tensor 13
            component, which corresponds to the inertia relative to the
            e_1 and e_3 axes, centered at the instantaneous center of mass.
        self.dry_I_23 : float
            Float value corresponding to rocket inertia tensor 23
            component, which corresponds to the inertia relative to the
            e_2 and e_3 axes, centered at the instantaneous center of mass.

        Notes
        -----
        The e_1 and e_2 directions are assumed to be the directions
        perpendicular to the rocket axial direction.
        The e_3 direction is assumed to be the direction parallel to the axis
        of symmetry of the rocket.
        RocketPy follows the definition of the inertia tensor as in [1], which
        includes the minus sign for all products of inertia.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor
        """
        # Compute axes distances
        noMCM_to_CDM = (
            self.center_of_mass_without_motor - self.center_of_dry_mass_position
        )
        motorCDM_to_CDM = (
            self.motor_center_of_dry_mass_position - self.center_of_dry_mass_position
        )

        # Compute dry inertias
        self.dry_I_11 = (
            self.I_11_without_motor
            + self.mass * noMCM_to_CDM**2
            + self.motor.dry_I_11
            + self.motor.dry_mass * motorCDM_to_CDM**2
        )
        self.dry_I_22 = (
            self.I_22_without_motor
            + self.mass * noMCM_to_CDM**2
            + self.motor.dry_I_22
            + self.motor.dry_mass * motorCDM_to_CDM**2
        )
        self.dry_I_33 = self.I_33_without_motor + self.motor.dry_I_33
        self.dry_I_12 = self.I_12_without_motor + self.motor.dry_I_12
        self.dry_I_13 = self.I_13_without_motor + self.motor.dry_I_13
        self.dry_I_23 = self.I_23_without_motor + self.motor.dry_I_23

        # Return inertias
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
        the rocket's center of mass. The inertias are saved and returned
        in units of kg*m².

        Parameters
        ----------
        None

        Returns
        -------
        self.I_11 : float
            Float value corresponding to rocket inertia tensor 11
            component, which corresponds to the inertia relative to the
            e_1 axis, centered at the instantaneous center of mass.
        self.I_22 : float
            Float value corresponding to rocket inertia tensor 22
            component, which corresponds to the inertia relative to the
            e_2 axis, centered at the instantaneous center of mass.
        self.I_33 : float
            Float value corresponding to rocket inertia tensor 33
            component, which corresponds to the inertia relative to the
            e_3 axis, centered at the instantaneous center of mass.

        Notes
        -----
        The e_1 and e_2 directions are assumed to be the directions
        perpendicular to the rocket axial direction.
        The e_3 direction is assumed to be the direction parallel to the axis
        of symmetry of the rocket.
        RocketPy follows the definition of the inertia tensor as in [1], which
        includes the minus sign for all products of inertia.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor
        """
        # Get masses
        prop_mass = self.motor.propellant_mass  # Propellant mass as a function of time
        dry_mass = self.dry_mass  # Constant rocket dry mass without propellant

        # Compute axes distances
        CM_to_CDM = self.center_of_mass - self.center_of_dry_mass_position
        CM_to_CPM = self.center_of_mass - self.center_of_propellant_position

        # Compute inertias
        self.I_11 = (
            self.dry_I_11
            + self.motor.I_11
            + dry_mass * CM_to_CDM**2
            + prop_mass * CM_to_CPM**2
        )
        self.I_22 = (
            self.dry_I_22
            + self.motor.I_22
            + dry_mass * CM_to_CDM**2
            + prop_mass * CM_to_CPM**2
        )
        self.I_33 = self.dry_I_33 + self.motor.I_33
        self.I_12 = self.dry_I_12 + self.motor.I_12
        self.I_13 = self.dry_I_13 + self.motor.I_13
        self.I_23 = self.dry_I_23 + self.motor.I_23

        # Return inertias
        return (
            self.I_11,
            self.I_22,
            self.I_33,
            self.I_12,
            self.I_13,
            self.I_23,
        )

    def evaluate_nozzle_gyration_tensor(self):
        pass

    def add_motor(self, motor, position):
        """Adds a motor to the rocket.

        Parameters
        ----------
        motor : Motor, SolidMotor, HybridMotor, EmptyMotor
            Motor to be added to the rocket. See Motor class for more
            information.
        position : int, float
            Position, in m, of the motor's nozzle exit area relative to the user
            defined rocket coordinate system.
            See `Rocket.coordinate_system_orientation` for more information
            regarding the rocket's coordinate system.

        Returns
        -------
        None
        """
        if hasattr(self, "motor") and not isinstance(self.motor, EmptyMotor):
            print(
                "Only one motor per rocket is currently supported. "
                + "Overwriting previous motor."
            )
        self.motor = motor
        self.motor_position = position
        _ = self._csys * self.motor._csys
        self.center_of_propellant_position = (
            self.motor.center_of_propellant_mass - self.motor.nozzle_position
        ) * _ + self.motor_position
        self.motor_center_of_mass_position = (
            self.motor.center_of_mass - self.motor.nozzle_position
        ) * _ + self.motor_position
        self.motor_center_of_dry_mass_position = (
            self.motor.center_of_dry_mass - self.motor.nozzle_position
        ) * _ + self.motor_position
        self.evaluate_dry_mass()
        self.evaluate_total_mass()
        self.evaluate_center_of_dry_mass()
        self.evaluate_center_of_mass()
        self.evaluate_dry_inertias()
        self.evaluate_inertias()
        self.evaluate_reduced_mass()
        self.evaluate_thrust_to_weight()
        self.evaluate_static_margin()
        return None

    def add_surfaces(self, surfaces, positions):
        """Adds one or more aerodynamic surfaces to the rocket. The aerodynamic
        surface must be an instance of a class that inherits from the
        AeroSurface (e.g. NoseCone, TrapezoidalFins, etc.)

        Parameters
        ----------
        surfaces : list, AeroSurface, NoseCone, TrapezoidalFins, EllipticalFins, Tail
            Aerodynamic surface to be added to the rocket. Can be a list of
            AeroSurface if more than one surface is to be added.
            See AeroSurface class for more information.
        positions : int, float, list
            Position, in m, of the aerodynamic surface's center of pressure
            relative to the user defined rocket coordinate system.
            See `Rocket.coordinate_system_orientation` for more information
            regarding the rocket's coordinate system.
            If a list is passed, it will correspond to the position of each item
            in the surfaces list.
            For NoseCone type, position is relative to the nose cone tip.
            For Fins type, position is relative to the point belonging to
            the root chord which is highest in the rocket coordinate system.
            For Tail type, position is relative to the point belonging to the
            tail which is highest in the rocket coordinate system.

        Returns
        -------
        None
        """
        try:
            for surface, position in zip(surfaces, positions):
                self.aerodynamic_surfaces.add(surface, position)
        except TypeError:
            self.aerodynamic_surfaces.add(surfaces, positions)

        self.evaluate_static_margin()
        return None

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
            By tail position, understand the point belonging to the tail which is
            highest in the rocket coordinate system (i.e. generally the point closest
            to the nose cone).
            See `Rocket.coordinate_system_orientation` for more information.

        Returns
        -------
        tail : Tail
            Tail object created.
        """

        # Modify reference radius if not provided
        radius = self.radius if radius is None else radius

        # Create new tail as an object of the Tail class
        tail = Tail(top_radius, bottom_radius, length, radius, name)

        # Add tail to aerodynamic surfaces
        self.add_surfaces(tail, position)

        # Return self
        return tail

    def add_nose(self, length, kind, position, name="Nosecone"):
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
            Nose cone type. Von Karman, conical, ogive, and lvhaack are
            supported.
        position : int, float
            Nose cone tip coordinate relative to the rocket's coordinate system.
            See `Rocket.coordinate_system_orientation` for more information.
        name : string
            Nose cone name. Default is "Nose Cone".

        Returns
        -------
        nose : Nose
            Nose cone object created.
        """
        # Create a nose as an object of NoseCone class
        nose = NoseCone(length, kind, self.radius, self.radius, name)

        # Add nose to the list of aerodynamic surfaces
        self.add_surfaces(nose, position)

        # Return self
        return nose

    def add_fins(self, *args, **kwargs):
        """See Rocket.add_trapezoidal_fins for documentation.
        This method is set to be deprecated in version 1.0.0 and fully removed
        by version 2.0.0. Use Rocket.add_trapezoidal_fins instead. It keeps the
        same arguments and signature."""
        warnings.warn(
            "This method is set to be deprecated in version 1.0.0 and fully "
            "removed by version 2.0.0. Use Rocket.add_trapezoidal_fins instead",
            PendingDeprecationWarning,
        )
        return self.add_trapezoidal_fins(*args, **kwargs)

    def add_trapezoidal_fins(
        self,
        n,
        root_chord,
        tip_chord,
        span,
        position,
        cant_angle=0,
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
            Number of fins, from 2 to infinity.
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
            generally the point closest to the nose cone tip).
            See `Rocket.coordinate_system_orientation` for more information.
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
            Reference radius to calculate lift coefficient. If None, which is
            default, use rocket radius.
        airfoil : tuple, optional
            Default is null, in which case fins will be treated as flat plates.
            Otherwise, if tuple, fins will be considered as airfoils. The
            tuple's first item specifies the airfoil's lift coefficient
            by angle of attack and must be either a .csv, .txt, ndarray
            or callable. The .csv and .txt files must contain no headers
            and the first column must specify the angle of attack, while
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

        # Return the created aerodynamic surface
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
            Number of fins, from 2 to infinity.
        root_chord : int, float
            Fin root chord in meters.
        span : int, float
            Fin span in meters.
        position : int, float
            Fin set position relative to the rocket's coordinate system. By fin
            set position, understand the point belonging to the root chord which
            is highest in the rocket coordinate system (i.e. generally the point
            closest to the nose cone tip).
            See `Rocket.coordinate_system_orientation` for more information.
        cant_angle : int, float, optional
            Fins cant angle with respect to the rocket centerline. Must be given
            in degrees.
        radius : int, float, optional
            Reference radius to calculate lift coefficient. If None, which
            is default, use rocket radius.
        airfoil : tuple, optional
            Default is null, in which case fins will be treated as flat plates.
            Otherwise, if tuple, fins will be considered as airfoils. The
            tuple's first item specifies the airfoil's lift coefficient
            by angle of attack and must be either a .csv, .txt, ndarray
            or callable. The .csv and .txt files must contain no headers
            and the first column must specify the angle of attack, while
            the second column must specify the lift coefficient. The
            ndarray should be as [(x0, y0), (x1, y1), (x2, y2), ...]
            where x0 is the angle of attack and y0 is the lift coefficient.
            If callable, it should take an angle of attack as input and
            return the lift coefficient at that angle of attack.
            The tuple's second item is the unit of the angle of attack,
            accepting either "radians" or "degrees".

        Returns
        -------
        fin_set : EllipticalFins
            Fin set object created.
        """

        # Modify radius if not given, use rocket radius, otherwise use given.
        radius = radius if radius is not None else self.radius

        # Create a fin set as an object of EllipticalFins class
        fin_set = EllipticalFins(n, root_chord, span, radius, cant_angle, airfoil, name)

        # Add fin set to the list of aerodynamic surfaces
        self.add_surfaces(fin_set, position)

        # Return self
        return fin_set

    def add_parachute(
        self, name, cd_s, trigger, sampling_rate=100, lag=0, noise=(0, 0, 0)
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
        trigger : function, float, string
            Trigger for the parachute deployment. Can be a float with the height
            in which the parachute is ejected (ejection happens after apogee); or
            the string "apogee", for ejection at apogee.
            Can also be a function which defines if the parachute ejection
            system is to be triggered. It must take as input the freestream
            pressure in pascal, the height in meters (above ground level), and
            the state vector of the simulation, which is defined by
            [x, y, z, vx, vy, vz, e0, e1, e2, e3, wx, wy, wz].
            The trigger will be called according to the sampling rate given next.
            It should return True if the parachute ejection system is to be
            triggered and False otherwise.
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

        Returns
        -------
        parachute : Parachute
            Parachute  containing trigger, sampling_rate, lag, cd_s, noise
            and name. Furthermore, it stores clean_pressure_signal,
            noise_signal and noisyPressureSignal which are filled in during
            Flight simulation.
        """
        # Create a parachute
        parachute = Parachute(name, cd_s, trigger, sampling_rate, lag, noise)

        # Add parachute to list of parachutes
        self.parachutes.append(parachute)

        # Return self
        return self.parachutes[-1]

    def set_rail_buttons(
        self, upper_button_position, lower_button_position, angular_position=45
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
            See `Rocket.coordinate_system_orientation` for more information.
        lower_button_position : int, float
            Position of the rail button closest to the nozzle relative to
            the rocket's coordinate system, in meters.
            See `Rocket.coordinate_system_orientation` for more information.
        angular_position : float, optional
            Angular position of the rail buttons in degrees measured
            as the rotation around the symmetry axis of the rocket
            relative to one of the other principal axis.
            Default value is 45 degrees, generally used in rockets with
            4 fins.

        Returns
        -------
        rail_buttons : RailButtons
            RailButtons object created
        """
        # Create a rail buttons object
        buttons_distance = abs(upper_button_position - lower_button_position)
        rail_buttons = RailButtons(
            buttons_distance=buttons_distance, angular_position=angular_position
        )
        self.rail_buttons.add(rail_buttons, lower_button_position)
        return rail_buttons

    def add_cm_eccentricity(self, x, y):
        """Moves line of action of aerodynamic and thrust forces by
        equal translation amount to simulate an eccentricity in the
        position of the center of mass of the rocket relative to its
        geometrical center line. Should not be used together with
        add_cp_eccentricity and add_thrust_eccentricity.

        Parameters
        ----------
        x : float
            Distance in meters by which the CM is to be translated in
            the x direction relative to geometrical center line.
        y : float
            Distance in meters by which the CM is to be translated in
            the y direction relative to geometrical center line.

        Returns
        -------
        self : Rocket
            Object of the Rocket class.
        """
        # Move center of pressure to -x and -y
        self.cp_eccentricity_x = -x
        self.cp_eccentricity_y = -y

        # Move thrust center by -x and -y
        self.thrust_eccentricity_y = -x
        self.thrust_eccentricity_x = -y

        # Return self
        return self

    def add_cp_eccentricity(self, x, y):
        """Moves line of action of aerodynamic forces to simulate an
        eccentricity in the position of the center of pressure relative
        to the center of mass of the rocket.

        Parameters
        ----------
        x : float
            Distance in meters by which the CP is to be translated in
            the x direction relative to the center of mass axial line.
        y : float
            Distance in meters by which the CP is to be translated in
            the y direction relative to the center of mass axial line.

        Returns
        -------
        self : Rocket
            Object of the Rocket class.
        """
        # Move center of pressure by x and y
        self.cp_eccentricity_x = x
        self.cp_eccentricity_y = y

        # Return self
        return self

    def add_thrust_eccentricity(self, x, y):
        """Moves line of action of thrust forces to simulate a
        misalignment of the thrust vector and the center of mass.

        Parameters
        ----------
        x : float
            Distance in meters by which the line of action of the
            thrust force is to be translated in the x direction
            relative to the center of mass axial line.
        y : float
            Distance in meters by which the line of action of the
            thrust force is to be translated in the x direction
            relative to the center of mass axial line.

        Returns
        -------
        self : Rocket
            Object of the Rocket class.
        """
        # Move thrust line by x and y
        self.thrust_eccentricity_y = x
        self.thrust_eccentricity_x = y

        # Return self
        return self

    def info(self):
        """Prints out a summary of the data and graphs available about
        the Rocket.

        Parameters
        ----------
        None

        Return
        ------
        None
        """
        # All prints
        self.prints.all()

        return None

    def all_info(self):
        """Prints out all data and graphs available about the Rocket.

        Parameters
        ----------
        None

        Return
        ------
        None
        """

        # All prints and plots
        self.info()
        self.plots.all()

        return None

    def add_fin(
        self,
        number_of_fins=4,
        cl=2 * np.pi,
        cpr=1,
        cpz=1,
        gammas=[0, 0, 0, 0],
        angular_positions=None,
    ):
        "Hey! I will document this function later"
        self.aerodynamic_surfaces = Components()
        pi = np.pi
        # Calculate angular positions if not given
        if angular_positions is None:
            angular_positions = (
                np.array(range(number_of_fins)) * 2 * pi / number_of_fins
            )
        else:
            angular_positions = np.array(angular_positions) * pi / 180
        # Convert gammas to degree
        if isinstance(gammas, (int, float)):
            gammas = [(pi / 180) * gammas for i in range(number_of_fins)]
        else:
            gammas = [(pi / 180) * gamma for gamma in gammas]
        for i in range(number_of_fins):
            # Get angular position and inclination for current fin
            angularPosition = angular_positions[i]
            gamma = gammas[i]
            # Calculate position vector
            cpx = cpr * np.cos(angularPosition)
            cpy = cpr * np.sin(angularPosition)
            positionVector = np.array([cpx, cpy, cpz])
            # Calculate chord vector
            auxVector = np.array([cpy, -cpx, 0]) / (cpr)
            chordVector = (
                np.cos(gamma) * np.array([0, 0, 1]) - np.sin(gamma) * auxVector
            )
            self.aerodynamic_surfaces.append([positionVector, chordVector])
        return None
