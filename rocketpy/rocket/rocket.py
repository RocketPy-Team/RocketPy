import warnings

import numpy as np

from rocketpy.mathutils.function import Function
from rocketpy.motors.motor import EmptyMotor
from rocketpy.plots.rocket_plots import _RocketPlots
from rocketpy.prints.rocket_prints import _RocketPrints
from rocketpy.rocket.aero_surface import (
    EllipticalFins,
    Fins,
    NoseCone,
    RailButtons,
    Tail,
    TrapezoidalFins,
)
from rocketpy.rocket.components import Components
from rocketpy.rocket.parachute import Parachute


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
    Rocket.center_of_mass : Function
        Position of the rocket's center of mass, including propellant, relative
        to the user defined rocket reference system.
        See :doc:`Positions and Coordinate Systems </user/positions>`
        for more information
        regarding the coordinate system.
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

        # Parachute, Aerodynamic and Rail buttons data initialization
        self.parachutes = []
        self.aerodynamic_surfaces = Components()
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
        rocket mass without motor. The function returns an object
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

        # Calculate total dry mass: motor (without propellant) + rocket mass
        self.dry_mass = self.mass + self.motor.dry_mass

        return self.dry_mass

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
                self.total_lift_coeff_der += aero_surface.clalpha
                self.cp_position += aero_surface.clalpha * (
                    position - self._csys * aero_surface.cpz
                )
            self.cp_position /= self.total_lift_coeff_der

        return self.cp_position

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
                (self.center_of_mass(time) - self.cp_position(mach)) / (2 * self.radius)
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
            lambda time: (self.center_of_mass(time) - self.cp_position(0))
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
        the rocket's center of mass. The inertias are saved and returned
        in units of kg*m². This does not consider propellant mass but does take
        into account the motor dry mass.

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
        dry_mass = self.dry_mass  # Constant rocket mass with motor, without propellant

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
        motor : Motor, SolidMotor, HybridMotor, LiquidMotor, GenericMotor
            Motor to be added to the rocket.
        position : int, float
            Position, in meters, of the motor's coordinate system origin
            relative to the user defined rocket coordinate system.

        See Also
        --------
        :ref:`add_surfaces`

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
            self.motor.center_of_propellant_mass * _ + self.motor_position
        )
        self.motor_center_of_mass_position = (
            self.motor.center_of_mass * _ + self.motor_position
        )
        self.motor_center_of_dry_mass_position = (
            self.motor.center_of_dry_mass_position * _ + self.motor_position
        )
        self.nozzle_position = self.motor.nozzle_position * _ + self.motor_position
        self.evaluate_dry_mass()
        self.evaluate_total_mass()
        self.evaluate_center_of_dry_mass()
        self.evaluate_center_of_mass()
        self.evaluate_dry_inertias()
        self.evaluate_inertias()
        self.evaluate_reduced_mass()
        self.evaluate_thrust_to_weight()
        self.evaluate_center_of_pressure()
        self.evaluate_stability_margin()
        self.evaluate_static_margin()

    def add_surfaces(self, surfaces, positions):
        """Adds one or more aerodynamic surfaces to the rocket. The aerodynamic
        surface must be an instance of a class that inherits from the
        AeroSurface (e.g. NoseCone, TrapezoidalFins, etc.)

        Parameters
        ----------
        surfaces : list, AeroSurface, NoseCone, TrapezoidalFins, EllipticalFins, Tail
            Aerodynamic surface to be added to the rocket. Can be a list of
            AeroSurface if more than one surface is to be added.
        positions : int, float, list
            Position, in m, of the aerodynamic surface's center of pressure
            relative to the user defined rocket coordinate system.
            If a list is passed, it will correspond to the position of each item
            in the surfaces list.
            For NoseCone type, position is relative to the nose cone tip.
            For Fins type, position is relative to the point belonging to
            the root chord which is highest in the rocket coordinate system.
            For Tail type, position is relative to the point belonging to the
            tail which is highest in the rocket coordinate system.

        See Also
        --------
        :ref:`add_surfaces`

        Returns
        -------
        None
        """
        try:
            for surface, position in zip(surfaces, positions):
                self.aerodynamic_surfaces.add(surface, position)
        except TypeError:
            self.aerodynamic_surfaces.add(surfaces, positions)

        self.evaluate_center_of_pressure()
        self.evaluate_stability_margin()
        self.evaluate_static_margin()

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

        See Also
        --------
        :ref:`add_surfaces`

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

    def add_nose(self, length, kind, position, bluffness=0, name="Nose Cone"):
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
        bluffness : float, optional
            Ratio between the radius of the circle on the tip of the ogive and
            the radius of the base of the ogive.
        name : string
            Nose cone name. Default is "Nose Cone".

        See Also
        --------
        :ref:`add_surfaces`

        Returns
        -------
        nose : Nose
            Nose cone object created.
        """
        nose = NoseCone(
            length=length,
            kind=kind,
            base_radius=self.radius,
            rocket_radius=self.radius,
            bluffness=bluffness,
            name=name,
        )
        self.add_surfaces(nose, position)
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
            the point closest to the nose cone tip).

            See Also
            --------
            :ref:`add_surfaces`
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
            is highest in the rocket coordinate system (i.e. the point
            closest to the nose cone tip).

            See Also
            --------
            :ref:`add_surfaces`
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
        radius = radius if radius is not None else self.radius
        fin_set = EllipticalFins(n, root_chord, span, radius, cant_angle, airfoil, name)
        self.add_surfaces(fin_set, position)
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
            This parameter defines the trigger condition for the parachute
            ejection system. It can be one of the following:

            - A callable function that takes three arguments:
                1. Freestream pressure in pascals.
                2. Height in meters above ground level.
                3. The state vector of the simulation, which is defined as:
                    [x, y, z, vx, vy, vz, e0, e1, e2, e3, wx, wy, wz].

            The function should return True if the parachute ejection system should
            be triggered and False otherwise.

            - A float value, representing an absolute height in meters. In this
            case, the parachute will be ejected when the rocket reaches this height
            above ground level.

            - The string "apogee," which triggers the parachute at apogee, i.e.,
            when the rocket reaches its highest point and starts descending.

            Note: The function will be called according to the sampling rate
            specified next.
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
        parachute = Parachute(name, cd_s, trigger, sampling_rate, lag, noise)
        self.parachutes.append(parachute)
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

        See Also
        --------
        :ref:`add_surfaces`

        Returns
        -------
        rail_buttons : RailButtons
            RailButtons object created
        """
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
        geometrical center line.

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

        Notes
        -----
        Should not be used together with add_cp_eccentricity and
        add_thrust_eccentricity.
        """
        self.cp_eccentricity_x = -x
        self.cp_eccentricity_y = -y
        self.thrust_eccentricity_y = -x
        self.thrust_eccentricity_x = -y
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
        self.cp_eccentricity_x = x
        self.cp_eccentricity_y = y
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
        self.thrust_eccentricity_y = x
        self.thrust_eccentricity_x = y
        return self

    def draw(self, vis_args=None):
        """Draws the rocket in a matplotlib figure.

        Parameters
        ----------
        vis_args : dict, optional
            Determines the visual aspects when drawing the rocket. If None,
            default values are used. Default values are:
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
        """
        self.plots.draw(vis_args)
        return None

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
