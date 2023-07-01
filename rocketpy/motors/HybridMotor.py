# -*- coding: utf-8 -*-

__author__ = "Giovani Hidalgo Ceotto, Oscar Mauricio Prada Ramirez, João Lemes Gribel Soares, Lucas Kierulff Balabram, Lucas Azevedo Pezente"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

try:
    from functools import cached_property
except ImportError:
    from rocketpy.tools import cached_property

from rocketpy.Function import funcify_method, reset_funcified_methods
from rocketpy.plots.hybrid_motor_plots import _HybridMotorPlots
from rocketpy.prints.hybrid_motor_prints import _HybridMotorPrints

from .LiquidMotor import LiquidMotor
from .Motor import Motor
from .SolidMotor import SolidMotor


class HybridMotor(Motor):
    """Class to specify characteristics and useful operations for Hybrid
    motors.

    Attributes
    ----------

        Geometrical attributes:
        Motor.coordinate_system_orientation : str
            Orientation of the motor's coordinate system. The coordinate system
            is defined by the motor's axis of symmetry. The origin of the
            coordinate system  may be placed anywhere along such axis, such as
            at the nozzle area, and must be kept the same for all other
            positions specified. Options are "nozzle_to_combustion_chamber" and
            "combustion_chamber_to_nozzle".
        Motor.nozzle_radius : float
            Radius of motor nozzle outlet in meters.
        Motor.nozzle_position : float
            Motor's nozzle outlet position in meters, specified in the motor's
            coordinate system. See `Motor.coordinate_system_orientation` for
            more information.
        Motor.throat_radius : float
            Radius of motor nozzle throat in meters.
        Motor.solid : SolidMotor
            Solid motor object that composes the hybrid motor.
        Motor.liquid : LiquidMotor
            Liquid motor object that composes the hybrid motor.

        Mass and moment of inertia attributes:
        Motor.dry_mass : float
            The total mass of the motor structure, including chambers
            and tanks, when it is empty and does not contain any propellant.
        Motor.propellant_initial_mass : float
            Total propellant initial mass in kg.
        Motor.total_mass : Function
            Total motor mass in kg as a function of time, defined as the sum
            of propellant and dry mass.
        Motor.propellant_mass : Function
            Total propellant mass in kg as a function of time.
        Motor.total_mass_flow_rate : Function
            Time derivative of propellant total mass in kg/s as a function
            of time as obtained by the thrust source.
        Motor.center_of_mass : Function
            Position of the motor center of mass in
            meters as a function of time.
            See `Motor.coordinate_system_orientation` for more information
            regarding the motor's coordinate system.
        Motor.center_of_propellant_mass : Function
            Position of the motor propellant center of mass in meters as a
            function of time.
            See `Motor.coordinate_system_orientation` for more information
            regarding the motor's coordinate system.
        Motor.I_11 : Function
            Component of the motor's inertia tensor relative to the e_1 axis
            in kg*m^2, as a function of time. The e_1 axis is the direction
            perpendicular to the motor body axis of symmetry, centered at
            the instantaneous motor center of mass.
        Motor.I_22 : Function
            Component of the motor's inertia tensor relative to the e_2 axis
            in kg*m^2, as a function of time. The e_2 axis is the direction
            perpendicular to the motor body axis of symmetry, centered at
            the instantaneous motor center of mass.
            Numerically equivalent to I_11 due to symmetry.
        Motor.I_33 : Function
            Component of the motor's inertia tensor relative to the e_3 axis
            in kg*m^2, as a function of time. The e_3 axis is the direction of
            the motor body axis of symmetry, centered at the instantaneous
            motor center of mass.
        Motor.I_12 : Function
            Component of the motor's inertia tensor relative to the e_1 and
            e_2 axes in kg*m^2, as a function of time. See Motor.I_11 and
            Motor.I_22 for more information.
        Motor.I_13 : Function
            Component of the motor's inertia tensor relative to the e_1 and
            e_3 axes in kg*m^2, as a function of time. See Motor.I_11 and
            Motor.I_33 for more information.
        Motor.I_23 : Function
            Component of the motor's inertia tensor relative to the e_2 and
            e_3 axes in kg*m^2, as a function of time. See Motor.I_22 and
            Motor.I_33 for more information.
        Motor.propellant_I_11 : Function
            Component of the propellant inertia tensor relative to the e_1
            axis in kg*m^2, as a function of time. The e_1 axis is the
            direction perpendicular to the motor body axis of symmetry,
            centered at the instantaneous propellant center of mass.
        Motor.propellant_I_22 : Function
            Component of the propellant inertia tensor relative to the e_2
            axis in kg*m^2, as a function of time. The e_2 axis is the
            direction perpendicular to the motor body axis of symmetry,
            centered at the instantaneous propellant center of mass.
            Numerically equivalent to propellant_I_11 due to symmetry.
        Motor.propellant_I_33 : Function
            Component of the propellant inertia tensor relative to the e_3
            axis in kg*m^2, as a function of time. The e_3 axis is the
            direction of the motor body axis of symmetry, centered at the
            instantaneous propellant center of mass.
        Motor.propellant_I_12 : Function
            Component of the propellant inertia tensor relative to the e_1 and
            e_2 axes in kg*m^2, as a function of time. See Motor.propellant_I_11
            and Motor.propellant_I_22 for more information.
        Motor.propellant_I_13 : Function
            Component of the propellant inertia tensor relative to the e_1 and
            e_3 axes in kg*m^2, as a function of time. See Motor.propellant_I_11
            and Motor.propellant_I_33 for more information.
        Motor.propellant_I_23 : Function
            Component of the propellant inertia tensor relative to the e_2 and
            e_3 axes in kg*m^2, as a function of time. See Motor.propellant_I_22
            and Motor.propellant_I_33 for more information.

        Thrust and burn attributes:
        Motor.thrust : Function
            Motor thrust force, in Newtons, as a function of time.
        Motor.total_impulse : float
            Total impulse of the thrust curve in N*s.
        Motor.max_thrust : float
            Maximum thrust value of the given thrust curve, in N.
        Motor.max_thrust_time : float
            Time, in seconds, in which the maximum thrust value is achieved.
        Motor.average_thrust : float
            Average thrust of the motor, given in N.
        Motor.burn_time : tuple of float
            Tuple containing the initial and final time of the motor's burn time
            in seconds.
        Motor.burn_start_time : float
            Motor burn start time, in seconds.
        Motor.burn_out_time : float
            Motor burn out time, in seconds.
        Motor.burn_duration : float
            Total motor burn duration, in seconds. It is the difference between the burn_out_time and the burn_start_time.
        Motor.exhaust_velocity : float
            Propulsion gases exhaust velocity, assumed constant, in m/s.
        Motor.burn_area : Function
            Total burn area considering all grains, made out of inner
            cylindrical burn area and grain top and bottom faces. Expressed
            in meters squared as a function of time.
        Motor.Kn : Function
            Motor Kn as a function of time. Defined as burn_area divided by
            nozzle throat cross sectional area. Has no units.
        Motor.burn_rate : Function
            Propellant burn rate in meter/second as a function of time.
        Motor.interpolate : string
            Method of interpolation used in case thrust curve is given
            by data set in .csv or .eng, or as an array. Options are 'spline'
            'akima' and 'linear'. Default is "linear".
    """

    def __init__(
        self,
        thrust_source,
        dry_mass,
        center_of_dry_mass,
        dry_inertia,
        grains_center_of_mass_position,
        grain_number,
        grain_density,
        grain_outer_radius,
        grain_initial_inner_radius,
        grain_initial_height,
        grain_separation,
        nozzle_radius,
        burn_time=None,
        nozzle_position=0,
        throat_radius=0.01,
        reshape_thrust_curve=False,
        interpolation_method="linear",
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    ):
        """Initialize Motor class, process thrust curve and geometrical
        parameters and store results.

        Parameters
        ----------
        thrust_source : int, float, callable, string, array
            Motor's thrust curve. Can be given as an int or float, in which
            case the thrust will be considered constant in time. It can
            also be given as a callable function, whose argument is time in
            seconds and returns the thrust supplied by the motor in the
            instant. If a string is given, it must point to a .csv or .eng file.
            The .csv file shall contain no headers and the first column must
            specify time in seconds, while the second column specifies thrust.
            Arrays may also be specified, following rules set by the class
            Function. See help(Function). Thrust units are Newtons.
        burn_time: float, tuple of float, optional
            Motor's burn time.
            If a float is given, the burn time is assumed to be between 0 and the
            given float, in seconds.
            If a tuple of float is given, the burn time is assumed to be between
            the first and second elements of the tuple, in seconds.
            If not specified, automatically sourced as the range between the first- and
            last-time step of the motor's thrust curve. This can only be used if the
            motor's thrust is defined by a list of points, such as a .csv file, a .eng
            file or a Function instance whose source is a list.
        dry_mass : int, float
            The total mass of the motor structure, including chambers
            and tanks, when it is empty and does not contain any propellant.
        center_of_dry_mass : int, float
            The position, in meters, of the motor's center of mass with respect
            to the motor's coordinate system when it is devoid of propellant.
            See `Motor.coordinate_system_orientation`.
        dry_inertia : tuple, list
            Tuple or list containing the motor's dry mass inertia tensor
            components, in kg*m^2. This inertia is defined with respect to the
            the `center_of_dry_mass` position.
            Assuming e_3 is the rocket's axis of symmetry, e_1 and e_2 are
            orthogonal and form a plane perpendicular to e_3, the dry mass
            inertia tensor components must be given in the following order:
            (I_11, I_22, I_33, I_12, I_13, I_23), where I_ij is the
            component of the inertia tensor in the direction of e_i x e_j.
            Alternatively, the inertia tensor can be given as (I_11, I_22, I_33),
            where I_12 = I_13 = I_23 = 0.
        grain_number : int
            Number of solid grains
        grain_density : int, float
            Solid grain density in kg/m3.
        grain_outer_radius : int, float
            Solid grain outer radius in meters.
        grain_initial_inner_radius : int, float
            Solid grain initial inner radius in meters.
        grain_initial_height : int, float
            Solid grain initial height in meters.
        grain_separation : int, float
            Distance between grains, in meters.
        nozzle_radius : int, float
            Motor's nozzle outlet radius in meters.
        nozzle_position : int, float, optional
            Motor's nozzle outlet position in meters, in the motor's coordinate
            system. See `Motor.coordinate_system_orientation` for details.
            Default is 0, in which case the origin of the coordinate system
            is placed at the motor's nozzle outlet.
        throat_radius : int, float, optional
            Motor's nozzle throat radius in meters. Used to calculate Kn curve.
            Optional if the Kn curve is not interesting. Its value does not
            impact trajectory simulation.
        reshape_thrust_curve : boolean, tuple, optional
            If False, the original thrust curve supplied is not altered. If a
            tuple is given, whose first parameter is a new burn out time and
            whose second parameter is a new total impulse in Ns, the thrust
            curve is reshaped to match the new specifications. May be useful
            for motors whose thrust curve shape is expected to remain similar
            in case the impulse and burn time varies slightly. Default is
            False.
        interpolation_method : string, optional
            Method of interpolation to be used in case thrust curve is given
            by data set in .csv or .eng, or as an array. Options are 'spline'
            'akima' and 'linear'. Default is "linear".
        coordinate_system_orientation : string, optional
            Orientation of the motor's coordinate system. The coordinate system
            is defined by the motor's axis of symmetry. The origin of the
            coordinate system  may be placed anywhere along such axis, such as
            at the nozzle area, and must be kept the same for all other
            positions specified. Options are "nozzle_to_combustion_chamber" and
            "combustion_chamber_to_nozzle". Default is "nozzle_to_combustion_chamber".

        Returns
        -------
        None
        """
        super().__init__(
            thrust_source,
            dry_mass,
            center_of_dry_mass,
            dry_inertia,
            nozzle_radius,
            burn_time,
            nozzle_position,
            reshape_thrust_curve,
            interpolation_method,
            coordinate_system_orientation,
        )
        self.liquid = LiquidMotor(
            thrust_source,
            dry_mass,
            center_of_dry_mass,
            dry_inertia,
            nozzle_radius,
            burn_time,
            nozzle_position,
            reshape_thrust_curve,
            interpolation_method,
            coordinate_system_orientation,
        )
        self.solid = SolidMotor(
            thrust_source,
            dry_mass,
            center_of_dry_mass,
            dry_inertia,
            grains_center_of_mass_position,
            grain_number,
            grain_density,
            grain_outer_radius,
            grain_initial_inner_radius,
            grain_initial_height,
            grain_separation,
            nozzle_radius,
            burn_time,
            nozzle_position,
            throat_radius,
            reshape_thrust_curve,
            interpolation_method,
            coordinate_system_orientation,
        )
        # Initialize plots and prints object
        self.prints = _HybridMotorPrints(self)
        self.plots = _HybridMotorPlots(self)
        return None

    @funcify_method("Time (s)", "Exhaust velocity (m/s)")
    def exhaust_velocity(self):
        """Exhaust velocity by assuming it as a constant. The formula used is
        total impulse/propellant initial mass.

        Returns
        -------
        self.exhaust_velocity : Function
            Gas exhaust velocity of the motor.
        """
        return self.total_impulse / self.propellant_initial_mass

    @funcify_method("Time (s)", "Mass (kg)")
    def propellant_mass(self):
        """Evaluates the total propellant mass of the motor as the sum
        of each tank mass and the grains mass.

        Returns
        -------
        Function
            Total propellant mass of the motor, in kg.
        """
        return self.solid.propellant_mass + self.liquid.propellant_mass

    @cached_property
    def propellant_initial_mass(self):
        """Returns the initial propellant mass of the motor.

        Returns
        -------
        float
            Initial propellant mass of the motor, in kg.
        """
        return self.solid.propellant_initial_mass + self.liquid.propellant_initial_mass

    @funcify_method("Time (s)", "mass flow rate (kg/s)", extrapolation="zero")
    def mass_flow_rate(self):
        """Evaluates the mass flow rate of the motor as the sum of each tank
        mass flow rate and the grains mass flow rate.

        Returns
        -------
        Function
            Mass flow rate of the motor, in kg/s.

        See Also
        --------
        `Motor.total_mass_flow_rate` :
            Calculates the total mass flow rate of the motor assuming
            constant exhaust velocity.
        """
        return self.solid.mass_flow_rate + self.liquid.mass_flow_rate

    @funcify_method("Time (s)", "center of mass (m)")
    def center_of_propellant_mass(self):
        """Position of the propellant center of mass as a function of time.
        The position is specified as a scalar, relative to the motor's
        coordinate system.

        Returns
        -------
        Function
            Position of the center of mass as a function of time.
        """
        massBalance = (
            self.solid.propellant_mass * self.solid.center_of_propellant_mass
            + self.liquid.propellant_mass * self.liquid.center_of_propellant_mass
        )
        return massBalance / self.propellant_mass

    @funcify_method("Time (s)", "Inertia I_11 (kg m²)")
    def propellant_I_11(self):
        """Inertia tensor 11 component of the propellant, the inertia is
        relative to the e_1 axis, centered at the instantaneous propellant
        center of mass.

        Returns
        -------
        Function
            Propellant inertia tensor 11 component at time t.

        Notes
        -----
        The e_1 direction is assumed to be the direction perpendicular to the
        motor body axis.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor
        """
        solidCorrection = (
            self.solid.propellant_mass
            * (self.solid.center_of_propellant_mass - self.center_of_mass) ** 2
        )
        liquidCorrection = (
            self.liquid.propellant_mass
            * (self.liquid.center_of_propellant_mass - self.center_of_mass) ** 2
        )

        I_11 = self.solid.I_11 + solidCorrection + self.liquid.I_11 + liquidCorrection
        return I_11

    @funcify_method("Time (s)", "Inertia I_22 (kg m²)")
    def propellant_I_22(self):
        """Inertia tensor 22 component of the propellant, the inertia is
        relative to the e_2 axis, centered at the instantaneous propellant
        center of mass.

        Returns
        -------
        Function
            Propellant inertia tensor 22 component at time t.

        Notes
        -----
        The e_2 direction is assumed to be the direction perpendicular to the
        motor body axis, and perpendicular to e_1.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor
        """
        return self.I_11

    @funcify_method("Time (s)", "Inertia I_33 (kg m²)")
    def propellant_I_33(self):
        """Inertia tensor 33 component of the propellant, the inertia is
        relative to the e_3 axis, centered at the instantaneous propellant
        center of mass.

        Returns
        -------
        Function
            Propellant inertia tensor 33 component at time t.

        Notes
        -----
        The e_3 direction is assumed to be the axial direction of the rocket
        motor.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor
        """
        return self.solid.propellant_I_33 + self.liquid.propellant_I_33

    @funcify_method("Time (s)", "Inertia I_12 (kg m²)")
    def propellant_I_12(self):
        return 0

    @funcify_method("Time (s)", "Inertia I_13 (kg m²)")
    def propellant_I_13(self):
        return 0

    @funcify_method("Time (s)", "Inertia I_23 (kg m²)")
    def propellant_I_23(self):
        return 0

    def add_tank(self, tank, position):
        """Adds a tank to the motor.

        Parameters
        ----------
        tank : Tank
            Tank object to be added to the motor.
        position : float
            Position of the tank relative to the nozzle exit. The
            tank reference point is its tank_geometry zero point.

        Returns
        -------
        None
        """
        self.liquid.add_tank(tank, position)
        self.solid.mass_flow_rate = (
            self.total_mass_flow_rate - self.liquid.mass_flow_rate
        )
        reset_funcified_methods(self)

    def info(self):
        """Prints out basic data about the Motor."""
        self.prints.all()
        self.plots.thrust()
        return None

    def all_info(self):
        """Prints out all data and graphs available about the Motor.

        Return
        ------
        None
        """
        self.prints.all()
        self.plots.all()
        return None
        return None
