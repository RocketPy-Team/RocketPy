# -*- coding: utf-8 -*-

__author__ = "Giovani Hidalgo Ceotto, Pedro Henrique Marinho Bressan, Mateus Stano Junqueira, Oscar Mauricio Prada Ramirez, João Lemes Gribel Soares, Lucas Kierulff Balabram, Lucas Azevedo Pezente"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

import numpy as np
from scipy import integrate

from rocketpy.plots.solid_motor_plots import _SolidMotorPlots
from rocketpy.prints.solid_motor_prints import _SolidMotorPrints

try:
    from functools import cached_property
except ImportError:
    from rocketpy.tools import cached_property

from rocketpy.Function import Function, funcify_method, reset_funcified_methods

from .Motor import Motor


class SolidMotor(Motor):
    """Class to specify characteristics and useful operations for solid motors.

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
        Motor.grain_number : int
            Number of solid grains.
        Motor.grains_center_of_mass_position : float
            Position of the center of mass of the grains in meters, specified in
            the motor's coordinate system.
            See `Motor.coordinate_system_orientation` for more information.
        Motor.grain_separation : float
            Distance between two grains in meters.
        Motor.grain_density : float
            Density of each grain in kg/meters cubed.
        Motor.grain_outer_radius : float
            Outer radius of each grain in meters.
        Motor.grain_initial_inner_radius : float
            Initial inner radius of each grain in meters.
        Motor.grain_initial_height : float
            Initial height of each grain in meters.
        Motor.grainInitialVolume : float
            Initial volume of each grain in meters cubed.
        Motor.grain_inner_radius : Function
            Inner radius of each grain in meters as a function of time.
        Motor.grain_height : Function
            Height of each grain in meters as a function of time.

        Mass and moment of inertia attributes:
        Motor.grainInitialMass : float
            Initial mass of each grain in kg.
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
        grains_center_of_mass_position : float
            Position of the center of mass of the grains in meters. More specifically,
            the coordinate of the center of mass specified in the motor's coordinate
            system. See `Motor.coordinate_system_orientation` for more information.
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
        # Nozzle parameters
        self.throat_radius = throat_radius
        self.throatArea = np.pi * throat_radius**2

        # Grain parameters
        self.grains_center_of_mass_position = grains_center_of_mass_position
        self.grain_number = grain_number
        self.grain_separation = grain_separation
        self.grain_density = grain_density
        self.grain_outer_radius = grain_outer_radius
        self.grain_initial_inner_radius = grain_initial_inner_radius
        self.grain_initial_height = grain_initial_height

        # Grains initial geometrical parameters
        self.grainInitialVolume = (
            self.grain_initial_height
            * np.pi
            * (self.grain_outer_radius**2 - self.grain_initial_inner_radius**2)
        )
        self.grainInitialMass = self.grain_density * self.grainInitialVolume

        self.evaluate_geometry()

        # Initialize plots and prints object
        self.prints = _SolidMotorPrints(self)
        self.plots = _SolidMotorPlots(self)
        return None

    @funcify_method("Time (s)", "Mass (kg)")
    def propellant_mass(self):
        """Evaluates the total propellant mass as a function of time.

        Returns
        -------
        Function
            Mass of the motor, in kg.
        """
        return self.grain_volume * self.grain_density * self.grain_number

    @funcify_method("Time (s)", "Grain volume (m³)")
    def grain_volume(self):
        """Evaluates the total propellant volume as a function of time. The
        propellant is assumed to be a cylindrical Bates grain under uniform
        burn.

        Returns
        -------
        Function
            Propellant volume as a function of time.
        """
        cross_section_area = np.pi * (
            self.grain_outer_radius**2 - self.grain_inner_radius**2
        )
        return cross_section_area * self.grain_height

    @funcify_method("Time (s)", "Exhaust velocity (m/s)")
    def exhaust_velocity(self):
        """Exhaust velocity by assuming it as a constant. The formula used is
        total impulse/propellant initial mass.

        Returns
        -------
        self.exhaust_velocity : rocketpy.Function
            Gas exhaust velocity of the motor.
        """
        return self.total_impulse / self.propellant_initial_mass

    @property
    def propellant_initial_mass(self):
        """Returns the initial propellant mass.

        Returns
        -------
        float
            Initial propellant mass in kg.
        """
        return self.grain_number * self.grainInitialMass

    @property
    def mass_flow_rate(self):
        """Time derivative of propellant mass. Assumes constant exhaust
        velocity. The formula used is the opposite of thrust divided by
        exhaust velocity.

        Returns
        -------
        self.mass_flow_rate : Function
            Time derivative of total propellant mass as a function of time.

        See Also
        --------
        `Motor.total_mass_flow_rate` :
            Calculates the total mass flow rate of the motor assuming
            constant exhaust velocity.
        """
        try:
            return self._massFlowRate
        except AttributeError:
            self._massFlowRate = self.total_mass_flow_rate
            return self._massFlowRate

    @mass_flow_rate.setter
    def mass_flow_rate(self, value):
        """Sets the mass flow rate of the motor.

        Parameters
        ----------
        value : Function
            Mass flow rate in kg/s.

        Returns
        -------
        None
        """
        self._massFlowRate = value.reset("Time (s)", "grain mass flow rate (kg/s)")
        self.evaluate_geometry()

    @funcify_method("Time (s)", "center of mass (m)", "linear")
    def center_of_propellant_mass(self):
        """Position of the propellant center of mass as a function of time.
        The position is specified as a scalar, relative to the motor's
        coordinate system.

        Returns
        -------
        rocketpy.Function
            Position of the propellant center of mass as a function of time.
        """
        timeSource = self.grain_inner_radius.x_array
        center_of_mass = np.full_like(timeSource, self.grains_center_of_mass_position)
        return np.column_stack((timeSource, center_of_mass))

    def evaluate_geometry(self):
        """Calculates grain inner radius and grain height as a function of time
        by assuming that every propellant mass burnt is exhausted. In order to
        do that, a system of differential equations is solved using
        scipy.integrate.odeint. Furthermore, the function calculates burn area,
        burn rate and Kn as a function of time using the previous results. All
        functions are stored as objects of the class Function in
        self.grain_inner_radius, self.grain_height, self.burn_area, self.burn_rate
        and self.Kn.


        Returns
        -------
        geometry : list of rocketpy.Functions
            First element is the Function representing the inner radius of a
            grain as a function of time. Second argument is the Function
            representing the height of a grain as a function of time.
        """
        # Define initial conditions for integration
        y0 = [self.grain_initial_inner_radius, self.grain_initial_height]

        # Define time mesh
        t = self.thrust.source[:, 0]
        t_span = t[0], t[-1]

        density = self.grain_density
        rO = self.grain_outer_radius

        # Define system of differential equations
        def geometryDot(t, y):
            grainMassDot = self.mass_flow_rate(t) / self.grain_number
            rI, h = y
            rIDot = (
                -0.5 * grainMassDot / (density * np.pi * (rO**2 - rI**2 + rI * h))
            )
            hDot = 1.0 * grainMassDot / (density * np.pi * (rO**2 - rI**2 + rI * h))
            return [rIDot, hDot]

        def terminateBurn(t, y):
            end_function = (self.grain_outer_radius - y[0]) * y[1]
            return end_function

        terminateBurn.terminal = True
        terminateBurn.direction = -1

        # Solve the system of differential equations
        sol = integrate.solve_ivp(
            geometryDot,
            t_span,
            y0,
            events=terminateBurn,
            atol=1e-12,
            rtol=1e-11,
            method="LSODA",
        )

        self.grainBurnOut = sol.t[-1]

        # Write down functions for innerRadius and height
        self.grain_inner_radius = Function(
            np.concatenate(([sol.t], [sol.y[0]])).transpose().tolist(),
            "Time (s)",
            "Grain Inner Radius (m)",
            self.interpolate,
            "constant",
        )
        self.grain_height = Function(
            np.concatenate(([sol.t], [sol.y[1]])).transpose().tolist(),
            "Time (s)",
            "Grain Height (m)",
            self.interpolate,
            "constant",
        )

        reset_funcified_methods(self)

        return [self.grain_inner_radius, self.grain_height]

    @funcify_method("Time (s)", "burn area (m²)")
    def burn_area(self):
        """Calculates the BurnArea of the grain for each time. Assuming that
        the grains are cylindrical BATES grains.

        Returns
        -------
        burn_area : rocketpy.Function
            Function representing the burn area progression with the time.
        """
        burn_area = (
            2
            * np.pi
            * (
                self.grain_outer_radius**2
                - self.grain_inner_radius**2
                + self.grain_inner_radius * self.grain_height
            )
            * self.grain_number
        )
        return burn_area

    @funcify_method("Time (s)", "burn rate (m/s)")
    def burn_rate(self):
        """Calculates the BurnRate with respect to time. This evaluation
        assumes that it was already calculated the massDot, burn_area time
        series.

        Returns
        -------
        burn_rate : rocketpy.Function
            Rate of progression of the inner radius during the combustion.
        """
        return -1 * self.mass_flow_rate / (self.burn_area * self.grain_density)

    @cached_property
    def Kn(self):
        """Calculates the motor Kn as a function of time. Defined as burn_area
        divided by the nozzle throat cross sectional area.

        Returns
        -------
        Kn : rocketpy.Function
            Kn as a function of time.
        """
        KnSource = (
            np.concatenate(
                (
                    [self.grain_inner_radius.source[:, 1]],
                    [self.burn_area.source[:, 1] / self.throatArea],
                )
            ).transpose()
        ).tolist()
        Kn = Function(
            KnSource,
            "Grain Inner Radius (m)",
            "Kn (m2/m2)",
            self.interpolate,
            "constant",
        )
        return Kn

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
        grainMass = self.propellant_mass / self.grain_number
        grain_number = self.grain_number
        grainInertia11 = grainMass * (
            (1 / 4) * (self.grain_outer_radius**2 + self.grain_inner_radius**2)
            + (1 / 12) * self.grain_height**2
        )

        # Calculate each grain's distance d to propellant center of mass
        # Assuming each grain's COM are evenly spaced
        initialValue = (grain_number - 1) / 2
        d = np.linspace(-initialValue, initialValue, grain_number)
        d = d * (self.grain_initial_height + self.grain_separation)

        # Calculate inertia for all grains
        I_11 = grain_number * grainInertia11 + grainMass * np.sum(d**2)

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
        I_33 = (
            (1 / 2.0)
            * self.propellant_mass
            * (self.grain_outer_radius**2 + self.grain_inner_radius**2)
        )
        return I_33

    @funcify_method("Time (s)", "Inertia I_12 (kg m²)")
    def propellant_I_12(self):
        return 0

    @funcify_method("Time (s)", "Inertia I_13 (kg m²)")
    def propellant_I_13(self):
        return 0

    @funcify_method("Time (s)", "Inertia I_23 (kg m²)")
    def propellant_I_23(self):
        return 0

    def info(self):
        """Prints out basic data about the Motor."""
        self.prints.all()
        self.plots.thrust()
        return None

    def all_info(self):
        """Prints out all data and graphs available about the Motor."""
        self.prints.all()
        self.plots.all()

        return None
