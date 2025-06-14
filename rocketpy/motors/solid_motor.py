from functools import cached_property

import numpy as np
from scipy import integrate

from ..mathutils.function import Function, funcify_method, reset_funcified_methods
from ..plots.solid_motor_plots import _SolidMotorPlots
from ..prints.solid_motor_prints import _SolidMotorPrints
from .motor import Motor


class SolidMotor(Motor):
    """Class to specify characteristics and useful operations for solid motors.

    Inherits from the abstract class rocketpy.Motor.

    See Also
    --------
    Motor

    Attributes
    ----------
    SolidMotor.coordinate_system_orientation : str
        Orientation of the motor's coordinate system. The coordinate system
        is defined by the motor's axis of symmetry. The origin of the
        coordinate system may be placed anywhere along such axis, such as
        at the nozzle area, and must be kept the same for all other
        positions specified. Options are "nozzle_to_combustion_chamber" and
        "combustion_chamber_to_nozzle".
    SolidMotor.nozzle_radius : float
        Radius of motor nozzle outlet in meters.
    SolidMotor.nozzle_area : float
        Area of motor nozzle outlet in square meters.
    SolidMotor.nozzle_position : float
        Motor's nozzle outlet position in meters, specified in the motor's
        coordinate system. See
        :doc:`Positions and Coordinate Systems </user/positions>` for
        more information.
    SolidMotor.throat_radius : float
        Radius of motor nozzle throat in meters.
    SolidMotor.grain_number : int
        Number of solid grains.
    SolidMotor.grains_center_of_mass_position : float
        Position of the center of mass of the grains in meters, specified in
        the motor's coordinate system.
        See :doc:`Positions and Coordinate Systems </user/positions>`
        for more information.
    SolidMotor.grain_separation : float
        Distance between two grains in meters.
    SolidMotor.grain_density : float
        Density of each grain in kg/meters cubed.
    SolidMotor.grain_outer_radius : float
        Outer radius of each grain in meters.
    SolidMotor.grain_initial_inner_radius : float
        Initial inner radius of each grain in meters.
    SolidMotor.grain_initial_height : float
        Initial height of each grain in meters.
    SolidMotor.grain_initial_volume : float
        Initial volume of each grain in meters cubed.
    SolidMotor.grain_inner_radius : Function
        Inner radius of each grain in meters as a function of time.
    SolidMotor.grain_height : Function
        Height of each grain in meters as a function of time.
    SolidMotor.grain_initial_mass : float
        Initial mass of each grain in kg.
    SolidMotor.dry_mass : float
        Same as in Motor class. See the :class:`Motor <rocketpy.Motor>` docs.
    SolidMotor.propellant_initial_mass : float
        Total propellant initial mass in kg.
    SolidMotor.total_mass : Function
        Total motor mass in kg as a function of time, defined as the sum
        of propellant and dry mass.
    SolidMotor.propellant_mass : Function
        Total propellant mass in kg as a function of time.
    SolidMotor.structural_mass_ratio: float
        Initial ratio between the dry mass and the total mass.
    SolidMotor.total_mass_flow_rate : Function
        Time derivative of propellant total mass in kg/s as a function
        of time as obtained by the thrust source.
    SolidMotor.center_of_mass : Function
        Position of the motor center of mass in meters as a function of time,
        with respect to the motor's coordinate system.
        See
        :doc:`Positions and Coordinate Systems </user/positions>` for more
        information regarding the motor's coordinate system.
    SolidMotor.center_of_propellant_mass : Function
        Position of the motor propellant center of mass in meters as a
        function of time.
        See
        :doc:`Positions and Coordinate Systems </user/positions>` for more
        information regarding the motor's coordinate system.
    SolidMotor.I_11 : Function
        Component of the motor's inertia tensor relative to the e_1 axis
        in kg*m^2, as a function of time. The e_1 axis is the direction
        perpendicular to the motor body axis of symmetry, centered at
        the instantaneous motor center of mass.
    SolidMotor.I_22 : Function
        Component of the motor's inertia tensor relative to the e_2 axis
        in kg*m^2, as a function of time. The e_2 axis is the direction
        perpendicular to the motor body axis of symmetry, centered at
        the instantaneous motor center of mass.
        Numerically equivalent to I_11 due to symmetry.
    SolidMotor.I_33 : Function
        Component of the motor's inertia tensor relative to the e_3 axis
        in kg*m^2, as a function of time. The e_3 axis is the direction of
        the motor body axis of symmetry, centered at the instantaneous
        motor center of mass.
    SolidMotor.I_12 : Function
        Component of the motor's inertia tensor relative to the e_1 and
        e_2 axes in kg*m^2, as a function of time. See SolidMotor.I_11 and
        SolidMotor.I_22 for more information.
    SolidMotor.I_13 : Function
        Component of the motor's inertia tensor relative to the e_1 and
        e_3 axes in kg*m^2, as a function of time. See SolidMotor.I_11 and
        SolidMotor.I_33 for more information.
    SolidMotor.I_23 : Function
        Component of the motor's inertia tensor relative to the e_2 and
        e_3 axes in kg*m^2, as a function of time. See SolidMotor.I_22 and
        SolidMotor.I_33 for more information.
    SolidMotor.propellant_I_11 : Function
        Component of the propellant inertia tensor relative to the e_1
        axis in kg*m^2, as a function of time. The e_1 axis is the
        direction perpendicular to the motor body axis of symmetry,
        centered at the instantaneous propellant center of mass.
    SolidMotor.propellant_I_22 : Function
        Component of the propellant inertia tensor relative to the e_2
        axis in kg*m^2, as a function of time. The e_2 axis is the
        direction perpendicular to the motor body axis of symmetry,
        centered at the instantaneous propellant center of mass.
        Numerically equivalent to propellant_I_11 due to symmetry.
    SolidMotor.propellant_I_33 : Function
        Component of the propellant inertia tensor relative to the e_3
        axis in kg*m^2, as a function of time. The e_3 axis is the
        direction of the motor body axis of symmetry, centered at the
        instantaneous propellant center of mass.
    SolidMotor.propellant_I_12 : Function
        Component of the propellant inertia tensor relative to the e_1 and
        e_2 axes in kg*m^2, as a function of time.
        See SolidMotor.propellant_I_11 and SolidMotor.propellant_I_22 for
        more information.
    SolidMotor.propellant_I_13 : Function
        Component of the propellant inertia tensor relative to the e_1 and
        e_3 axes in kg*m^2, as a function of time.
        See SolidMotor.propellant_I_11 and SolidMotor.propellant_I_33 for
        more information.
    SolidMotor.propellant_I_23 : Function
        Component of the propellant inertia tensor relative to the e_2 and
        e_3 axes in kg*m^2, as a function of time.
        See SolidMotor.propellant_I_22 and SolidMotor.propellant_I_33 for more
        information.
    SolidMotor.thrust : Function
        Motor thrust force obtained from thrust source, in Newtons, as a
        function of time.
    SolidMotor.vacuum_thrust : Function
        Motor thrust force when the rocket is in a vacuum. In Newtons, as a
        function of time.
    SolidMotor.total_impulse : float
        Total impulse of the thrust curve in N*s.
    SolidMotor.max_thrust : float
        Maximum thrust value of the given thrust curve, in N.
    SolidMotor.max_thrust_time : float
        Time, in seconds, in which the maximum thrust value is achieved.
    SolidMotor.average_thrust : float
        Average thrust of the motor, given in N.
    SolidMotor.burn_time : tuple of float
        Tuple containing the initial and final time of the motor's burn time
        in seconds.
    SolidMotor.burn_start_time : float
        Motor burn start time, in seconds.
    SolidMotor.burn_out_time : float
        Motor burn out time, in seconds.
    SolidMotor.burn_duration : float
        Total motor burn duration, in seconds. It is the difference between the
        ``burn_out_time`` and the ``burn_start_time``.
    SolidMotor.exhaust_velocity : Function
        Effective exhaust velocity of the propulsion gases in m/s. Computed
        as the thrust divided by the mass flow rate. This corresponds to the
        actual exhaust velocity only when the nozzle exit pressure equals the
        atmospheric pressure.
    SolidMotor.burn_area : Function
        Total burn area considering all grains, made out of inner
        cylindrical burn area and grain top and bottom faces. Expressed
        in meters squared as a function of time.
    SolidMotor.Kn : Function
        Motor Kn as a function of time. Defined as burn_area divided by
        nozzle throat cross sectional area. Has no units.
    SolidMotor.burn_rate : Function
        Propellant burn rate in meter/second as a function of time.
    SolidMotor.interpolate : string
        Method of interpolation used in case thrust curve is given
        by data set in .csv or .eng, or as an array. Options are 'spline'
        'akima' and 'linear'. Default is "linear".
    SolidMotor.reference_pressure : int, float
        Atmospheric pressure in Pa at which the thrust data was recorded.
        It will allow to obtain the net thrust in the Flight class.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        thrust_source,
        dry_mass,
        dry_inertia,
        nozzle_radius,
        grain_number,
        grain_density,
        grain_outer_radius,
        grain_initial_inner_radius,
        grain_initial_height,
        grain_separation,
        grains_center_of_mass_position,
        center_of_dry_mass_position,
        nozzle_position=0.0,
        burn_time=None,
        throat_radius=0.01,
        reshape_thrust_curve=False,
        interpolation_method="linear",
        coordinate_system_orientation="nozzle_to_combustion_chamber",
        reference_pressure=None,
    ):
        """Initialize Motor class, process thrust curve and geometrical
        parameters and store results.

        Parameters
        ----------
        thrust_source : int, float, callable, string, array, Function
            Motor's thrust curve. Can be given as an int or float, in which
            case the thrust will be considered constant in time. It can
            also be given as a callable function, whose argument is time in
            seconds and returns the thrust supplied by the motor in the
            instant. If a string is given, it must point to a .csv or .eng file.
            The .csv file can contain a single line header and the first column
            must specify time in seconds, while the second column specifies
            thrust. Arrays may also be specified, following rules set by the
            class Function. Thrust units are Newtons.

            .. seealso:: :doc:`Thrust Source Details </user/motors/thrust>`
        nozzle_radius : int, float
            Motor's nozzle outlet radius in meters.
        dry_mass : int, float
            Same as in Motor class. See the :class:`Motor <rocketpy.Motor>` docs
        dry_inertia : tuple, list
            Tuple or list containing the motor's dry mass inertia tensor
            components, in kg*m^2. This inertia is defined with respect to the
            the `center_of_dry_mass_position` position.
            Assuming e_3 is the rocket's axis of symmetry, e_1 and e_2 are
            orthogonal and form a plane perpendicular to e_3, the dry mass
            inertia tensor components must be given in the following order:
            (I_11, I_22, I_33, I_12, I_13, I_23), where I_ij is the
            component of the inertia tensor in the direction of e_i x e_j.
            Alternatively, the inertia tensor can be given as
            (I_11, I_22, I_33), where I_12 = I_13 = I_23 = 0.
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
        grains_center_of_mass_position : float
            Position of the center of mass of the grains in meters. More
            specifically, the coordinate of the center of mass specified in the
            motor's coordinate system.
            See
            :doc:`Positions and Coordinate Systems </user/positions>`
            for more information.
        center_of_dry_mass_position : int, float
            The position, in meters, of the motor's center of mass with respect
            to the motor's coordinate system when it is devoid of propellant.
            See :doc:`Positions and Coordinate Systems </user/positions>`.
        nozzle_position : int, float, optional
            Motor's nozzle outlet position in meters, in the motor's coordinate
            system. See :doc:`Positions and Coordinate Systems </user/positions>`
            for details. Default is 0, in which case the origin of the
            coordinate system is placed at the motor's nozzle outlet.
        burn_time: float, tuple of float, optional
            Motor's burn time.
            If a float is given, the burn time is assumed to be between 0 and
            the given float, in seconds. If a tuple of float is given, the burn
            time is assumed to be between the first and second elements of the
            tuple, in seconds. If not specified, automatically sourced as the
            range between the first- and last-time step of the motor's thrust
            curve. This can only be used if the motor's thrust is defined by a
            list of points, such as a .csv file, a .eng file or a Function
            instance whose source is a list.
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
            coordinate system may be placed anywhere along such axis, such as
            at the nozzle area, and must be kept the same for all other
            positions specified. Options are "nozzle_to_combustion_chamber" and
            "combustion_chamber_to_nozzle". Default is
            "nozzle_to_combustion_chamber".
        reference_pressure : int, float, optional
            Atmospheric pressure in Pa at which the thrust data was recorded.

        Returns
        -------
        None
        """
        super().__init__(
            thrust_source=thrust_source,
            dry_inertia=dry_inertia,
            nozzle_radius=nozzle_radius,
            center_of_dry_mass_position=center_of_dry_mass_position,
            dry_mass=dry_mass,
            nozzle_position=nozzle_position,
            burn_time=burn_time,
            reshape_thrust_curve=reshape_thrust_curve,
            interpolation_method=interpolation_method,
            coordinate_system_orientation=coordinate_system_orientation,
            reference_pressure=reference_pressure,
        )
        # Nozzle parameters
        self.throat_radius = throat_radius
        self.throat_area = np.pi * throat_radius**2

        # Grain parameters
        self.grains_center_of_mass_position = grains_center_of_mass_position
        self.grain_number = grain_number
        self.grain_separation = grain_separation
        self.grain_density = grain_density
        self.grain_outer_radius = grain_outer_radius
        self.grain_initial_inner_radius = grain_initial_inner_radius
        self.grain_initial_height = grain_initial_height

        # Grains initial geometrical parameters
        self.grain_initial_volume = (
            self.grain_initial_height
            * np.pi
            * (self.grain_outer_radius**2 - self.grain_initial_inner_radius**2)
        )
        self.grain_initial_mass = self.grain_density * self.grain_initial_volume

        self.evaluate_geometry()

        # Initialize plots and prints object
        self.prints = _SolidMotorPrints(self)
        self.plots = _SolidMotorPlots(self)

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
        self.exhaust_velocity : Function
            Gas exhaust velocity of the motor.

        Notes
        -----
        This corresponds to the actual exhaust velocity only when the nozzle
        exit pressure equals the atmospheric pressure.
        """
        return Function(
            self.total_impulse / self.propellant_initial_mass
        ).set_discrete_based_on_model(self.thrust)

    @property
    def propellant_initial_mass(self):
        """Returns the initial propellant mass.

        Returns
        -------
        float
            Initial propellant mass in kg.
        """
        return self.grain_number * self.grain_initial_mass

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
        Motor.total_mass_flow_rate :
            Calculates the total mass flow rate of the motor assuming
            constant exhaust velocity.
        """
        try:
            return self._mass_flow_rate
        except AttributeError:
            self._mass_flow_rate = self.total_mass_flow_rate
            return self._mass_flow_rate

    @mass_flow_rate.setter
    def mass_flow_rate(self, value):
        """Sets the mass flow rate of the motor. This includes all the grains
        burning all at once.

        Parameters
        ----------
        value : Function
            Mass flow rate in kg/s.

        Returns
        -------
        None
        """
        self._mass_flow_rate = value.reset("Time (s)", "Grain mass flow rate (kg/s)")
        self.evaluate_geometry()

    @funcify_method("Time (s)", "Center of Propellant Mass (m)", "linear")
    def center_of_propellant_mass(self):
        """Position of the propellant center of mass as a function of time.
        The position is specified as a scalar, relative to the motor's
        coordinate system.

        Returns
        -------
        Function
            Position of the propellant center of mass as a function of time.
        """
        time_source = self.grain_inner_radius.x_array
        center_of_mass = np.full_like(time_source, self.grains_center_of_mass_position)
        return np.column_stack((time_source, center_of_mass))

    # pylint: disable=too-many-arguments, too-many-statements
    def evaluate_geometry(self):
        """Calculates grain inner radius and grain height as a function of time
        by assuming that every propellant mass burnt is exhausted. In order to
        do that, a system of differential equations is solved using
        scipy.integrate.solve_ivp.

        Returns
        -------
        None
        """
        # Define initial conditions for integration
        y0 = [self.grain_initial_inner_radius, self.grain_initial_height]

        # Define time mesh
        t = self.thrust.source[:, 0]
        t_span = t[0], t[-1]

        density = self.grain_density
        grain_outer_radius = self.grain_outer_radius
        n_grain = self.grain_number

        # Define system of differential equations
        def geometry_dot(t, y):
            # Store physical parameters
            volume_diff = self.mass_flow_rate(t) / (n_grain * density)

            # Compute state vector derivative
            grain_inner_radius, grain_height = y
            burn_area = (
                2
                * np.pi
                * (
                    grain_outer_radius**2
                    - grain_inner_radius**2
                    + grain_inner_radius * grain_height
                )
            )
            grain_inner_radius_derivative = -volume_diff / burn_area
            grain_height_derivative = -2 * grain_inner_radius_derivative

            return [grain_inner_radius_derivative, grain_height_derivative]

        # Define jacobian of the system of differential equations
        def geometry_jacobian(t, y):
            # Store physical parameters
            volume_diff = self.mass_flow_rate(t) / (n_grain * density)

            # Compute jacobian
            grain_inner_radius, grain_height = y
            factor = volume_diff / (
                2
                * np.pi
                * (
                    grain_outer_radius**2
                    - grain_inner_radius**2
                    + grain_inner_radius * grain_height
                )
                ** 2
            )
            inner_radius_derivative_wrt_inner_radius = factor * (
                grain_height - 2 * grain_inner_radius
            )
            inner_radius_derivative_wrt_height = factor * grain_inner_radius
            height_derivative_wrt_inner_radius = (
                -2 * inner_radius_derivative_wrt_inner_radius
            )
            height_derivative_wrt_height = -2 * inner_radius_derivative_wrt_height

            return [
                [
                    inner_radius_derivative_wrt_inner_radius,
                    inner_radius_derivative_wrt_height,
                ],
                [height_derivative_wrt_inner_radius, height_derivative_wrt_height],
            ]

        def terminate_burn(t, y):  # pylint: disable=unused-argument
            end_function = (self.grain_outer_radius - y[0]) * y[1]
            return end_function

        terminate_burn.terminal = True
        terminate_burn.direction = -1

        # Solve the system of differential equations
        sol = integrate.solve_ivp(
            geometry_dot,
            t_span,
            y0,
            jac=geometry_jacobian,
            events=terminate_burn,
            atol=1e-12,
            rtol=1e-11,
            method="LSODA",
        )

        self.grain_burn_out = sol.t[-1]

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

    @funcify_method("Time (s)", "burn area (m²)")
    def burn_area(self):
        """Calculates the BurnArea of the grain for each time. Assuming that
        the grains are cylindrical BATES grains.

        Returns
        -------
        burn_area : Function
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
        """Calculates the burn_rate with respect to time. This evaluation
        assumes that it was already calculated the mass_dot, burn_area time
        series.

        Returns
        -------
        burn_rate : Function
            Rate of progression of the inner radius during the combustion.
        """
        return -1 * self.mass_flow_rate / (self.burn_area * self.grain_density)

    @cached_property
    def Kn(self):
        """Calculates the motor Kn as a function of time. Defined as burn_area
        divided by the nozzle throat cross sectional area.

        Returns
        -------
        Kn : Function
            Kn as a function of time.
        """
        Kn_source = (
            np.concatenate(
                (
                    [self.grain_inner_radius.source[:, 1]],
                    [self.burn_area.source[:, 1] / self.throat_area],
                )
            ).transpose()
        ).tolist()
        Kn = Function(
            Kn_source,
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

        See Also
        --------
        https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor
        """
        grain_mass = self.propellant_mass / self.grain_number
        grain_number = self.grain_number
        grain_inertia11 = grain_mass * (
            (1 / 4) * (self.grain_outer_radius**2 + self.grain_inner_radius**2)
            + (1 / 12) * self.grain_height**2
        )

        # Calculate each grain's distance d to propellant center of mass
        # Assuming each grain's COM are evenly spaced
        initial_value = (grain_number - 1) / 2
        d = np.linspace(-initial_value, initial_value, grain_number)
        d = d * (self.grain_initial_height + self.grain_separation)

        # Calculate inertia for all grains
        I_11 = grain_number * grain_inertia11 + grain_mass * np.sum(d**2)

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

        See Also
        --------
        https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor
        """
        return self.propellant_I_11

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

        See Also
        --------
        https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor
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

    def draw(self, *, filename=None):
        """Draw a representation of the SolidMotor.

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
        self.plots.draw(filename=filename)

    def to_dict(self, **kwargs):
        data = super().to_dict(**kwargs)
        data.update(
            {
                "nozzle_radius": self.nozzle_radius,
                "throat_radius": self.throat_radius,
                "grain_number": self.grain_number,
                "grain_density": self.grain_density,
                "grain_outer_radius": self.grain_outer_radius,
                "grain_initial_inner_radius": self.grain_initial_inner_radius,
                "grain_initial_height": self.grain_initial_height,
                "grain_separation": self.grain_separation,
                "grains_center_of_mass_position": self.grains_center_of_mass_position,
            }
        )

        if kwargs.get("include_outputs", False):
            burn_rate = self.burn_rate
            if kwargs.get("discretize", False):
                burn_rate = burn_rate.set_discrete_based_on_model(
                    self.thrust, mutate_self=False
                )
            data.update(
                {
                    "grain_inner_radius": self.grain_inner_radius,
                    "grain_height": self.grain_height,
                    "burn_area": self.burn_area,
                    "burn_rate": burn_rate,
                    "Kn": self.Kn,
                }
            )

        return data

    @classmethod
    def from_dict(cls, data):
        return cls(
            thrust_source=data["thrust_source"],
            dry_mass=data["dry_mass"],
            dry_inertia=(
                data["dry_I_11"],
                data["dry_I_22"],
                data["dry_I_33"],
                data["dry_I_12"],
                data["dry_I_13"],
                data["dry_I_23"],
            ),
            nozzle_radius=data["nozzle_radius"],
            grain_number=data["grain_number"],
            grain_density=data["grain_density"],
            grain_outer_radius=data["grain_outer_radius"],
            grain_initial_inner_radius=data["grain_initial_inner_radius"],
            grain_initial_height=data["grain_initial_height"],
            grain_separation=data["grain_separation"],
            grains_center_of_mass_position=data["grains_center_of_mass_position"],
            center_of_dry_mass_position=data["center_of_dry_mass_position"],
            nozzle_position=data["nozzle_position"],
            burn_time=data["burn_time"],
            throat_radius=data["throat_radius"],
            interpolation_method=data["interpolate"],
            coordinate_system_orientation=data["coordinate_system_orientation"],
            reference_pressure=data.get("reference_pressure"),
        )
