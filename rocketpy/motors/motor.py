import re
import warnings
from abc import ABC, abstractmethod

import numpy as np

from ..mathutils.function import Function, funcify_method
from ..plots.motor_plots import _MotorPlots
from ..prints.motor_prints import _MotorPrints
from ..tools import tuple_handler

try:
    from functools import cached_property
except ImportError:
    from ..tools import cached_property


class Motor(ABC):
    """Abstract class to specify characteristics and useful operations for
    motors. Cannot be instantiated.

    Attributes
    ----------
    Motor.coordinate_system_orientation : str
        Orientation of the motor's coordinate system. The coordinate system
        is defined by the motor's axis of symmetry. The origin of the
        coordinate system may be placed anywhere along such axis, such as
        at the nozzle exit area, and must be kept the same for all other
        positions specified. Options are "nozzle_to_combustion_chamber" and
        "combustion_chamber_to_nozzle".
    Motor.nozzle_radius : float
        Radius of motor nozzle outlet in meters.
    Motor.nozzle_position : float
        Motor's nozzle outlet position in meters, specified in the motor's
        coordinate system. See
        :doc:`Positions and Coordinate Systems </user/positions>` for more
        information.
    Motor.dry_mass : float
        The mass of the motor when devoid of any propellants, measured in
        kilograms (kg). It encompasses the structural weight of the motor,
        including the combustion chamber, nozzles, tanks, and fasteners.
        Excluded from this measure are the propellants and any other elements
        that are dynamically accounted for in the `mass` parameter of the rocket
        class. Ensure that mass contributions from components shared with the
        rocket structure are not recounted here. This parameter does not vary
        with time.
    Motor.propellant_initial_mass : float
        Total propellant initial mass in kg, including solid, liquid and gas
        phases.
    Motor.total_mass : Function
        Total motor mass in kg as a function of time, defined as the sum
        of propellant mass and the motor's dry mass (i.e. structure mass).
    Motor.propellant_mass : Function
        Total propellant mass in kg as a function of time, including solid,
        liquid and gas phases.
    Motor.total_mass_flow_rate : Function
        Time derivative of propellant total mass in kg/s as a function
        of time as obtained by the thrust source.
    Motor.center_of_mass : Function
        Position of the motor center of mass in
        meters as a function of time.
        See :doc:`Positions and Coordinate Systems </user/positions>`
        for more information regarding the motor's coordinate system.
    Motor.center_of_propellant_mass : Function
        Position of the motor propellant center of mass in meters as a
        function of time.
        See :doc:`Positions and Coordinate Systems </user/positions>`
        for more information regarding the motor's coordinate system.
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
        Total motor burn duration, in seconds. It is the difference between
        the burn_out_time and the burn_start_time.
    Motor.exhaust_velocity : Function
        Propulsion gases exhaust velocity in m/s.
    Motor.interpolate : string
        Method of interpolation used in case thrust curve is given
        by data set in .csv or .eng, or as an array. Options are 'spline'
        'akima' and 'linear'. Default is "linear".
    """

    def __init__(
        self,
        thrust_source,
        dry_mass,
        dry_inertia,
        nozzle_radius,
        center_of_dry_mass_position,
        nozzle_position=0,
        burn_time=None,
        reshape_thrust_curve=False,
        interpolation_method="linear",
        coordinate_system_orientation="nozzle_to_combustion_chamber",
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
            The .csv file shall contain no headers and the first column must
            specify time in seconds, while the second column specifies thrust.
            Arrays may also be specified, following rules set by the class
            Function. Thrust units are Newtons.

            .. seealso:: :doc:`Thrust Source Details </user/motors/thrust>`

        dry_mass : int, float
            Same as in Motor class. See the :class:`Motor <rocketpy.Motor>` docs
        center_of_dry_mass_position : int, float
            The position, in meters, of the motor's center of mass with respect
            to the motor's coordinate system when it is devoid of propellant.
            See :doc:`Positions and Coordinate Systems </user/positions>`
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
        nozzle_radius : int, float, optional
            Motor's nozzle outlet radius in meters.
        burn_time: float, tuple of float, optional
            Motor's burn time.
            If a float is given, the burn time is assumed to be between 0 and
            the given float, in seconds.
            If a tuple of float is given, the burn time is assumed to be between
            the first and second elements of the tuple, in seconds.
            If not specified, automatically sourced as the range between the
            first and last-time step of the motor's thrust curve. This can only
            be used if the motor's thrust is defined by a list of points, such
            as a .csv file, a .eng file or a Function instance whose source is a
            list.
        nozzle_position : int, float, optional
            Motor's nozzle outlet position in meters, in the motor's coordinate
            system. See :doc:`Positions and Coordinate Systems </user/positions>`
            for details. Default is 0, in which case the origin of the
            coordinate system is placed at the motor's nozzle outlet.
        reshape_thrust_curve : boolean, tuple, optional
            If False, the original thrust curve supplied is not altered. If a
            tuple is given, whose first parameter is a new burn out time and
            whose second parameter is a new total impulse in Ns, the thrust
            curve is reshaped to match the new specifications. May be useful
            for motors whose thrust curve shape is expected to remain similar
            in case the impulse and burn time varies slightly. Default is
            False. Note that the Motor burn_time parameter must include the new
            reshaped burn time.
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

        Returns
        -------
        None
        """
        # Define coordinate system orientation
        self.coordinate_system_orientation = coordinate_system_orientation
        if coordinate_system_orientation == "nozzle_to_combustion_chamber":
            self._csys = 1
        elif coordinate_system_orientation == "combustion_chamber_to_nozzle":
            self._csys = -1
        else:
            raise ValueError(
                "Invalid coordinate system orientation. Options are "
                "'nozzle_to_combustion_chamber' and 'combustion_chamber_to_nozzle'."
            )

        # Motor parameters
        self.dry_mass = dry_mass
        self.interpolate = interpolation_method
        self.nozzle_position = nozzle_position
        self.nozzle_radius = nozzle_radius
        self.center_of_dry_mass_position = center_of_dry_mass_position

        # Inertia tensor setup
        inertia = (*dry_inertia, 0, 0, 0) if len(dry_inertia) == 3 else dry_inertia
        self.dry_I_11 = inertia[0]
        self.dry_I_22 = inertia[1]
        self.dry_I_33 = inertia[2]
        self.dry_I_12 = inertia[3]
        self.dry_I_13 = inertia[4]
        self.dry_I_23 = inertia[5]

        # Handle .eng file inputs
        if isinstance(thrust_source, str):
            if thrust_source[-3:] == "eng":
                _, _, points = Motor.import_eng(thrust_source)
                thrust_source = points

        # Evaluate raw thrust source
        self.thrust_source = thrust_source
        self.thrust = Function(
            thrust_source, "Time (s)", "Thrust (N)", self.interpolate, "zero"
        )

        # Handle burn_time input
        self.burn_time = burn_time

        if callable(self.thrust.source):
            self.thrust.set_discrete(*self.burn_time, 50, self.interpolate, "zero")

        # Reshape thrust_source if needed
        if reshape_thrust_curve:
            # Overwrites burn_time and thrust
            self.thrust = Motor.reshape_thrust_curve(self.thrust, *reshape_thrust_curve)
            self.burn_time = (self.thrust.x_array[0], self.thrust.x_array[-1])

        # Post process thrust
        self.thrust = Motor.clip_thrust(self.thrust, self.burn_time)

        # Auxiliary quantities
        self.burn_start_time = self.burn_time[0]
        self.burn_out_time = self.burn_time[1]
        self.burn_duration = self.burn_time[1] - self.burn_time[0]

        # Define motor attributes
        self.nozzle_radius = nozzle_radius
        self.nozzle_position = nozzle_position

        # Compute thrust metrics
        self.max_thrust = np.amax(self.thrust.y_array)
        max_thrust_index = np.argmax(self.thrust.y_array)
        self.max_thrust_time = self.thrust.source[max_thrust_index, 0]
        self.average_thrust = self.total_impulse / self.burn_duration

        # Initialize plots and prints object
        self.prints = _MotorPrints(self)
        self.plots = _MotorPlots(self)
        return None

    @property
    def burn_time(self):
        """Burn time range in seconds.

        Returns
        -------
        tuple
            Burn time range in seconds.
        """
        return self._burn_time

    @burn_time.setter
    def burn_time(self, burn_time):
        """Sets burn time range in seconds.

        Parameters
        ----------
        burn_time : float or two position array_like
            Burn time range in seconds.
        """
        if burn_time:
            self._burn_time = tuple_handler(burn_time)
        else:
            if not callable(self.thrust.source):
                self._burn_time = (self.thrust.x_array[0], self.thrust.x_array[-1])
            else:
                raise ValueError(
                    "When using a float or callable as thrust source, a burn_time"
                    " argument must be specified."
                )

    @cached_property
    def total_impulse(self):
        """Calculates and returns total impulse by numerical integration
        of the thrust curve in SI units.

        Returns
        -------
        self.total_impulse : float
            Motor total impulse in Ns.
        """
        return self.thrust.integral(*self.burn_time)

    @property
    @abstractmethod
    def exhaust_velocity(self):
        """Exhaust velocity of the motor gases.

        Returns
        -------
        self.exhaust_velocity : Function
            Gas exhaust velocity of the motor.

        Notes
        -----
        This method is implemented in the following manner by the child
        Motor classes:

        - The ``SolidMotor`` assumes a constant exhaust velocity and computes
          it as the ratio of the total impulse and the propellant mass;
        - The ``HybridMotor`` assumes a constant exhaust velocity and computes
          it as the ratio of the total impulse and the propellant mass;
        - The ``LiquidMotor`` class favors the more accurate data from the
          Tanks's mass flow rates. Therefore the exhaust velocity is generally
          variable, being the ratio of the motor thrust by the mass flow rate.
        """
        pass

    @funcify_method("Time (s)", "Total mass (kg)")
    def total_mass(self):
        """Total mass of the motor as a function of time. It is defined as the
        propellant mass plus the dry mass.

        Returns
        -------
        Function
            Motor total mass as a function of time.
        """
        return self.propellant_mass + self.dry_mass

    @funcify_method("Time (s)", "Propellant mass (kg)")
    def propellant_mass(self):
        """Total propellant mass as a Function of time.

        Returns
        -------
        Function
            Total propellant mass as a function of time.
        """
        return (
            self.total_mass_flow_rate.integral_function() + self.propellant_initial_mass
        )

    @funcify_method("Time (s)", "Mass flow rate (kg/s)", extrapolation="zero")
    def total_mass_flow_rate(self):
        """Time derivative of the propellant mass as a function of time. The
        formula used is the opposite of thrust divided by exhaust velocity.

        Returns
        -------
        Function
            Time derivative of total propellant mass a function of time.

        See Also
        --------
        SolidMotor.mass_flow_rate :
            Numerically equivalent to ``total_mass_flow_rate``.
        HybridMotor.mass_flow_rate :
            Numerically equivalent to ``total_mass_flow_rate``.
        LiquidMotor.mass_flow_rate :
            Independent of ``total_mass_flow_rate`` favoring more accurate
            sum of Tanks' mass flow rates.

        Notes
        -----
        This function computes the total mass flow rate of the motor by
        dividing the thrust data by the exhaust velocity. This is an
        approximation, and it  is used by the child Motor classes as follows:

        - The ``SolidMotor`` class uses this approximation to compute the
          grain's mass flow rate;
        - The ``HybridMotor`` class uses this approximation as a reference
          to the sum of the oxidizer and fuel (grains) mass flow rates;
        - The ``LiquidMotor`` class favors the more accurate data from the
          Tanks's mass flow rates. Therefore this value is numerically
          independent of the ``LiquidMotor.mass_flow_rate``.
        - The ``GenericMotor`` class considers the total_mass_flow_rate as the
        same as the mass_flow_rate.

        It should be noted that, for hybrid motors, the oxidizer mass flow
        rate should not be greater than `total_mass_flow_rate`, otherwise the
        grains mass flow rate will be negative, losing physical meaning.
        """
        return -1 * self.thrust / self.exhaust_velocity

    @property
    @abstractmethod
    def propellant_initial_mass(self):
        """Propellant initial mass in kg, including solid, liquid and gas phases

        Returns
        -------
        float
            Propellant initial mass in kg.
        """
        pass

    @funcify_method("Time (s)", "Motor center of mass (m)")
    def center_of_mass(self):
        """Position of the center of mass as a function of time. The position
        is specified as a scalar, relative to the motor's coordinate system.

        Returns
        -------
        Function
            Position of the center of mass as a function of time.
        """
        mass_balance = (
            self.center_of_propellant_mass * self.propellant_mass
            + self.dry_mass * self.center_of_dry_mass_position
        )
        return mass_balance / self.total_mass

    @property
    @abstractmethod
    def center_of_propellant_mass(self):
        """Position of the propellant center of mass as a function of time.
        The position is specified as a scalar, relative to the origin of the
        motor's coordinate system.

        Returns
        -------
        Function
            Position of the propellant center of mass as a function of time.
        """
        pass

    @funcify_method("Time (s)", "Inertia I_11 (kg m²)")
    def I_11(self):
        """Inertia tensor 11 component, which corresponds to the inertia
        relative to the e_1 axis, centered at the instantaneous center of mass.

        Returns
        -------
        Function
            Propellant inertia tensor 11 component at time t.

        Notes
        -----
        The e_1 direction is assumed to be the direction perpendicular to the
        motor body axis. Also, due to symmetry, I_11 = I_22.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor
        """
        # Propellant inertia tensor 11 component wrt propellant center of mass
        propellant_I_11 = self.propellant_I_11

        # Dry inertia tensor 11 component wrt dry center of mass
        dry_I_11 = self.dry_I_11

        # Steiner theorem the get inertia wrt motor center of mass
        propellant_I_11 += (
            self.propellant_mass
            * (self.center_of_propellant_mass - self.center_of_mass) ** 2
        )

        dry_I_11 += (
            self.dry_mass
            * (self.center_of_dry_mass_position - self.center_of_mass) ** 2
        )

        # Sum of inertia components
        return propellant_I_11 + dry_I_11

    @funcify_method("Time (s)", "Inertia I_22 (kg m²)")
    def I_22(self):
        """Inertia tensor 22 component, which corresponds to the inertia
        relative to the e_2 axis, centered at the instantaneous center of mass.

        Returns
        -------
        Function
            Propellant inertia tensor 22 component at time t.

        Notes
        -----
        The e_2 direction is assumed to be the direction perpendicular to the
        motor body axis, and perpendicular to e_1. Also, due to symmetry,
        I_22 = I_11.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor
        """
        # Due to symmetry, I_22 = I_11
        return self.I_11

    @funcify_method("Time (s)", "Inertia I_33 (kg m²)")
    def I_33(self):
        """Inertia tensor 33 component, which corresponds to the inertia
        relative to the e_3 axis, centered at the instantaneous center of mass.

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
        # Propellant inertia tensor 33 component wrt propellant center of mass
        propellant_I_33 = self.propellant_I_33

        # Dry inertia tensor 33 component wrt dry center of mass
        dry_I_33 = self.dry_I_33

        # Both inertia components wrt the same axis, Steiner not needed
        return propellant_I_33 + dry_I_33

    @funcify_method("Time (s)", "Inertia I_12 (kg m²)")
    def I_12(self):
        """Inertia tensor 12 component, which corresponds to the product of
        inertia relative to axes e_1 and e_2, centered at the instantaneous
        center of mass.

        Returns
        -------
        Function
            Propellant inertia tensor 12 component at time t.

        Notes
        -----
        The e_1 direction is assumed to be the direction perpendicular to the
        motor body axis.
        The e_2 direction is assumed to be the direction perpendicular to the
        motor body axis, and perpendicular to e_1.
        RocketPy follows the definition of the inertia tensor as in [1], which
        includes the minus sign for all products of inertia.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor
        """
        # Propellant inertia tensor 12 component wrt propellant center of mass
        propellant_I_12 = self.propellant_I_12

        # Dry inertia tensor 12 component wrt dry center of mass
        dry_I_12 = self.dry_I_12

        # Steiner correction not needed since the centers only move in the e_3 axis
        return propellant_I_12 + dry_I_12

    @funcify_method("Time (s)", "Inertia I_13 (kg m²)")
    def I_13(self):
        """Inertia tensor 13 component, which corresponds to the product of
        inertia relative to the axes e_1 and e_3, centered at the instantaneous
        center of mass.

        Returns
        -------
        Function
            Propellant inertia tensor 13 component at time t.

        Notes
        -----
        The e_1 direction is assumed to be the direction perpendicular to the
        motor body axis.
        The e_3 direction is assumed to be the axial direction of the rocket
        motor.
        RocketPy follows the definition of the inertia tensor as in [1], which
        includes the minus sign for all products of inertia.

        References
        ----------
        https://en.wikipedia.org/wiki/Moment_of_inertia
        """
        # Propellant inertia tensor 13 component wrt propellant center of mass
        propellant_I_13 = self.propellant_I_13

        # Dry inertia tensor 13 component wrt dry center of mass
        dry_I_13 = self.dry_I_13

        # Steiner correction not needed since the centers only move in the e_3 axis
        return propellant_I_13 + dry_I_13

    @funcify_method("Time (s)", "Inertia I_23 (kg m²)")
    def I_23(self):
        """Inertia tensor 23 component, which corresponds to the product of
        inertia relative the axes e_2 and e_3, centered at the instantaneous
        center of mass.

        Returns
        -------
        Function
            Propellant inertia tensor 23 component at time t.

        Notes
        -----
        The e_2 direction is assumed to be the direction perpendicular to the
        motor body axis, and perpendicular to e_1.
        The e_3 direction is assumed to be the axial direction of the rocket
        motor.
        RocketPy follows the definition of the inertia tensor as in [1], which
        includes the minus sign for all products of inertia.

        References
        ----------
        https://en.wikipedia.org/wiki/Moment_of_inertia
        """
        # wrt = with respect to
        # Propellant inertia tensor 23 component wrt propellant center of mass
        propellant_I_23 = self.propellant_I_23

        # Dry inertia tensor 23 component wrt dry center of mass
        dry_I_23 = self.dry_I_23

        # Steiner correction not needed since the centers only move in the e_3 axis
        return propellant_I_23 + dry_I_23

    @property
    @abstractmethod
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
        pass

    @property
    @abstractmethod
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
        pass

    @property
    @abstractmethod
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
        pass

    @property
    @abstractmethod
    def propellant_I_12(self):
        """Inertia tensor 12 component of the propellant, the product of inertia
        is relative to axes e_1 and e_2, centered at the instantaneous propellant
        center of mass.

        Returns
        -------
        Function
            Propellant inertia tensor 12 component at time t.

        Notes
        -----
        The e_1 direction is assumed to be the direction perpendicular to the
        motor body axis.
        The e_2 direction is assumed to be the direction perpendicular to the
        motor body axis, and perpendicular to e_1.
        RocketPy follows the definition of the inertia tensor as in [1], which
        includes the minus sign for all products of inertia.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor
        """
        pass

    @property
    @abstractmethod
    def propellant_I_13(self):
        """Inertia tensor 13 component of the propellant, the product of inertia
        is relative to axes e_1 and e_3, centered at the instantaneous
        propellant center of mass.

        Returns
        -------
        Function
            Propellant inertia tensor 13 component at time t.

        Notes
        -----
        The e_1 direction is assumed to be the direction perpendicular to the
        motor body axis.
        The e_3 direction is assumed to be the axial direction of the rocket
        motor.
        RocketPy follows the definition of the inertia tensor as in [1], which
        includes the minus sign for all products of inertia.

        References
        ----------
        https://en.wikipedia.org/wiki/Moment_of_inertia
        """
        pass

    @property
    @abstractmethod
    def propellant_I_23(self):
        """Inertia tensor 23 component of the propellant, the product of inertia
        is relative to axes e_2 and e_3, centered at the instantaneous
        propellant center of mass.

        Returns
        -------
        Function
            Propellant inertia tensor 23 component at time t.

        Notes
        -----
        The e_2 direction is assumed to be the direction perpendicular to the
        motor body axis, and perpendicular to e_1.
        The e_3 direction is assumed to be the axial direction of the rocket
        motor.
        RocketPy follows the definition of the inertia tensor as in [1], which
        includes the minus sign for all products of inertia.

        References
        ----------
        https://en.wikipedia.org/wiki/Moment_of_inertia
        """
        pass

    @staticmethod
    def reshape_thrust_curve(thrust, new_burn_time, total_impulse):
        """Transforms the thrust curve supplied by changing its total
        burn time and/or its total impulse, without altering the
        general shape of the curve.

        Parameters
        ----------
        thrust : Function
            Thrust curve to be reshaped.
        new_burn_time : float, tuple of float
            New desired burn time in seconds.
        total_impulse : float
            New desired total impulse.

        Returns
        -------
        Function
            Reshaped thrust curve.
        """
        # Retrieve current thrust curve data points
        time_array, thrust_array = thrust.x_array, thrust.y_array
        new_burn_time = tuple_handler(new_burn_time)

        # Compute old thrust based on new time discretization
        # Adjust scale
        new_time_array = (
            (new_burn_time[1] - new_burn_time[0]) / (time_array[-1] - time_array[0])
        ) * time_array
        # Adjust origin
        new_time_array = new_time_array - new_time_array[0] + new_burn_time[0]
        source = np.column_stack((new_time_array, thrust_array))
        thrust = Function(
            source, "Time (s)", "Thrust (N)", thrust.__interpolation__, "zero"
        )

        # Get old total impulse
        old_total_impulse = thrust.integral(*new_burn_time)

        # Compute new thrust values
        new_thrust_array = (total_impulse / old_total_impulse) * thrust_array
        source = np.column_stack((new_time_array, new_thrust_array))
        thrust = Function(
            source, "Time (s)", "Thrust (N)", thrust.__interpolation__, "zero"
        )

        return thrust

    @staticmethod
    def clip_thrust(thrust, new_burn_time):
        """Clips the thrust curve data points according to the new_burn_time
        parameter. If the burn_time range does not coincides with the thrust
        dataset, their values are interpolated.

        Parameters
        ----------
        thrust : Function
            Thrust curve to be clipped.
        new_burn_time : float, tuple of float
            New desired burn time in seconds for the thrust curve.
            Must be within the thrust curve time range, otherwise
            the thrust time range is used instead.

        Returns
        -------
        Function
            Clipped thrust curve.
        """
        # Check if burn_time is within thrust_source range
        changed_burn_time = False
        burn_time = list(tuple_handler(new_burn_time))

        if burn_time[1] > thrust.x_array[-1]:
            burn_time[1] = thrust.x_array[-1]
            changed_burn_time = True

        if burn_time[0] < thrust.x_array[0]:
            burn_time[0] = thrust.x_array[0]
            changed_burn_time = True

        if changed_burn_time:
            warnings.warn(
                f"burn_time argument {new_burn_time} is out of "
                "thrust source time range. "
                "Using thrust_source boundary times instead: "
                f"({burn_time[0]}, {burn_time[1]}) s.\n"
                "If you want to change the burn out time of the "
                "curve please use the 'reshape_thrust_curve' argument."
            )

        # Clip thrust input according to burn_time
        bound_mask = np.logical_and(
            thrust.x_array > burn_time[0],
            thrust.x_array < burn_time[1],
        )
        clipped_source = thrust.source[bound_mask]

        # Update source with burn_time points
        end_burn_data = [(burn_time[1], thrust(burn_time[1]))]
        clipped_source = np.append(clipped_source, end_burn_data, 0)
        start_burn_data = [(burn_time[0], thrust(burn_time[0]))]
        clipped_source = np.insert(clipped_source, 0, start_burn_data, 0)

        return Function(
            clipped_source,
            "Time (s)",
            "Thrust (N)",
            thrust.__interpolation__,
            "zero",
        )

    @staticmethod
    def import_eng(file_name):
        """Read content from .eng file and process it, in order to return the
        comments, description and data points.

        Parameters
        ----------
        file_name : string
            Name of the .eng file. E.g. 'test.eng'.
            Note that the .eng file must not contain the 0 0 point.

        Returns
        -------
        comments : list
            All comments in the .eng file, separated by line in a list. Each
            line is an entry of the list.
        description: list
            Description of the motor. All attributes are returned separated in
            a list. E.g. "F32 24 124 5-10-15 .0377 .0695 RV" is return as
            ['F32', '24', '124', '5-10-15', '.0377', '.0695', 'RV']
        dataPoints: list
            List of all data points in file. Each data point is an entry in
            the returned list and written as a list of two entries.
        """

        # Initialize arrays
        comments = []
        description = []
        data_points = [[0, 0]]

        # Open and read .eng file
        with open(file_name) as file:
            for line in file:
                if re.search(r";.*", line):
                    # Extract comment
                    comments.append(re.findall(r";.*", line)[0])
                    line = re.sub(r";.*", "", line)
                if line.strip():
                    if description == []:
                        # Extract description
                        description = line.strip().split(" ")
                    else:
                        # Extract thrust curve data points
                        time, thrust = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line)
                        data_points.append([float(time), float(thrust)])

        # Return all extract content
        return comments, description, data_points

    def export_eng(self, file_name, motor_name):
        """Exports thrust curve data points and motor description to
        .eng file format. A description of the format can be found
        here: http://www.thrustcurve.org/raspformat.shtml

        Parameters
        ----------
        file_name : string
            Name of the .eng file to be exported. E.g. 'test.eng'
        motor_name : string
            Name given to motor. Will appear in the description of the
            .eng file. E.g. 'Mandioca'

        Returns
        -------
        None
        """
        # Open file
        file = open(file_name, "w")

        # Write first line
        file.write(
            motor_name
            + " {:3.1f} {:3.1f} 0 {:2.3} {:2.3} RocketPy\n".format(
                2000 * self.grain_outer_radius,
                1000
                * self.grain_number
                * (self.grain_initial_height + self.grain_separation),
                self.propellant_initial_mass,
                self.propellant_initial_mass,
            )
        )

        # Write thrust curve data points
        for time, thrust in self.thrust.source[1:-1, :]:
            # time, thrust = item
            file.write("{:.4f} {:.3f}\n".format(time, thrust))

        # Write last line
        file.write("{:.4f} {:.3f}\n".format(self.thrust.source[-1, 0], 0))

        # Close file
        file.close()

        return None

    def info(self):
        """Prints out a summary of the data and graphs available about the
        Motor.
        """
        # Print motor details
        self.prints.all()
        self.plots.thrust()
        return None

    @abstractmethod
    def all_info(self):
        """Prints out all data and graphs available about the Motor."""
        self.prints.all()
        self.plots.all()
        return None


class GenericMotor(Motor):
    """Class that represents a simple motor defined mainly by its thrust curve.
    There is no distinction between the propellant types (e.g. Solid, Liquid).
    This class is meant for rough estimations of the motor performance,
    therefore for more accurate results, use the ``SolidMotor``, ``HybridMotor``
    or ``LiquidMotor`` classes."""

    def __init__(
        self,
        thrust_source,
        burn_time,
        chamber_radius,
        chamber_height,
        chamber_position,
        propellant_initial_mass,
        nozzle_radius,
        dry_mass=0,
        center_of_dry_mass_position=None,
        dry_inertia=(0, 0, 0),
        nozzle_position=0,
        reshape_thrust_curve=False,
        interpolation_method="linear",
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    ):
        """Initialize GenericMotor class, process thrust curve and geometrical
        parameters and store results.

        Parameters
        ----------
        thrust_source : int, float, callable, string, array, Function
            Motor's thrust curve. Can be given as an int or float, in which
            case the thrust will be considered constant in time. It can
            also be given as a callable function, whose argument is time in
            seconds and returns the thrust supplied by the motor in the
            instant. If a string is given, it must point to a .csv or .eng file.
            The .csv file shall contain no headers and the first column must
            specify time in seconds, while the second column specifies thrust.
            Arrays may also be specified, following rules set by the class
            Function. Thrust units are Newtons.

            .. seealso:: :doc:`Thrust Source Details </user/motors/thrust>`

        chamber_radius : int, float
            The radius of a overall cylindrical chamber of propellant in meters.
            This is a rough estimate for the motor's propellant chamber or tanks.
        chamber_height : int, float
            The height of a overall cylindrical chamber of propellant in meters.
            This is a rough estimate for the motor's propellant chamber or tanks.
        chamber_position : int, float
            The position, in meters, of the centroid (half height) of the motor's
            overall cylindrical chamber of propellant with respect to the motor's
            coordinate system.
            See :doc:`Positions and Coordinate Systems </user/positions>`
        dry_mass : int, float
            Same as in Motor class. See the :class:`Motor <rocketpy.Motor>` docs
        propellant_initial_mass : int, float
            The initial mass of the propellant in the motor.
        center_of_dry_mass_position : int, float, optional
            The position, in meters, of the motor's center of mass with respect
            to the motor's coordinate system when it is devoid of propellant.
            If not specified, automatically sourced as the chamber position.
            See :doc:`Positions and Coordinate Systems </user/positions>`
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
        nozzle_radius : int, float, optional
            Motor's nozzle outlet radius in meters.
        burn_time: float, tuple of float, optional
            Motor's burn time.
            If a float is given, the burn time is assumed to be between 0 and
            the given float, in seconds.
            If a tuple of float is given, the burn time is assumed to be between
            the first and second elements of the tuple, in seconds.
            If not specified, automatically sourced as the range between the
            first and last-time step of the motor's thrust curve. This can only
            be used if the motor's thrust is defined by a list of points, such
            as a .csv file, a .eng file or a Function instance whose source is a
            list.
        nozzle_position : int, float, optional
            Motor's nozzle outlet position in meters, in the motor's coordinate
            system. See :doc:`Positions and Coordinate Systems </user/positions>`
            for details. Default is 0, in which case the origin of the
            coordinate system is placed at the motor's nozzle outlet.
        reshape_thrust_curve : boolean, tuple, optional
            If False, the original thrust curve supplied is not altered. If a
            tuple is given, whose first parameter is a new burn out time and
            whose second parameter is a new total impulse in Ns, the thrust
            curve is reshaped to match the new specifications. May be useful
            for motors whose thrust curve shape is expected to remain similar
            in case the impulse and burn time varies slightly. Default is
            False. Note that the Motor burn_time parameter must include the new
            reshaped burn time.
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

        Returns
        -------
        None
        """
        super().__init__(
            thrust_source,
            dry_mass,
            dry_inertia,
            nozzle_radius,
            center_of_dry_mass_position,
            nozzle_position,
            burn_time,
            reshape_thrust_curve,
            interpolation_method,
            coordinate_system_orientation,
        )

        self.chamber_radius = chamber_radius
        self.chamber_height = chamber_height
        self.chamber_position = chamber_position
        self.propellant_initial_mass = propellant_initial_mass

        # Set center of mass and estimate to chamber position if not given
        self.center_of_dry_mass_position = (
            center_of_dry_mass_position
            if center_of_dry_mass_position
            else chamber_position
        )
        # Initialize plots and prints object
        self.prints = _MotorPrints(self)
        self.plots = _MotorPlots(self)
        return None

    @cached_property
    def propellant_initial_mass(self):
        """Calculates the initial mass of the propellant.

        Returns
        -------
        float
            Initial mass of the propellant.
        """
        return self.propellant_initial_mass

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

    @funcify_method("Time (s)", "Mass Flow Rate (kg/s)")
    def mass_flow_rate(self):
        """Time derivative of propellant mass. Assumes constant exhaust
        velocity. The formula used is the opposite of thrust divided by
        exhaust velocity.
        """
        return -1 * self.thrust / self.exhaust_velocity

    @funcify_method("Time (s)", "center of mass (m)")
    def center_of_propellant_mass(self):
        """Estimates the propellant center of mass as fixed in the chamber
        position. For a more accurate evaluation, use the classes SolidMotor,
        LiquidMotor or HybridMotor.

        Returns
        -------
        Function
            Function representing the center of mass of the motor.
        """
        return self.chamber_position

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
        return (
            self.propellant_mass
            * (3 * self.chamber_radius**2 + self.chamber_height**2)
            / 12
        )

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

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor
        """
        return self.propellant_mass * self.chamber_radius**2 / 2

    @funcify_method("Time (s)", "Inertia I_12 (kg m²)")
    def propellant_I_12(self):
        return Function(0)

    @funcify_method("Time (s)", "Inertia I_13 (kg m²)")
    def propellant_I_13(self):
        return Function(0)

    @funcify_method("Time (s)", "Inertia I_23 (kg m²)")
    def propellant_I_23(self):
        return Function(0)

    def all_info(self):
        """Prints out all data and graphs available about the Motor."""
        # Print motor details
        self.prints.all()
        self.plots.all()
        return None


class EmptyMotor:
    """Class that represents an empty motor with no mass and no thrust."""

    # TODO: This is a temporary solution. It should be replaced by a class that
    # inherits from the abstract Motor class. Currently cannot be done easily.
    def __init__(self):
        """Initializes an empty motor with no mass and no thrust.

        Notes
        -----
        This class is a temporary solution to the problem of having a motor
        with no mass and no thrust. It should be replaced by a class that
        inherits from the abstract Motor class. Currently cannot be done easily.
        """
        self._csys = 1
        self.dry_mass = 0
        self.nozzle_radius = 0
        self.thrust = Function(0, "Time (s)", "Thrust (N)")
        self.propellant_mass = Function(0, "Time (s)", "Propellant Mass (kg)")
        self.total_mass = Function(0, "Time (s)", "Total Mass (kg)")
        self.total_mass_flow_rate = Function(
            0, "Time (s)", "Mass Depletion Rate (kg/s)"
        )
        self.burn_out_time = 1
        self.nozzle_position = 0
        self.nozzle_radius = 0
        self.center_of_dry_mass_position = 0
        self.center_of_propellant_mass = Function(
            0, "Time (s)", "Center of Propellant Mass (kg)"
        )
        self.center_of_mass = Function(0, "Time (s)", "Center of Mass (kg)")
        self.dry_I_11 = 0
        self.dry_I_22 = 0
        self.dry_I_33 = 0
        self.dry_I_12 = 0
        self.dry_I_13 = 0
        self.dry_I_23 = 0
        self.I_11 = Function(0)
        self.I_22 = Function(0)
        self.I_33 = Function(0)
        self.I_12 = Function(0)
        self.I_13 = Function(0)
        self.I_23 = Function(0)
        return None
