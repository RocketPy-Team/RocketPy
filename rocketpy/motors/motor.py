import base64
import re
import tempfile
import warnings
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from functools import cached_property
from os import path, remove
from pathlib import Path

import numpy as np
import requests

from ..mathutils.function import Function, funcify_method
from ..plots.motor_plots import _MotorPlots
from ..prints.motor_prints import _MotorPrints
from ..tools import parallel_axis_theorem_from_com, tuple_handler

# pylint: disable=too-many-public-methods
# ThrustCurve API cache
CACHE_DIR = Path.home() / ".rocketpy_cache"


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
    Motor.nozzle_area : float
        Area of motor nozzle outlet in square meters.
    Motor.nozzle_position : float
        Motor's nozzle outlet position in meters, specified in the motor's
        coordinate system. See :ref:`positions` for more information.
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
    Motor.structural_mass_ratio: float
        Initial ratio between the dry mass and the total mass.
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
        Motor thrust force obtained from the thrust source, in Newtons, as a
        function of time.
    Motor.vacuum_thrust : Function
        Motor thrust force when the rocket is in a vacuum. In Newtons, as a
        function of time.
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
        Effective exhaust velocity of the propulsion gases in m/s. Computed
        as the thrust divided by the mass flow rate. This corresponds to the
        actual exhaust velocity only when the nozzle exit pressure equals the
        atmospheric pressure.
    Motor.interpolate : string
        Method of interpolation used in case thrust curve is given
        by data set in .csv or .eng, or as an array. Options are 'spline'
        'akima' and 'linear'. Default is "linear".
    Motor.reference_pressure : int, float, None
        Atmospheric pressure in Pa at which the thrust data was recorded.
        It will allow to obtain the net thrust in the Flight class.
    """

    # pylint: disable=too-many-statements
    def __init__(
        self,
        thrust_source,
        dry_inertia,
        nozzle_radius,
        center_of_dry_mass_position,
        dry_mass=None,
        nozzle_position=0,
        burn_time=None,
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
        reference_pressure : int, float, optional
            Atmospheric pressure in Pa at which the thrust data was recorded.

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
        else:  # pragma: no cover
            raise ValueError(
                "Invalid coordinate system orientation. Options are "
                "'nozzle_to_combustion_chamber' and 'combustion_chamber_to_nozzle'."
            )

        # Motor parameters
        self.interpolate = interpolation_method
        self.nozzle_position = nozzle_position
        self.nozzle_radius = nozzle_radius
        self.nozzle_area = np.pi * nozzle_radius**2
        self.center_of_dry_mass_position = center_of_dry_mass_position
        self.reference_pressure = reference_pressure

        # Inertia tensor setup
        inertia = (*dry_inertia, 0, 0, 0) if len(dry_inertia) == 3 else dry_inertia
        self.dry_I_11 = inertia[0]
        self.dry_I_22 = inertia[1]
        self.dry_I_33 = inertia[2]
        self.dry_I_12 = inertia[3]
        self.dry_I_13 = inertia[4]
        self.dry_I_23 = inertia[5]

        # Handle .eng or .rse file inputs
        self.description_eng_file = None
        self.rse_motor_data = None
        if isinstance(thrust_source, str):
            if (
                path.exists(thrust_source)
                and path.splitext(path.basename(thrust_source))[1] == ".eng"
            ):
                _, self.description_eng_file, points = Motor.import_eng(thrust_source)
                thrust_source = points
            elif (
                path.exists(thrust_source)
                and path.splitext(path.basename(thrust_source))[1] == ".rse"
            ):
                self.rse_motor_data, points = Motor.import_rse(thrust_source)
                thrust_source = points
        # Evaluate raw thrust source
        self.thrust_source = thrust_source
        self.thrust = Function(
            thrust_source, "Time (s)", "Thrust (N)", self.interpolate, "zero"
        )

        # Handle dry_mass input
        self.dry_mass = dry_mass

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

        # Compute thrust metrics
        self.max_thrust = np.amax(self.thrust.y_array)
        max_thrust_index = np.argmax(self.thrust.y_array)
        self.max_thrust_time = self.thrust.source[max_thrust_index, 0]
        self.average_thrust = self.total_impulse / self.burn_duration

        # Initialize plots and prints object
        self.prints = _MotorPrints(self)
        self.plots = _MotorPlots(self)

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
            else:  # pragma: no cover
                raise ValueError(
                    "When using a float or callable as thrust source, a burn_time"
                    " argument must be specified."
                )

    @property
    def dry_mass(self):
        """Dry mass of the motor in kg.

        Returns
        -------
        self.dry_mass : float
            Motor dry mass in kg.
        """
        return self._dry_mass

    @dry_mass.setter
    def dry_mass(self, dry_mass):
        """Sets dry mass of the motor in kg.

        Parameters
        ----------
        dry_mass : float
            Motor dry mass in kg.
        """
        if dry_mass is not None:
            if isinstance(dry_mass, (int, float)):
                self._dry_mass = dry_mass
            else:
                raise ValueError("Dry mass must be a number.")
        elif self.description_eng_file:
            self._dry_mass = float(self.description_eng_file[-2]) - float(
                self.description_eng_file[-3]
            )
        elif self.rse_motor_data:
            self._dry_mass = float(
                self.rse_motor_data["description"]["total_mass"]
            ) - float(self.rse_motor_data["description"]["propellant_mass"])
        else:
            raise ValueError("Dry mass must be specified.")

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
        """Effective exhaust velocity of the motor gases.

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

        This corresponds to the actual exhaust velocity only when the nozzle
        exit pressure equals the atmospheric pressure.
        """

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
            Independent of ``total_mass_flow_rate`` favoring more accurate \
            sum of Tanks' mass flow rates.

        Notes
        -----
        This function computes the total mass flow rate of the motor by
        dividing the thrust data by the exhaust velocity. This is an
        approximation, and it  is used by the child Motor classes as follows:

        - The ``SolidMotor`` class uses this approximation to compute the \
            grain's mass flow rate;
        - The ``HybridMotor`` class uses this approximation as a reference \
            to the sum of the oxidizer and fuel (grains) mass flow rates;
        - The ``LiquidMotor`` class favors the more accurate data from the \
            Tanks's mass flow rates. Therefore this value is numerically \
            independent of the ``LiquidMotor.mass_flow_rate``.
        - The ``GenericMotor`` class considers the total_mass_flow_rate as the \
            same as the mass_flow_rate.

        It should also be noted that, for hybrid motors, the oxidizer mass flow
        rate should not be greater than `total_mass_flow_rate`, otherwise the
        grains mass flow rate will be negative, losing physical meaning.
        """
        average_exhaust_velocity = self.total_impulse / self.propellant_initial_mass
        return self.thrust / -average_exhaust_velocity

    @property
    @abstractmethod
    def propellant_initial_mass(self):
        """Propellant initial mass in kg, including solid, liquid and gas phases

        Returns
        -------
        float
            Propellant initial mass in kg.
        """

    @property
    def structural_mass_ratio(self):
        """Calculates the structural mass ratio. The ratio is defined as
        the dry mass divided by the initial total mass.

        Returns
        -------
        float
            Initial structural mass ratio.
        """
        initial_total_mass = self.dry_mass + self.propellant_initial_mass
        try:
            return self.dry_mass / initial_total_mass
        except ZeroDivisionError as e:
            raise ValueError(
                "Total motor mass (dry + propellant) cannot be zero"
            ) from e

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

        See Also
        --------
        https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor
        """

        prop_I_11 = self.propellant_I_11
        dry_I_11 = self.dry_I_11

        prop_to_cm = self.center_of_propellant_mass - self.center_of_mass
        dry_to_cm = self.center_of_dry_mass_position - self.center_of_mass

        prop_I_11 = parallel_axis_theorem_from_com(
            prop_I_11, self.propellant_mass, prop_to_cm
        )
        dry_I_11 = parallel_axis_theorem_from_com(dry_I_11, self.dry_mass, dry_to_cm)

        return prop_I_11 + dry_I_11

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

        See Also
        --------
        https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor
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
        https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor
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
        https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor
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
        https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor
        """

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
        https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor
        """

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
        https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor
        """

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
        https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor
        """

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

    @staticmethod
    def reshape_thrust_curve(thrust, new_burn_time, total_impulse):
        """Transforms the thrust curve supplied by changing its total
        burn time and/or its total impulse, without altering the
        general shape of the curve. This method does not mutate the original
        object, it only returns a new thrust curve.

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

        Tip
        ---
        See the User Guide page for examples on how to use this method.
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
    def import_rse(file_name):
        """
        Reads motor data from a file and extracts comments, model, description, and data points.

        Parameters
        ----------
        file_path : str
            Path to the motor data file.

        Returns
        -------
        dict
            A dictionary containing the extracted data:
            - comments: List of comments in the file.
            - model: Dictionary with manufacturer, code, and type of the motor.
            - description: Dictionary with performance data (dimensions, weights, thrust, etc.).
            - data_points: List of temporal data points (time, thrust, mass, cg).
        tuple
            A tuple representing the thrust curve (time, thrust).
        """

        # Parse the XML file
        tree = ET.parse(file_name)
        root = tree.getroot()

        # Extract comments
        comments = []
        for comment in root.iter():
            if comment.tag.startswith("<!--"):
                comments.append(comment.text.strip())

        # Extract model data
        engine = root.find(".//engine")
        model = {
            "manufacturer": engine.attrib.get("mfg"),
            "code": engine.attrib.get("code"),
            "type": engine.attrib.get("Type"),
        }

        # Extract description data
        description = {
            "diameter": float(engine.attrib.get("dia", 0)) / 1000,
            "length": float(engine.attrib.get("len", 0)) / 1000,
            "throat_diameter": float(engine.attrib.get("throatDia", 0)) / 1000,
            "exit_diameter": float(engine.attrib.get("exitDia", 0)) / 1000,
            "total_mass": float(engine.attrib.get("initWt", 0)) / 1000,
            "propellant_mass": float(engine.attrib.get("propWt", 0)) / 1000,
            "average_thrust": float(engine.attrib.get("avgThrust", 0)),
            "peak_thrust": float(engine.attrib.get("peakThrust", 0)),
            "total_impulse": float(engine.attrib.get("Itot", 0)),
            "burn_time": float(engine.attrib.get("burn-time", 0)),
            "isp": float(engine.attrib.get("Isp", 0)),
            "mass_fraction": float(engine.attrib.get("massFrac", 0)) / 100,
        }

        # Extract data points
        data_points = []
        thrust_source = []
        for eng_data in engine.find("data").findall("eng-data"):
            time = float(eng_data.attrib.get("t", 0))
            thrust = float(eng_data.attrib.get("f", 0))
            mass = float(eng_data.attrib.get("m", 0))
            cg = float(eng_data.attrib.get("cg", 0))
            data_points.append({"time": time, "thrust": thrust, "mass": mass, "cg": cg})
            thrust_source.append([time, thrust])

        # Create the dictionary to return
        rse_file_data = {
            "comments": comments,
            "model": model,
            "description": description,
            "data_points": data_points,
        }

        return rse_file_data, thrust_source

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
                    if not description:
                        # Extract description
                        description = line.strip().split(" ")
                    else:
                        # Extract thrust curve data points
                        time, thrust = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line)
                        data_points.append([float(time), float(thrust)])

        # Return all extract content
        return comments, description, data_points

    @cached_property
    def vacuum_thrust(self):
        """Calculate the vacuum thrust from the raw thrust and the reference
        pressure at which the thrust data was recorded.

        Returns
        -------
        vacuum_thrust : Function
            The rocket's thrust in a vacuum.
        """
        if self.reference_pressure is None:
            warnings.warn(
                "Reference pressure not set. Returning thrust instead.",
                UserWarning,
            )
            return self.thrust

        return self.thrust + self.reference_pressure * self.nozzle_area

    def pressure_thrust(self, pressure):
        """Computes the contribution to thrust due to the difference between
        the atmospheric pressure and the reference pressure at which the
        thrust data was recorded.

        Parameters
        ----------
        pressure : float
            Atmospheric pressure in Pa.

        Returns
        -------
        pressure_thrust : float
            Thrust component resulting from the pressure difference.
        """
        if self.reference_pressure is None:
            return 0

        return (self.reference_pressure - pressure) * self.nozzle_area

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
        with open(file_name, "w") as file:
            # Write first line
            def get_attr_value(obj, attr_name, multiplier=1):
                return multiplier * getattr(obj, attr_name, 0)

            grain_outer_radius = get_attr_value(self, "grain_outer_radius", 2000)
            grain_number = get_attr_value(self, "grain_number", 1000)
            grain_initial_height = get_attr_value(self, "grain_initial_height")
            grain_separation = get_attr_value(self, "grain_separation")

            grain_total = grain_number * (grain_initial_height + grain_separation)

            if grain_outer_radius == 0 or grain_total == 0:
                warnings.warn(
                    "The motor object doesn't have some grain-related attributes. "
                    "Using zeros to write to file."
                )

            file.write(
                f"{motor_name} {grain_outer_radius:3.1f} {grain_total:3.1f} 0 "
                f"{self.propellant_initial_mass:2.3} "
                f"{self.propellant_initial_mass:2.3} RocketPy\n"
            )

            # Write thrust curve data points
            for time, thrust in self.thrust.source[1:-1, :]:
                file.write(f"{time:.4f} {thrust:.3f}\n")

            # Write last line
            file.write(f"{self.thrust.source[-1, 0]:.4f} {0:.3f}\n")

    def to_dict(self, **kwargs):
        data = {
            "thrust_source": self.thrust,
            "dry_I_11": self.dry_I_11,
            "dry_I_22": self.dry_I_22,
            "dry_I_33": self.dry_I_33,
            "dry_I_12": self.dry_I_12,
            "dry_I_13": self.dry_I_13,
            "dry_I_23": self.dry_I_23,
            "nozzle_radius": self.nozzle_radius,
            "nozzle_area": self.nozzle_area,
            "center_of_dry_mass_position": self.center_of_dry_mass_position,
            "dry_mass": self.dry_mass,
            "nozzle_position": self.nozzle_position,
            "burn_time": self.burn_time,
            "interpolate": self.interpolate,
            "coordinate_system_orientation": self.coordinate_system_orientation,
            "reference_pressure": self.reference_pressure,
        }

        if kwargs.get("include_outputs", False):
            total_mass = self.total_mass
            propellant_mass = self.propellant_mass
            mass_flow_rate = self.total_mass_flow_rate
            center_of_mass = self.center_of_mass
            center_of_propellant_mass = self.center_of_propellant_mass
            exhaust_velocity = self.exhaust_velocity
            I_11 = self.I_11
            I_22 = self.I_22
            I_33 = self.I_33
            I_12 = self.I_12
            I_13 = self.I_13
            I_23 = self.I_23
            propellant_I_11 = self.propellant_I_11
            propellant_I_22 = self.propellant_I_22
            propellant_I_33 = self.propellant_I_33
            propellant_I_12 = self.propellant_I_12
            propellant_I_13 = self.propellant_I_13
            propellant_I_23 = self.propellant_I_23
            if kwargs.get("discretize", False):
                total_mass = total_mass.set_discrete_based_on_model(
                    self.thrust, mutate_self=False
                )
                propellant_mass = propellant_mass.set_discrete_based_on_model(
                    self.thrust, mutate_self=False
                )
                mass_flow_rate = mass_flow_rate.set_discrete_based_on_model(
                    self.thrust, mutate_self=False
                )
                center_of_mass = center_of_mass.set_discrete_based_on_model(
                    self.thrust, mutate_self=False
                )
                center_of_propellant_mass = (
                    center_of_propellant_mass.set_discrete_based_on_model(
                        self.thrust, mutate_self=False
                    )
                )
                exhaust_velocity = exhaust_velocity.set_discrete_based_on_model(
                    self.thrust, mutate_self=False
                )
                I_11 = I_11.set_discrete_based_on_model(self.thrust, mutate_self=False)
                I_22 = I_22.set_discrete_based_on_model(self.thrust, mutate_self=False)
                I_33 = I_33.set_discrete_based_on_model(self.thrust, mutate_self=False)
                I_12 = I_12.set_discrete_based_on_model(self.thrust, mutate_self=False)
                I_13 = I_13.set_discrete_based_on_model(self.thrust, mutate_self=False)
                I_23 = I_23.set_discrete_based_on_model(self.thrust, mutate_self=False)
                propellant_I_11 = propellant_I_11.set_discrete_based_on_model(
                    self.thrust, mutate_self=False
                )
                propellant_I_22 = propellant_I_22.set_discrete_based_on_model(
                    self.thrust, mutate_self=False
                )
                propellant_I_33 = propellant_I_33.set_discrete_based_on_model(
                    self.thrust, mutate_self=False
                )
                propellant_I_12 = propellant_I_12.set_discrete_based_on_model(
                    self.thrust, mutate_self=False
                )
                propellant_I_13 = propellant_I_13.set_discrete_based_on_model(
                    self.thrust, mutate_self=False
                )
                propellant_I_23 = propellant_I_23.set_discrete_based_on_model(
                    self.thrust, mutate_self=False
                )
            data.update(
                {
                    "vacuum_thrust": self.vacuum_thrust,
                    "total_mass": total_mass,
                    "propellant_mass": propellant_mass,
                    "mass_flow_rate": mass_flow_rate,
                    "center_of_mass": center_of_mass,
                    "center_of_propellant_mass": center_of_propellant_mass,
                    "total_impulse": self.total_impulse,
                    "exhaust_velocity": exhaust_velocity,
                    "propellant_initial_mass": self.propellant_initial_mass,
                    "structural_mass_ratio": self.structural_mass_ratio,
                    "I_11": I_11,
                    "I_22": I_22,
                    "I_33": I_33,
                    "I_12": I_12,
                    "I_13": I_13,
                    "I_23": I_23,
                    "propellant_I_11": propellant_I_11,
                    "propellant_I_22": propellant_I_22,
                    "propellant_I_33": propellant_I_33,
                    "propellant_I_12": propellant_I_12,
                    "propellant_I_13": propellant_I_13,
                    "propellant_I_23": propellant_I_23,
                }
            )

        return data

    def info(self, *, filename=None):
        """Prints out a summary of the data and graphs available about the
        Motor.

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
        # Print motor details
        self.prints.all()
        self.plots.thrust(filename=filename)

    def all_info(self):
        """Prints out all data and graphs available about the Motor."""
        self.prints.all()
        self.plots.all()


# TODO: move this class to a separate file, needs a breaking change warning
class GenericMotor(Motor):
    """Class that represents a simple motor defined mainly by its thrust curve.
    There is no distinction between the propellant types (e.g. Solid, Liquid).
    This class is meant for rough estimations of the motor performance,
    therefore for more accurate results, use the ``SolidMotor``, ``HybridMotor``
    or ``LiquidMotor`` classes."""

    # pylint: disable=too-many-arguments
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
        reference_pressure=None,
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
            The .csv file can contain a single line header and the first column must
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
        reference_pressure : int, float, optional
            Atmospheric pressure in Pa at which the thrust data was recorded.
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

        Notes
        -----
        This corresponds to the actual exhaust velocity only when the nozzle
        exit pressure equals the atmospheric pressure.
        """
        return Function(
            self.total_impulse / self.propellant_initial_mass
        ).set_discrete_based_on_model(self.thrust)

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
        return Function(self.chamber_position).set_discrete_based_on_model(self.thrust)

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
        https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor
        """
        return Function(
            self.propellant_mass
            * (3 * self.chamber_radius**2 + self.chamber_height**2)
            / 12
        ).set_discrete_based_on_model(self.thrust)

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

        References
        ----------
        https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor
        """
        return Function(
            self.propellant_mass * self.chamber_radius**2 / 2
        ).set_discrete_based_on_model(self.thrust)

    @funcify_method("Time (s)", "Inertia I_12 (kg m²)")
    def propellant_I_12(self):
        return Function(0).set_discrete_based_on_model(self.thrust)

    @funcify_method("Time (s)", "Inertia I_13 (kg m²)")
    def propellant_I_13(self):
        return Function(0).set_discrete_based_on_model(self.thrust)

    @funcify_method("Time (s)", "Inertia I_23 (kg m²)")
    def propellant_I_23(self):
        return Function(0).set_discrete_based_on_model(self.thrust)

    @staticmethod
    def load_from_eng_file(
        file_name,
        nozzle_radius=None,
        chamber_radius=None,
        chamber_height=None,
        chamber_position=0,
        propellant_initial_mass=None,
        dry_mass=None,
        burn_time=None,
        center_of_dry_mass_position=None,
        dry_inertia=(0, 0, 0),
        nozzle_position=0,
        reshape_thrust_curve=False,
        interpolation_method="linear",
        coordinate_system_orientation="nozzle_to_combustion_chamber",
        reference_pressure=None,
    ):
        """Loads motor data from a .eng file and processes it.

        Parameters
        ----------
        file_name : string
            Name of the .eng file. E.g. 'test.eng'.
        nozzle_radius : int, float
            Motor's nozzle outlet radius in meters.
        chamber_radius : int, float, optional
            The radius of a overall cylindrical chamber of propellant in meters.
        chamber_height : int, float, optional
            The height of a overall cylindrical chamber of propellant in meters.
        chamber_position : int, float, optional
            The position, in meters, of the centroid (half height) of the motor's
            overall cylindrical chamber of propellant with respect to the motor's
            coordinate system.
        propellant_initial_mass : int, float, optional
            The initial mass of the propellant in the motor.
        dry_mass : int, float, optional
            Same as in Motor class. See the :class:`Motor <rocketpy.Motor>` docs
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
        center_of_dry_mass_position : int, float, optional
            The position, in meters, of the motor's center of mass with respect
            to the motor's coordinate system when it is devoid of propellant.
            If not specified, automatically sourced as the chamber position.
        dry_inertia : tuple, list
            Tuple or list containing the motor's dry mass inertia tensor
        nozzle_position : int, float, optional
            Motor's nozzle outlet position in meters, in the motor's coordinate
            system. Default is 0, in which case the origin of the
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
        Generic Motor object
        """
        if isinstance(file_name, str):
            if path.splitext(path.basename(file_name))[1] == ".eng":
                _, description, thrust_source = Motor.import_eng(file_name)
            else:
                raise ValueError("File must be a .eng file.")
        else:
            raise ValueError("File name must be a string.")

        thrust = Function(thrust_source, "Time (s)", "Thrust (N)", "linear", "zero")

        # handle eng parameters
        if not chamber_radius:
            chamber_radius = (
                float(description[1]) / 1000
            )  # get motor diameter in meters

        if not chamber_height:
            chamber_height = float(description[2]) / 1000  # get motor length in meters

        if not propellant_initial_mass:
            propellant_initial_mass = float(description[-3])

        if not dry_mass:
            total_mass = float(description[-2])
            dry_mass = total_mass - propellant_initial_mass

        if not nozzle_radius:
            nozzle_radius = 0.85 * chamber_radius

        return GenericMotor(
            thrust_source=thrust,
            burn_time=burn_time,
            chamber_radius=chamber_radius,
            chamber_height=chamber_height,
            chamber_position=chamber_position,
            propellant_initial_mass=propellant_initial_mass,
            nozzle_radius=nozzle_radius,
            dry_mass=dry_mass,
            center_of_dry_mass_position=center_of_dry_mass_position,
            dry_inertia=dry_inertia,
            nozzle_position=nozzle_position,
            reshape_thrust_curve=reshape_thrust_curve,
            interpolation_method=interpolation_method,
            coordinate_system_orientation=coordinate_system_orientation,
            reference_pressure=reference_pressure,
        )

    @staticmethod
    def load_from_rse_file(
        file_name,
        nozzle_radius=None,
        chamber_radius=None,
        chamber_height=None,
        chamber_position=0,
        propellant_initial_mass=None,
        dry_mass=None,
        burn_time=None,
        center_of_dry_mass_position=None,
        dry_inertia=(0, 0, 0),
        nozzle_position=0,
        reshape_thrust_curve=False,
        interpolation_method="linear",
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    ):
        """Loads motor data from a .rse file and processes it.

        Parameters
        ----------
        file_name : string
            Name of the .eng file. E.g. 'test.eng'.
        nozzle_radius : int, float
            Motor's nozzle outlet radius in meters.
        chamber_radius : int, float, optional
            The radius of a overall cylindrical chamber of propellant in meters.
        chamber_height : int, float, optional
            The height of a overall cylindrical chamber of propellant in meters.
        chamber_position : int, float, optional
            The position, in meters, of the centroid (half height) of the motor's
            overall cylindrical chamber of propellant with respect to the motor's
            coordinate system.
        propellant_initial_mass : int, float, optional
            The initial mass of the propellant in the motor.
        dry_mass : int, float, optional
            Same as in Motor class. See the :class:`Motor <rocketpy.Motor>` docs
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
        center_of_dry_mass_position : int, float, optional
            The position, in meters, of the motor's center of mass with respect
            to the motor's coordinate system when it is devoid of propellant.
            If not specified, automatically sourced as the chamber position.
        dry_inertia : tuple, list
            Tuple or list containing the motor's dry mass inertia tensor
        nozzle_position : int, float, optional
            Motor's nozzle outlet position in meters, in the motor's coordinate
            system. Default is 0, in which case the origin of the
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
        Generic Motor object
        """
        if isinstance(file_name, str):
            if path.splitext(path.basename(file_name))[1] == ".rse":
                description, thrust_source = Motor.import_rse(file_name)
            else:
                raise ValueError("File must be a .rse file.")
        else:
            raise ValueError("File name must be a string.")

        thrust = Function(thrust_source, "Time (s)", "Thrust (N)", "linear", "zero")

        # handle eng parameters
        if not chamber_radius:
            chamber_radius = description["description"][
                "diameter"
            ]  # get motor diameter in meters

        if not chamber_height:
            chamber_height = description["description"][
                "length"
            ]  # get motor length in meters

        if not propellant_initial_mass:
            propellant_initial_mass = description["description"]["propellant_mass"]

        if not dry_mass:
            total_mass = description["description"]["total_mass"]
            dry_mass = total_mass - propellant_initial_mass

        if not nozzle_radius:
            nozzle_radius = description["description"]["exit_diameter"]

        return GenericMotor(
            thrust_source=thrust,
            burn_time=burn_time,
            chamber_radius=chamber_radius,
            chamber_height=chamber_height,
            chamber_position=chamber_position,
            propellant_initial_mass=propellant_initial_mass,
            nozzle_radius=nozzle_radius,
            dry_mass=dry_mass,
            center_of_dry_mass_position=center_of_dry_mass_position,
            dry_inertia=dry_inertia,
            nozzle_position=nozzle_position,
            reshape_thrust_curve=reshape_thrust_curve,
            interpolation_method=interpolation_method,
            coordinate_system_orientation=coordinate_system_orientation,
        )

    @staticmethod
    def _call_thrustcurve_api(name: str, no_cache: bool = False):  # pylint: disable=too-many-statements
        """
        Download a .eng file from the ThrustCurve API
        based on the given motor name.

        Parameters
        ----------
        name : str
            The motor name according to the API (e.g., "Cesaroni_M1670" or "M1670").
            Both manufacturer-prefixed and shorthand names are commonly used; if multiple
            motors match the search, the first result is used.
        no_cache : bool, optional
            If True, forces a new API fetch even if the motor is cached.

        Returns
        -------
        data_base64 : str
            The .eng file of the motor in base64

        Raises
        ------
        ValueError
            If no motor is found or if the downloaded .eng data is missing.
        requests.exceptions.RequestException
            If a network or HTTP error occurs during the API call.

        Notes
        -----
        - The cache prevents multiple network requests for the same motor name across sessions.
        - Cached files are stored in `~/.rocketpy_cache` and reused unless `no_cache=True`.
        - Filenames are sanitized to avoid invalid characters.
        """
        try:
            CACHE_DIR.mkdir(exist_ok=True)
        except OSError as e:
            warnings.warn(f"Could not create cache directory: {e}. Caching disabled.")
            no_cache = True
        # File path in the cache
        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
        cache_file = CACHE_DIR / f"{safe_name}.eng.b64"
        if not no_cache and cache_file.exists():
            try:
                return cache_file.read_text()
            except (OSError, UnicodeDecodeError) as e:
                warnings.warn(
                    f"Failed to read cached motor file '{cache_file}': {e}. "
                    "Fetching fresh data from API."
                )

        base_url = "https://www.thrustcurve.org/api/v1"
        # Step 1. Search motor
        response = requests.get(f"{base_url}/search.json", params={"commonName": name})
        response.raise_for_status()
        data = response.json()

        if not data.get("results"):
            raise ValueError(
                f"No motor found for name '{name}'. "
                "Please verify the motor name format (e.g., 'Cesaroni_M1670' or 'M1670') and try again."
            )

        motor_info = data["results"][0]
        motor_id = motor_info.get("motorId")
        # NOTE: commented bc we don't use it, but keeping for possible future use
        # designation = motor_info.get("designation", "").replace("/", "-")
        # manufacturer = motor_info.get("manufacturer", "")

        # Step 2. Download the .eng file
        dl_response = requests.get(
            f"{base_url}/download.json",
            params={"motorIds": motor_id, "format": "RASP", "data": "file"},
        )
        dl_response.raise_for_status()
        dl_data = dl_response.json()

        if not dl_data.get("results"):
            raise ValueError(
                f"No .eng file found for motor '{name}' in the ThrustCurve API."
            )

        data_base64 = dl_data["results"][0].get("data")
        if not data_base64:
            raise ValueError(
                f"Downloaded .eng data for motor '{name}' is empty or invalid."
            )
        if not no_cache:
            try:
                cache_file.write_text(data_base64)
            except (OSError, PermissionError) as e:
                warnings.warn(
                    f"Could not write to cache file '{cache_file}': {e}. "
                    "Continuing without caching.",
                    RuntimeWarning,
                )

        return data_base64

    @staticmethod
    def load_from_thrustcurve_api(name: str, no_cache: bool = False, **kwargs):
        """
        Creates a Motor instance by downloading a .eng file from the ThrustCurve API
        based on the given motor name.

        Parameters
        ----------
        name : str
            The motor name according to the API (e.g., "Cesaroni_M1670" or "M1670").
            Both manufacturer-prefixed and shorthand names are commonly used; if multiple
            motors match the search, the first result is used.
        **kwargs :
            Additional arguments passed to the Motor constructor or loader, such as
            dry_mass, nozzle_radius, etc.

        Returns
        -------
        instance : GenericMotor
            A new GenericMotor instance initialized using the downloaded .eng file.

        Raises
        ------
        ValueError
            If no motor is found or if the downloaded .eng data is missing.
        requests.exceptions.RequestException
            If a network or HTTP error occurs during the API call.
        """

        data_base64 = GenericMotor._call_thrustcurve_api(name, no_cache=no_cache)
        data_bytes = base64.b64decode(data_base64)

        # Step 3. Create the motor from the .eng file
        tmp_path = None
        try:
            # create a temporary file that persists until we explicitly remove it
            with tempfile.NamedTemporaryFile(suffix=".eng", delete=False) as tmp_file:
                tmp_file.write(data_bytes)
                tmp_file.flush()
                tmp_path = tmp_file.name

            return GenericMotor.load_from_eng_file(tmp_path, **kwargs)
        finally:
            # Ensuring the temporary file is removed
            if tmp_path and path.exists(tmp_path):
                try:
                    remove(tmp_path)
                except OSError:
                    # If cleanup fails, don't raise: we don't want to mask prior exceptions.
                    pass

    def all_info(self):
        """Prints out all data and graphs available about the Motor."""
        # Print motor details
        self.prints.all()
        self.plots.all()

    def to_dict(self, **kwargs):
        data = super().to_dict(**kwargs)
        data.update(
            {
                "chamber_radius": self.chamber_radius,
                "chamber_height": self.chamber_height,
                "chamber_position": self.chamber_position,
                "propellant_initial_mass": self.propellant_initial_mass,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data):
        return cls(
            thrust_source=data["thrust_source"],
            burn_time=data["burn_time"],
            chamber_radius=data["chamber_radius"],
            chamber_height=data["chamber_height"],
            chamber_position=data["chamber_position"],
            propellant_initial_mass=data["propellant_initial_mass"],
            nozzle_radius=data["nozzle_radius"],
            dry_mass=data["dry_mass"],
            center_of_dry_mass_position=data["center_of_dry_mass_position"],
            dry_inertia=(
                data["dry_I_11"],
                data["dry_I_22"],
                data["dry_I_33"],
                data["dry_I_12"],
                data["dry_I_13"],
                data["dry_I_23"],
            ),
            nozzle_position=data["nozzle_position"],
            interpolation_method=data["interpolate"],
            reference_pressure=data.get("reference_pressure"),
        )
