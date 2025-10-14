from functools import cached_property

import numpy as np

from rocketpy.mathutils.function import funcify_method, reset_funcified_methods
from rocketpy.tools import parallel_axis_theorem_from_com

from ..plots.liquid_motor_plots import _LiquidMotorPlots
from ..prints.liquid_motor_prints import _LiquidMotorPrints
from .motor import Motor


class LiquidMotor(Motor):
    """Class to specify characteristics and useful operations for Liquid
    motors. This class inherits from the Motor class.

    See Also
    --------
    Motor

    Attributes
    ----------
    LiquidMotor.coordinate_system_orientation : str
        Orientation of the motor's coordinate system. The coordinate system
        is defined by the motor's axis of symmetry. The origin of the
        coordinate system may be placed anywhere along such axis, such as
        at the nozzle area, and must be kept the same for all other
        positions specified. Options are "nozzle_to_combustion_chamber" and
        "combustion_chamber_to_nozzle".
    LiquidMotor.nozzle_radius : float
        Radius of motor nozzle outlet in meters.
    LiquidMotor.nozzle_area : float
        Area of motor nozzle outlet in square meters.
    LiquidMotor.nozzle_position : float
        Motor's nozzle outlet position in meters, specified in the motor's
        coordinate system. See
        :doc:`Positions and Coordinate Systems </user/positions>` for more
        information.
    LiquidMotor.positioned_tanks : list
        List containing the motor's added tanks and their respective
        positions.
    LiquidMotor.dry_mass : float
        Same as in Motor class. See the :class:`Motor <rocketpy.Motor>` docs.
    LiquidMotor.propellant_initial_mass : float
        Total propellant initial mass in kg, includes fuel and oxidizer.
    LiquidMotor.total_mass : Function
        Total motor mass in kg as a function of time, defined as the sum
        of propellant mass and the motor's dry mass (i.e. structure mass).
    LiquidMotor.propellant_mass : Function
        Total propellant mass in kg as a function of time, includes fuel
        and oxidizer.
    LiquidMotor.structural_mass_ratio: float
        Initial ratio between the dry mass and the total mass.
    LiquidMotor.total_mass_flow_rate : Function
        Time derivative of propellant total mass in kg/s as a function
        of time as obtained by the tanks mass flow.
    LiquidMotor.center_of_mass : Function
        Position of the motor center of mass in
        meters as a function of time.
        See :doc:`Positions and Coordinate Systems </user/positions>`
        for more information regarding the motor's coordinate system.
    LiquidMotor.center_of_propellant_mass : Function
        Position of the motor propellant center of mass in meters as a
        function of time.
        See :doc:`Positions and Coordinate Systems </user/positions>`
        for more information regarding the motor's coordinate system.
    LiquidMotor.I_11 : Function
        Component of the motor's inertia tensor relative to the e_1 axis
        in kg*m^2, as a function of time. The e_1 axis is the direction
        perpendicular to the motor body axis of symmetry, centered at
        the instantaneous motor center of mass.
    LiquidMotor.I_22 : Function
        Component of the motor's inertia tensor relative to the e_2 axis
        in kg*m^2, as a function of time. The e_2 axis is the direction
        perpendicular to the motor body axis of symmetry, centered at
        the instantaneous motor center of mass.
        Numerically equivalent to I_11 due to symmetry.
    LiquidMotor.I_33 : Function
        Component of the motor's inertia tensor relative to the e_3 axis
        in kg*m^2, as a function of time. The e_3 axis is the direction of
        the motor body axis of symmetry, centered at the instantaneous
        motor center of mass.
    LiquidMotor.I_12 : Function
        Component of the motor's inertia tensor relative to the e_1 and
        e_2 axes in kg*m^2, as a function of time. See LiquidMotor.I_11 and
        LiquidMotor.I_22 for more information.
    LiquidMotor.I_13 : Function
        Component of the motor's inertia tensor relative to the e_1 and
        e_3 axes in kg*m^2, as a function of time. See LiquidMotor.I_11 and
        LiquidMotor.I_33 for more information.
    LiquidMotor.I_23 : Function
        Component of the motor's inertia tensor relative to the e_2 and
        e_3 axes in kg*m^2, as a function of time. See LiquidMotor.I_22 and
        LiquidMotor.I_33 for more information.
    LiquidMotor.propellant_I_11 : Function
        Component of the propellant inertia tensor relative to the e_1
        axis in kg*m^2, as a function of time. The e_1 axis is the
        direction perpendicular to the motor body axis of symmetry,
        centered at the instantaneous propellant center of mass.
    LiquidMotor.propellant_I_22 : Function
        Component of the propellant inertia tensor relative to the e_2
        axis in kg*m^2, as a function of time. The e_2 axis is the
        direction perpendicular to the motor body axis of symmetry,
        centered at the instantaneous propellant center of mass.
        Numerically equivalent to propellant_I_11 due to symmetry.
    LiquidMotor.propellant_I_33 : Function
        Component of the propellant inertia tensor relative to the e_3
        axis in kg*m^2, as a function of time. The e_3 axis is the
        direction of the motor body axis of symmetry, centered at the
        instantaneous propellant center of mass.
    LiquidMotor.propellant_I_12 : Function
        Component of the propellant inertia tensor relative to the e_1 and
        e_2 axes in kg*m^2, as a function of time. See
        LiquidMotor.propellant_I_11 and LiquidMotor.propellant_I_22 for
        more information.
    LiquidMotor.propellant_I_13 : Function
        Component of the propellant inertia tensor relative to the e_1 and
        e_3 axes in kg*m^2, as a function of time. See
        LiquidMotor.propellant_I_11 and LiquidMotor.propellant_I_33 for
        more information.
    LiquidMotor.propellant_I_23 : Function
        Component of the propellant inertia tensor relative to the e_2 and
        e_3 axes in kg*m^2, as a function of time. See
        LiquidMotor.propellant_I_22 and LiquidMotor.propellant_I_33 for
        more information.
    LiquidMotor.thrust : Function
        Motor thrust force obtained from thrust source, in Newtons, as a
        function of time.
    LiquidMotor.vacuum_thrust : Function
        Motor thrust force when the rocket is in a vacuum. In Newtons, as a
        function of time.
    LiquidMotor.total_impulse : float
        Total impulse of the thrust curve in N*s.
    LiquidMotor.max_thrust : float
        Maximum thrust value of the given thrust curve, in N.
    LiquidMotor.max_thrust_time : float
        Time, in seconds, in which the maximum thrust value is achieved.
    LiquidMotor.average_thrust : float
        Average thrust of the motor, given in N.
    LiquidMotor.burn_time : tuple of float
        Tuple containing the initial and final time of the motor's burn time
        in seconds.
    LiquidMotor.burn_start_time : float
        Motor burn start time, in seconds.
    LiquidMotor.burn_out_time : float
        Motor burn out time, in seconds.
    LiquidMotor.burn_duration : float
        Total motor burn duration, in seconds. It is the difference between the
        burn_out_time and the burn_start_time.
    LiquidMotor.exhaust_velocity : Function
        Effective exhaust velocity of the propulsion gases in m/s. Computed
        as the thrust divided by the mass flow rate. This corresponds to the
        actual exhaust velocity only when the nozzle exit pressure equals the
        atmospheric pressure.
    LiquidMotor.reference_pressure : int, float
        Atmospheric pressure in Pa at which the thrust data was recorded.
        It will allow to obtain the net thrust in the Flight class.
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
        reference_pressure=None,
    ):
        """Initialize LiquidMotor class, process thrust curve and geometrical
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
            Same as in Motor class. See the :class:`Motor <rocketpy.Motor>` docs.
        dry_inertia : tuple, list
            Tuple or list containing the motor's dry mass inertia tensor
            components, in kg*m^2. This inertia is defined with respect to the
            the ``center_of_dry_mass_position`` position.
            Assuming e_3 is the rocket's axis of symmetry, e_1 and e_2 are
            orthogonal and form a plane perpendicular to e_3, the dry mass
            inertia tensor components must be given in the following order:
            (I_11, I_22, I_33, I_12, I_13, I_23), where I_ij is the
            component of the inertia tensor in the direction of e_i x e_j.
            Alternatively, the inertia tensor can be given as
            (I_11, I_22, I_33), where I_12 = I_13 = I_23 = 0.
        nozzle_radius : int, float
            Motor's nozzle outlet radius in meters.
        center_of_dry_mass_position : int, float
            The position, in meters, of the motor's center of mass with respect
            to the motor's coordinate system when it is devoid of propellant.
            See :doc:`Positions and Coordinate Systems </user/positions>`
        nozzle_position : float
            Motor's nozzle outlet position in meters, specified in the motor's
            coordinate system. See
            :doc:`Positions and Coordinate Systems </user/positions>` for
            more information.
        burn_time: float, tuple of float, optional
            Motor's burn time.
            If a float is given, the burn time is assumed to be between 0 and
            the given float, in seconds.
            If a tuple of float is given, the burn time is assumed to be between
            the first and second elements of the tuple, in seconds.
            If not specified, automatically sourced as the range between the
            first and last-time step of the motor's thrust curve. This can only
            be used if the motor's thrust is defined by a list of points, such
            as a .csv file, a .eng file or a Function instance whose source is
            a list.
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
            positions specified. Options are "nozzle_to_combustion_chamber"
            and "combustion_chamber_to_nozzle". Default is
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

        self.positioned_tanks = []

        # Initialize plots and prints object
        self.prints = _LiquidMotorPrints(self)
        self.plots = _LiquidMotorPlots(self)

    @funcify_method("Time (s)", "Exhaust Velocity (m/s)")
    def exhaust_velocity(self):
        """Computes the exhaust velocity of the motor from its mass flow
        rate and thrust.

        Returns
        -------
        self.exhaust_velocity : Function
            Gas exhaust velocity of the motor.

        Notes
        -----
        The exhaust velocity is computed as the ratio of the thrust and the
        mass flow rate. Therefore, this will vary with time if the mass flow
        rate varies with time. This corresponds to the actual exhaust velocity
        only when the nozzle exit pressure equals the atmospheric pressure.
        """
        times, thrusts = self.thrust.source[:, 0], self.thrust.source[:, 1]
        mass_flow_rates = self.mass_flow_rate(times)
        exhaust_velocity = np.zeros_like(mass_flow_rates)

        # Compute exhaust velocity only for non-zero mass flow rates
        valid_indices = mass_flow_rates != 0

        exhaust_velocity[valid_indices] = (
            -thrusts[valid_indices] / mass_flow_rates[valid_indices]
        )

        return np.column_stack([times, exhaust_velocity])

    @funcify_method("Time (s)", "Propellant Mass (kg)")
    def propellant_mass(self):
        """Evaluates the total propellant mass of the motor as the sum of fluids
        mass in each tank, which may include fuel and oxidizer and usually vary
        with time.

        Returns
        -------
        Function
            Mass of the motor, in kg.
        """
        propellant_mass = 0

        for positioned_tank in self.positioned_tanks:
            propellant_mass += positioned_tank.get("tank").fluid_mass

        return propellant_mass

    @cached_property
    def propellant_initial_mass(self):
        """Property to store the initial mass of the propellant, this includes
        fuel and oxidizer.

        Returns
        -------
        float
            Initial mass of the propellant, in kg.
        """
        return self.propellant_mass(self.burn_start_time)

    @funcify_method("Time (s)", "Mass flow rate (kg/s)", extrapolation="zero")
    def mass_flow_rate(self):
        """Evaluates the mass flow rate of the motor as the sum of mass flow
        rate from each tank, which may include fuel and oxidizer and usually
        vary with time.

        Returns
        -------
        Function
            Mass flow rate of the motor, in kg/s.

        See Also
        --------
        Motor.total_mass_flow_rate :
            Calculates the total mass flow rate of the motor assuming
            constant exhaust velocity.
        """
        mass_flow_rate = 0

        for positioned_tank in self.positioned_tanks:
            mass_flow_rate += positioned_tank.get("tank").net_mass_flow_rate

        return mass_flow_rate

    @funcify_method("Time (s)", "Center of mass (m)")
    def center_of_propellant_mass(self):
        """Evaluates the center of mass of the motor from each tank center of
        mass and positioning. The center of mass height is measured relative to
        the origin of the motor's coordinate system.

        Returns
        -------
        Function
            Position of the propellant center of mass, in meters.
        """
        total_mass = 0
        mass_balance = 0

        for positioned_tank in self.positioned_tanks:
            tank = positioned_tank.get("tank")
            tank_position = positioned_tank.get("position")
            total_mass += tank.fluid_mass
            mass_balance += tank.fluid_mass * (tank_position + tank.center_of_mass)

        return mass_balance / total_mass

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
        I_11 = 0
        center_of_mass = self.center_of_propellant_mass

        for positioned_tank in self.positioned_tanks:
            tank = positioned_tank.get("tank")
            tank_position = positioned_tank.get("position")
            distance = tank_position + tank.center_of_mass - center_of_mass
            I_11 += parallel_axis_theorem_from_com(
                tank.inertia, tank.fluid_mass, distance
            )

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
        return 0

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
        """Adds a tank to the rocket motor.

        Parameters
        ----------
        tank : Tank
            Tank object to be added to the rocket motor.
        position : float
            Position of the tank relative to the origin of the motor
            coordinate system. The tank reference point is its
            geometry zero reference point.

        See Also
        --------
        :ref:`Adding Tanks`
        """
        self.positioned_tanks.append({"tank": tank, "position": position})
        reset_funcified_methods(self)

    def draw(self, *, filename=None):
        """Draw a representation of the LiquidMotor.

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
                "positioned_tanks": [
                    {"tank": tank["tank"], "position": tank["position"]}
                    for tank in self.positioned_tanks
                ],
            }
        )
        return data

    @classmethod
    def from_dict(cls, data):
        motor = cls(
            thrust_source=data["thrust_source"],
            burn_time=data["burn_time"],
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
            coordinate_system_orientation=data["coordinate_system_orientation"],
            reference_pressure=data.get("reference_pressure"),
        )

        for tank in data["positioned_tanks"]:
            motor.add_tank(tank["tank"], tank["position"])

        return motor
