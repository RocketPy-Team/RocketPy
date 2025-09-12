from functools import cached_property

from rocketpy.tools import parallel_axis_theorem_from_com

from ..mathutils.function import Function, funcify_method, reset_funcified_methods
from ..plots.hybrid_motor_plots import _HybridMotorPlots
from ..prints.hybrid_motor_prints import _HybridMotorPrints
from .liquid_motor import LiquidMotor
from .motor import Motor
from .solid_motor import SolidMotor


class HybridMotor(Motor):
    """Class to specify characteristics and useful operations for Hybrid
    motors. This class inherits from the Motor class.

    See Also
    --------
    Motor

    Attributes
    ----------
    HybridMotor.coordinate_system_orientation : str
        Orientation of the motor's coordinate system. The coordinate system
        is defined by the motor's axis of symmetry. The origin of the
        coordinate system may be placed anywhere along such axis, such as
        at the nozzle area, and must be kept the same for all other
        positions specified. Options are "nozzle_to_combustion_chamber" and
        "combustion_chamber_to_nozzle".
    HybridMotor.nozzle_radius : float
        Radius of motor nozzle outlet in meters.
    HybridMotor.nozzle_area : float
        Area of motor nozzle outlet in square meters.
    HybridMotor.nozzle_position : float
        Motor's nozzle outlet position in meters, specified in the motor's
        coordinate system. See
        :doc:`Positions and Coordinate Systems </user/positions>` for more
        information.
    HybridMotor.throat_radius : float
        Radius of motor nozzle throat in meters.
    HybridMotor.solid : SolidMotor
        Solid motor object that composes the hybrid motor.
    HybridMotor.liquid : LiquidMotor
        Liquid motor object that composes the hybrid motor.
    HybridMotor.positioned_tanks : list
        List containing the motor's added tanks and their respective
        positions.
    HybridMotor.grains_center_of_mass_position : float
        Position of the center of mass of the grains in meters, specified in
        the motor's coordinate system.
        See :doc:`Positions and Coordinate Systems </user/positions>`
        for more information.
    HybridMotor.grain_number : int
        Number of solid grains.
    HybridMotor.grain_density : float
        Density of each grain in kg/meters cubed.
    HybridMotor.grain_outer_radius : float
        Outer radius of each grain in meters.
    HybridMotor.grain_initial_inner_radius : float
        Initial inner radius of each grain in meters.
    HybridMotor.grain_initial_height : float
        Initial height of each grain in meters.
    HybridMotor.grain_separation : float
        Distance between two grains in meters.
    HybridMotor.dry_mass : float
        Same as in Motor class. See the :class:`Motor <rocketpy.Motor>` docs.
    HybridMotor.propellant_initial_mass : float
        Total propellant initial mass in kg. This is the sum of the initial
        mass of fluids in each tank and the initial mass of the solid grains.
    HybridMotor.total_mass : Function
        Total motor mass in kg as a function of time, defined as the sum
        of the dry mass (motor's structure mass) and the propellant mass, which
        varies with time.
    HybridMotor.propellant_mass : Function
        Total propellant mass in kg as a function of time, this includes the
        mass of fluids in each tank and the mass of the solid grains.
    HybridMotor.structural_mass_ratio: float
        Initial ratio between the dry mass and the total mass.
    HybridMotor.total_mass_flow_rate : Function
        Time derivative of propellant total mass in kg/s as a function
        of time as obtained by the thrust source.
    HybridMotor.center_of_mass : Function
        Position of the motor center of mass in
        meters as a function of time.
        See :doc:`Positions and Coordinate Systems </user/positions>`
        for more information regarding the motor's coordinate system.
    HybridMotor.center_of_propellant_mass : Function
        Position of the motor propellant center of mass in meters as a
        function of time.
        See :doc:`Positions and Coordinate Systems </user/positions>`
        for more information regarding the motor's coordinate system.
    HybridMotor.I_11 : Function
        Component of the motor's inertia tensor relative to the e_1 axis
        in kg*m^2, as a function of time. The e_1 axis is the direction
        perpendicular to the motor body axis of symmetry, centered at
        the instantaneous motor center of mass.
    HybridMotor.I_22 : Function
        Component of the motor's inertia tensor relative to the e_2 axis
        in kg*m^2, as a function of time. The e_2 axis is the direction
        perpendicular to the motor body axis of symmetry, centered at
        the instantaneous motor center of mass.
        Numerically equivalent to I_11 due to symmetry.
    HybridMotor.I_33 : Function
        Component of the motor's inertia tensor relative to the e_3 axis
        in kg*m^2, as a function of time. The e_3 axis is the direction of
        the motor body axis of symmetry, centered at the instantaneous
        motor center of mass.
    HybridMotor.I_12 : Function
        Component of the motor's inertia tensor relative to the e_1 and
        e_2 axes in kg*m^2, as a function of time. See HybridMotor.I_11 and
        HybridMotor.I_22 for more information.
    HybridMotor.I_13 : Function
        Component of the motor's inertia tensor relative to the e_1 and
        e_3 axes in kg*m^2, as a function of time. See HybridMotor.I_11 and
        HybridMotor.I_33 for more information.
    HybridMotor.I_23 : Function
        Component of the motor's inertia tensor relative to the e_2 and
        e_3 axes in kg*m^2, as a function of time. See HybridMotor.I_22 and
        HybridMotor.I_33 for more information.
    HybridMotor.propellant_I_11 : Function
        Component of the propellant inertia tensor relative to the e_1
        axis in kg*m^2, as a function of time. The e_1 axis is the
        direction perpendicular to the motor body axis of symmetry,
        centered at the instantaneous propellant center of mass.
    HybridMotor.propellant_I_22 : Function
        Component of the propellant inertia tensor relative to the e_2
        axis in kg*m^2, as a function of time. The e_2 axis is the
        direction perpendicular to the motor body axis of symmetry,
        centered at the instantaneous propellant center of mass.
        Numerically equivalent to propellant_I_11 due to symmetry.
    HybridMotor.propellant_I_33 : Function
        Component of the propellant inertia tensor relative to the e_3
        axis in kg*m^2, as a function of time. The e_3 axis is the
        direction of the motor body axis of symmetry, centered at the
        instantaneous propellant center of mass.
    HybridMotor.propellant_I_12 : Function
        Component of the propellant inertia tensor relative to the e_1 and
        e_2 axes in kg*m^2, as a function of time. See
        HybridMotor.propellant_I_11 and HybridMotor.propellant_I_22 for
        more information.
    HybridMotor.propellant_I_13 : Function
        Component of the propellant inertia tensor relative to the e_1 and
        e_3 axes in kg*m^2, as a function of time. See
        HybridMotor.propellant_I_11 and HybridMotor.propellant_I_33 for
        more information.
    HybridMotor.propellant_I_23 : Function
        Component of the propellant inertia tensor relative to the e_2 and
        e_3 axes in kg*m^2, as a function of time. See
        HybridMotor.propellant_I_22 and HybridMotor.propellant_I_33 for
        more information.
    HybridMotor.thrust : Function
        Motor thrust force obtained from thrust source, in Newtons, as a
        function of time.
    HybridMotor.vacuum_thrust : Function
        Motor thrust force when the rocket is in a vacuum. In Newtons, as a
        function of time.
    HybridMotor.total_impulse : float
        Total impulse of the thrust curve in N*s.
    HybridMotor.max_thrust : float
        Maximum thrust value of the given thrust curve, in N.
    HybridMotor.max_thrust_time : float
        Time, in seconds, in which the maximum thrust value is achieved.
    HybridMotor.average_thrust : float
        Average thrust of the motor, given in N.
    HybridMotor.burn_time : tuple of float
        Tuple containing the initial and final time of the motor's burn time
        in seconds.
    HybridMotor.burn_start_time : float
        Motor burn start time, in seconds.
    HybridMotor.burn_out_time : float
        Motor burn out time, in seconds.
    HybridMotor.burn_duration : float
        Total motor burn duration, in seconds. It is the difference between the
        ``burn_out_time`` and the ``burn_start_time``.
    HybridMotor.exhaust_velocity : Function
        Effective exhaust velocity of the propulsion gases in m/s. Computed
        as the thrust divided by the mass flow rate. This corresponds to the
        actual exhaust velocity only when the nozzle exit pressure equals the
        atmospheric pressure.
    HybridMotor.burn_area : Function
        Total burn area considering all grains, made out of inner
        cylindrical burn area and grain top and bottom faces. Expressed
        in meters squared as a function of time.
    HybridMotor.Kn : Function
        Motor Kn as a function of time. Defined as burn_area divided by
        nozzle throat cross sectional area. Has no units.
    HybridMotor.burn_rate : Function
        Propellant burn rate in meter/second as a function of time.
    HybridMotor.interpolate : string
        Method of interpolation used in case thrust curve is given
        by data set in .csv or .eng, or as an array. Options are 'spline'
        'akima' and 'linear'. Default is "linear".
    HybridMotor.reference_pressure : int, float
        Atmospheric pressure in Pa at which the thrust data was recorded.
        It will allow to obtain the net thrust in the Flight class.
    """

    def __init__(  # pylint: disable=too-many-arguments
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
        nozzle_position=0,
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
        nozzle_radius : int, float
            Motor's nozzle outlet radius in meters.
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
            See :doc:`Positions and Coordinate Systems </user/positions>` for
            more information.
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
            the given float, in seconds.
            If a tuple of float is given, the burn time is assumed to be between
            the first and second elements of the tuple, in seconds.
            If not specified, automatically sourced as the range between the
            first and last-time step of the motor's thrust curve. This can only
            be used if the motor's thrust is defined by a list of points, such
            as a .csv file, a .eng file or a Function instance whose source is
            a list.
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
        self.liquid = LiquidMotor(
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
            reference_pressure,
        )
        self.solid = SolidMotor(
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
            nozzle_position,
            burn_time,
            throat_radius,
            reshape_thrust_curve,
            interpolation_method,
            coordinate_system_orientation,
            reference_pressure,
        )

        self.positioned_tanks = self.liquid.positioned_tanks
        self.grain_number = grain_number
        self.grain_density = grain_density
        self.grain_outer_radius = grain_outer_radius
        self.grain_initial_inner_radius = grain_initial_inner_radius
        self.grain_initial_height = grain_initial_height
        self.grain_separation = grain_separation
        self.grains_center_of_mass_position = grains_center_of_mass_position
        self.throat_radius = throat_radius

        # Initialize plots and prints object
        self.prints = _HybridMotorPrints(self)
        self.plots = _HybridMotorPlots(self)

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

    @funcify_method("Time (s)", "Mass (kg)")
    def propellant_mass(self):
        """Evaluates the total propellant mass of the motor as the sum
        of fluids mass in each tank and the grains mass.

        Returns
        -------
        Function
            Total propellant mass of the motor as a function of time, in kg.
        """
        return self.solid.propellant_mass + self.liquid.propellant_mass

    @cached_property
    def propellant_initial_mass(self):
        """Returns the initial propellant mass of the motor. See the docs of the
        HybridMotor.propellant_mass property for more information.

        Returns
        -------
        float
            Initial propellant mass of the motor, in kg.
        """
        return self.solid.propellant_initial_mass + self.liquid.propellant_initial_mass

    @funcify_method("Time (s)", "mass flow rate (kg/s)", extrapolation="zero")
    def mass_flow_rate(self):
        """Evaluates the mass flow rate of the motor as the sum of mass flow
        rates from all tanks and the solid grains mass flow rate.

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
        mass_balance = (
            self.solid.propellant_mass * self.solid.center_of_propellant_mass
            + self.liquid.propellant_mass * self.liquid.center_of_propellant_mass
        )
        return mass_balance / self.propellant_mass

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

        solid_mass = self.solid.propellant_mass
        liquid_mass = self.liquid.propellant_mass

        cm = self.center_of_propellant_mass
        solid_cm_to_cm = self.solid.center_of_propellant_mass - cm
        liquid_cm_to_cm = self.liquid.center_of_propellant_mass - cm

        solid_prop_inertia = self.solid.propellant_I_11
        liquid_prop_inertia = self.liquid.propellant_I_11

        I_11 = parallel_axis_theorem_from_com(
            solid_prop_inertia, solid_mass, solid_cm_to_cm
        ) + parallel_axis_theorem_from_com(
            liquid_prop_inertia, liquid_mass, liquid_cm_to_cm
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
        return self.solid.propellant_I_33 + self.liquid.propellant_I_33

    @funcify_method("Time (s)", "Inertia I_12 (kg m²)")
    def propellant_I_12(self):
        """Inertia tensor 12 component of the propellant, the inertia is
        relative to the e_1 and e_2 axes, centered at the instantaneous
        propellant center of mass.

        Returns
        -------
        Function
            Propellant inertia tensor 12 component at time t.

        Notes
        -----
            This is assumed to be zero due to axial symmetry of the motor. This
            could be improved in the future to account for the fact that the
            motor is not perfectly symmetric.
        """
        return 0

    @funcify_method("Time (s)", "Inertia I_13 (kg m²)")
    def propellant_I_13(self):
        """Inertia tensor 13 component of the propellant, the inertia is
        relative to the e_1 and e_3 axes, centered at the instantaneous
        propellant center of mass.

        Returns
        -------
        Function
            Propellant inertia tensor 13 component at time t.

        Notes
        -----
            This is assumed to be zero due to axial symmetry of the motor. This
            could be improved in the future to account for the fact that the
            motor is not perfectly symmetric.
        """
        return 0

    @funcify_method("Time (s)", "Inertia I_23 (kg m²)")
    def propellant_I_23(self):
        """Inertia tensor 23 component of the propellant, the inertia is
        relative to the e_2 and e_3 axes, centered at the instantaneous
        propellant center of mass.

        Returns
        -------
        Function
            Propellant inertia tensor 23 component at time t.

        Notes
        -----
            This is assumed to be zero due to axial symmetry of the motor. This
            could be improved in the future to account for the fact that the
            motor is not perfectly symmetric.
        """
        return 0

    def add_tank(self, tank, position):
        """Adds a tank to the motor.

        Parameters
        ----------
        tank : Tank
            Tank object to be added to the motor.
        position : float
            Position of the tank relative to the origin of the motor
            coordinate system. The tank reference point is its
            geometry zero reference point.

        See Also
        --------
        :ref:'Adding Tanks`

        Returns
        -------
        None
        """
        self.liquid.add_tank(tank, position)
        self.solid.mass_flow_rate = (
            self.total_mass_flow_rate - self.liquid.mass_flow_rate
        )
        reset_funcified_methods(self)

    def draw(self, *, filename=None):
        """Draws a representation of the HybridMotor.

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
                "grain_number": self.grain_number,
                "grain_density": self.grain_density,
                "grain_outer_radius": self.grain_outer_radius,
                "grain_initial_inner_radius": self.grain_initial_inner_radius,
                "grain_initial_height": self.grain_initial_height,
                "grain_separation": self.grain_separation,
                "grains_center_of_mass_position": self.grains_center_of_mass_position,
                "throat_radius": self.throat_radius,
                "positioned_tanks": [
                    {"tank": tank["tank"], "position": tank["position"]}
                    for tank in self.positioned_tanks
                ],
            }
        )

        if kwargs.get("include_outputs", False):
            burn_rate = self.solid.burn_rate
            if kwargs.get("discretize", False):
                burn_rate = burn_rate.set_discrete_based_on_model(
                    self.thrust, mutate_self=False
                )
            data.update(
                {
                    "grain_inner_radius": self.solid.grain_inner_radius,
                    "grain_height": self.solid.grain_height,
                    "burn_area": self.solid.burn_area,
                    "burn_rate": burn_rate,
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
            interpolation_method=data["interpolate"],
            coordinate_system_orientation=data["coordinate_system_orientation"],
            grain_number=data["grain_number"],
            grain_density=data["grain_density"],
            grain_outer_radius=data["grain_outer_radius"],
            grain_initial_inner_radius=data["grain_initial_inner_radius"],
            grain_initial_height=data["grain_initial_height"],
            grain_separation=data["grain_separation"],
            grains_center_of_mass_position=data["grains_center_of_mass_position"],
            nozzle_position=data["nozzle_position"],
            throat_radius=data["throat_radius"],
            reference_pressure=data.get("reference_pressure"),
        )

        for tank in data["positioned_tanks"]:
            motor.add_tank(tank["tank"], tank["position"])

        return motor
