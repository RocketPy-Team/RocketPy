from functools import cached_property

from ..mathutils.function import Function, funcify_method, reset_funcified_methods
from ..mathutils.inertia import (
    parallel_axis_theorem_I11,
    parallel_axis_theorem_I12,
    parallel_axis_theorem_I13,
    parallel_axis_theorem_I22,
    parallel_axis_theorem_I23,
    parallel_axis_theorem_I33,
)
from ..mathutils.vector_matrix import Vector
from ..plots.hybrid_motor_plots import _HybridMotorPlots
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
    SolidMotor.only_radial_burn : bool
        If True, grain regression is restricted to radial burn only (inner radius growth).
        Grain length remains constant throughout the burn. Default is True.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        thrust_source,
        dry_mass,
        dry_inertia,
        nozzle_radius,
        burn_time,
        center_of_dry_mass_position,
        grain_number,
        grain_separation,
        grain_density,
        grain_outer_radius,
        grain_initial_inner_radius,
        grain_initial_height,
        grains_center_of_mass_position,
        tanks_mass,
        oxidizer_initial_mass,
        oxidizer_mass_flow_rate_curve,
        oxidizer_density,
        oxidizer_tanks_geometries,
        oxidizer_tanks_positions,
        oxidizer_initial_inertia=(0, 0, 0),
        nozzle_position=0,
        reshape_thrust_curve=False,
        interpolation_method="linear",
        coordinate_system_orientation="nozzle_to_combustion_chamber",
        reference_pressure=None,
        only_radial_burn=True,
    ):
        """Initializes HybridMotor class, process thrust curve and geometrical
        parameters and store results.

        Parameters
        ----------
        thrust_source : int, float, callable, string, numpy.ndarray, list
            Motor's thrust curve. Can be given as Thrust-Time pairs array or
            file path (csv or eng format). Is passed to the Function class, see
            help(Function) for more information. Thrust units are Newtons, time
            units are seconds.
        dry_mass : int, float
            Motor's dry mass in kg. This is the mass of the motor without
            propellant.
        dry_inertia : tuple, list
            Tuple or list containing the motor's dry mass inertia tensor
            components, in kg*m^2. This inertia is defined with respect to the
            motor's center of dry mass position. Assuming e_3 is the motor's axis
            of symmetry, e_1 and e_2 are orthogonal and form a plane
            perpendicular to e_3, the dry mass inertia tensor components must be
            given in the following order: (I_11, I_22, I_33, I_12, I_13, I_23).
            Alternatively, the inertia tensor can be given as (I_11, I_22, I_33),
            where I_12 = I_13 = I_23 = 0.
        nozzle_radius : int, float
            Motor's nozzle radius in meters.
        burn_time: int, float, tuple of int, float
            Motor's burn time.
            If a tuple is passed, the first value is the ignition time and the
            second value is the end of burn time. If a single number is passed,
            the ignition time is assumed to be 0 and the end of burn time is
            the number passed.
        center_of_dry_mass_position : int, float
            Position of the motor's center of dry mass (i.e. center of mass
            without propellant) relative to the motor's coordinate system
            origin, in meters. See the ``coordinate_system_orientation``
            parameter for details on the coordinate system.
        grain_number : int
            Number of solid grains.
        grain_separation : int, float
            Distance between grains, in meters.
        grain_density : int, float
            Solid grain density in kg/m³.
        grain_outer_radius : int, float
            Solid grain outer radius in meters.
        grain_initial_inner_radius : int, float
            Solid grain initial inner radius in meters.
        grain_initial_height : int, float
            Solid grain initial height in meters.
        grains_center_of_mass_position : int, float
            Position of the center of mass of the grains relative to the motor's
            coordinate system origin in meters. Generally equal to
            ``center_of_dry_mass_position``.
        grain_initial_inertia : tuple, list, optional
            Tuple or list containing the initial inertia tensor components of a
            single grain, in kg*m^2. This inertia is defined with respect to the
            the grains_center_of_mass_position position. If not specified, the
            grain is assumed to be a hollow cylinder with the initial dimensions.
            Assuming e_3 is the grain's axis of symmetry, e_1 and e_2 are
            orthogonal and form a plane perpendicular to e_3, the initial inertia
            tensor components must be given in the following order:
            (I_11, I_22, I_33, I_12, I_13, I_23). Alternatively, the inertia
            tensor can be given as (I_11, I_22, I_33), where I_12 = I_13 = I_23 = 0.
            Default is (0, 0, 0).
        tanks_mass : float
            Total mass of the oxidizer tanks structures in kg. Includes the mass
            of the tanks themselves, valves, pipes, etc. It is assumed constant
            over time.
        oxidizer_initial_mass : float
            Initial mass of the oxidizer, including liquid and gas phases, in kg.
        oxidizer_mass_flow_rate_curve : int, float, callable, string, numpy.ndarray, list
            Oxidizer mass flow rate curve. Can be given as MassFlowRate-Time
            pairs array or file path (csv format). It is used to calculate the
            oxidizer mass and center of mass position as a function of time.
            If int or float is given, it is assumed constant. Mass flow rate
            units are kg/s, time units are seconds. Passed to the Function
            class, see help(Function) for more information.
        oxidizer_density : float
            Density of the oxidizer in kg/m³. It is used to calculate the volume
            and height of the oxidizer in the tanks. It is assumed constant over
            time.
        oxidizer_tanks_geometries : list
            List of tuples, where each tuple represents the geometry of an
            oxidizer tank. Accepted geometries are:
            ('cylinder', (top_radius, bottom_radius, height))
            ('sphere', radius)
            ('ullage', volume)
            Dimensions should be in meters and volume in cubic meters.
            The list must contain at least one tank geometry. Ullage tanks can only be
            placed at the top or bottom of the tanks stack.
            Example: [('ullage', 0.01), ('cylinder', (0.1, 0.1, 0.5)), ('cylinder', (0.1, 0.05, 0.2))]
        oxidizer_tanks_positions : list
            List of floats, representing the position of the centroid of each
            oxidizer tank's geometry with respect to the motor's coordinate system
            origin, in meters. The list must have the same length as
            ``oxidizer_tanks_geometries``.
            See the ``coordinate_system_orientation`` parameter for details on the coordinate system.
        oxidizer_tanks_initial_liquid_level : float, optional
            Initial liquid level in the tanks, measured in meters from the bottom
            of the tanks stack. If specified, this parameter overrides the initial
            oxidizer mass calculation based on ``oxidizer_initial_mass``, allowing
            precise control over the starting volume of the liquid oxidizer. If
            None, the initial liquid level is derived from ``oxidizer_initial_mass``.
            Default is None.
        oxidizer_tanks_initial_ullage_mass : float, optional
            Initial mass of the ullage gas in kg. If not specified, it is assumed
            to be 0. Default is 0.
        oxidizer_tanks_initial_ullage_volume : float, optional
            Initial volume of the ullage gas in cubic meters. If not specified, it
            is automatically calculated based on the tanks geometries and the
            initial liquid level. Default is None.
        oxidizer_initial_inertia : tuple, list, optional
            Tuple or list containing the initial inertia tensor components of the
            oxidizer (liquid + gas), in kg*m^2. This inertia is defined with
            respect to the initial oxidizer center of mass position. If not
            specified, the oxidizer is assumed to be a point mass. Default is (0, 0, 0).
            Assuming e_3 is the motor's axis of symmetry, e_1 and e_2 are
            orthogonal and form a plane perpendicular to e_3, the initial inertia
            tensor components must be given in the following order:
            (I_11, I_22, I_33, I_12, I_13, I_23). Alternatively, the inertia
            tensor can be given as (I_11, I_22, I_33), where I_12 = I_13 = I_23 = 0.
        nozzle_position : int, float, optional
            Motor's nozzle outlet position in meters, specified in the motor's
            coordinate system. Default is 0, which corresponds to the motor's
            origin. See the ``coordinate_system_orientation`` parameter for
            details on the coordinate system.
        reshape_thrust_curve : boolean, tuple, optional
            If False, the original thrust curve supplied is used. If a tuple is
            given, the thrust curve is reshaped to match the new grain mass
            flow rate and burn time. The tuple should contain the initial grain
            mass and the final grain mass, in kg. Default is False.
        interpolation_method : string, optional
            Method of interpolation to be used in case thrust curve is given
            by data points. Options are 'spline', 'akima' and 'linear'.
            Default is "linear".
        coordinate_system_orientation : string, optional
            Orientation of the motor's coordinate system. The coordinate system
            has its origin at the motor's center of mass and is oriented
            according to the following options:
            "nozzle_to_combustion_chamber" : the coordinate system is oriented
            with the z-axis pointing from the nozzle towards the combustion
            chamber.
            "combustion_chamber_to_nozzle" : the coordinate system is oriented
            with the z-axis pointing from the combustion chamber towards the
            nozzle.
            Default is "nozzle_to_combustion_chamber".
        reference_pressure : int, float, optional
            Reference pressure in Pa used to calculate vacuum thrust and
            pressure thrust. This corresponds to the atmospheric pressure
            measured during the static test of the motor. Default is None,
            which means no pressure correction is applied.

        Returns
        -------
        None
        """
        # Call SolidMotor init to initialize grain parameters
        # Note: dry_mass and dry_inertia are temporarily set to 0, they will be
        # calculated later considering the tanks mass.
        SolidMotor.__init__(
            self,
            thrust_source=thrust_source,
            dry_mass=0,
            dry_inertia=(0, 0, 0),
            nozzle_radius=nozzle_radius,
            burn_time=burn_time,
            center_of_dry_mass_position=center_of_dry_mass_position,
            grain_number=grain_number,
            grain_separation=grain_separation,
            grain_density=grain_density,
            grain_outer_radius=grain_outer_radius,
            grain_initial_inner_radius=grain_initial_inner_radius,
            grain_initial_height=grain_initial_height,
            grains_center_of_mass_position=grains_center_of_mass_position,
            nozzle_position=nozzle_position,
            reshape_thrust_curve=reshape_thrust_curve,
            interpolation_method=interpolation_method,
            coordinate_system_orientation=coordinate_system_orientation,
            reference_pressure=reference_pressure,
        )

        # Oxidizer parameters initialization
        self.tanks_mass = tanks_mass
        self.oxidizer_initial_mass = oxidizer_initial_mass
        self.oxidizer_density = oxidizer_density
        self.oxidizer_tanks_geometries = oxidizer_tanks_geometries
        self.oxidizer_tanks_positions = oxidizer_tanks_positions
        self.oxidizer_initial_inertia = (
            (*oxidizer_initial_inertia, 0, 0, 0)
            if len(oxidizer_initial_inertia) == 3
            else oxidizer_initial_inertia
        )

        # Oxidizer mass flow rate definition and processing
        self.oxidizer_mass_flow_rate = Function(
            oxidizer_mass_flow_rate_curve,
            "Time (s)",
            "Oxidizer Mass Flow Rate (kg/s)",
            interpolation_method,
            extrapolation="zero",
        )

        # Correct dry mass and dry inertia to include tanks mass
        self.dry_mass = dry_mass + tanks_mass
        dry_inertia = (*dry_inertia, 0, 0, 0) if len(dry_inertia) == 3 else dry_inertia
        self.dry_I_11 = dry_inertia[0]
        self.dry_I_22 = dry_inertia[1]
        self.dry_I_33 = dry_inertia[2]
        self.dry_I_12 = dry_inertia[3]
        self.dry_I_13 = dry_inertia[4]
        self.dry_I_23 = dry_inertia[5]
        # TODO: Calculate tanks inertia tensor based on their geometry and mass
        """
        # Initialize Tanks object
        self.tanks = Tank(
            geometries=oxidizer_tanks_geometries,
            positions=oxidizer_tanks_positions,
            fluid_mass=self.oxidizer_mass,
            fluid_density=self.oxidizer_density,
            initial_liquid_level=oxidizer_tanks_initial_liquid_level,
            initial_ullage_mass=oxidizer_tanks_initial_ullage_mass,
            initial_ullage_volume=oxidizer_tanks_initial_ullage_volume,
        )
        """
        # Store important functions
        self.liquid_propellant_mass = self.tanks.liquid_mass
        self.gas_propellant_mass = self.tanks.gas_mass
        self.center_of_liquid_propellant_mass = self.tanks.liquid_center_of_mass
        self.center_of_gas_propellant_mass = self.tanks.gas_center_of_mass
        self.liquid_propellant_I_11 = self.tanks.liquid_I_11
        self.liquid_propellant_I_22 = self.tanks.liquid_I_22
        self.liquid_propellant_I_33 = self.tanks.liquid_I_33
        self.gas_propellant_I_11 = self.tanks.gas_I_11
        self.gas_propellant_I_22 = self.tanks.gas_I_22
        self.gas_propellant_I_33 = self.tanks.gas_I_33

        # Rename grain attributes for clarity
        self.grain_propellant_mass = self.grain_mass
        self.center_of_grain_propellant_mass = self.grains_center_of_mass_position
        self.grain_propellant_I_11 = self.grains_I_11
        self.grain_propellant_I_22 = self.grains_I_22
        self.grain_propellant_I_33 = self.grains_I_33

        # Overall propellant inertia tensor components relative to propellant CoM
        # We need to recalculate the total propellant CoM function first
        # (Assuming self.liquid_propellant_mass and self.grain_propellant_mass exist and are Functions)
        # (Assuming self.center_of_liquid_propellant_mass and self.center_of_grain_propellant_mass exist and are Functions returning scalars)
        self._propellant_mass = self.liquid_propellant_mass + self.grain_propellant_mass
        self._center_of_propellant_mass = (
            self.center_of_liquid_propellant_mass * self.liquid_propellant_mass
            + self.center_of_grain_propellant_mass * self.grain_propellant_mass
        ) / self._propellant_mass
        # Ensure division by zero is handled if needed, although propellant mass shouldn't be zero initially

        # Create Functions returning distance vectors relative to the overall propellant CoM
        liquid_com_to_prop_com = (
            self.center_of_liquid_propellant_mass - self._center_of_propellant_mass
        )
        grain_com_to_prop_com = (
            self.center_of_grain_propellant_mass - self._center_of_propellant_mass
        )

        # Convert scalar distances to 3D vectors for PAT functions
        # The distance is along the Z-axis in the motor's coordinate system
        liquid_dist_vec_func = Function(
            lambda t: Vector([0, 0, liquid_com_to_prop_com(t)]), inputs="t"
        )
        grain_dist_vec_func = Function(
            lambda t: Vector([0, 0, grain_com_to_prop_com(t)]), inputs="t"
        )

        # Apply PAT using the new specific functions
        # Inertias relative to component CoMs are needed (e.g., self.liquid_propellant_I_11_from_liquid_CM)
        # Assuming these exist, otherwise adjust the first argument of the PAT functions

        # --- I_11 ---
        # Assuming self.liquid_propellant_I_11 refers to inertia relative to liquid CoM
        liquid_I_11_prop_com = parallel_axis_theorem_I11(
            self.liquid_propellant_I_11,  # Inertia relative to liquid's own CoM
            self.liquid_propellant_mass,
            liquid_dist_vec_func,  # Distance from total prop CoM to liquid CoM
        )
        # Assuming self.grain_propellant_I_11 refers to inertia relative to grain CoM
        grain_I_11_prop_com = parallel_axis_theorem_I11(
            self.grain_propellant_I_11,  # Inertia relative to grain's own CoM
            self.grain_propellant_mass,
            grain_dist_vec_func,  # Distance from total prop CoM to grain CoM
        )
        self.propellant_I_11_from_propellant_CM = (
            liquid_I_11_prop_com + grain_I_11_prop_com
        )

        # --- I_22 ---
        liquid_I_22_prop_com = parallel_axis_theorem_I22(
            self.liquid_propellant_I_22,  # Inertia relative to liquid's own CoM
            self.liquid_propellant_mass,
            liquid_dist_vec_func,
        )
        grain_I_22_prop_com = parallel_axis_theorem_I22(
            self.grain_propellant_I_22,  # Inertia relative to grain's own CoM
            self.grain_propellant_mass,
            grain_dist_vec_func,
        )
        self.propellant_I_22_from_propellant_CM = (
            liquid_I_22_prop_com + grain_I_22_prop_com
        )

        # --- I_33 ---
        liquid_I_33_prop_com = parallel_axis_theorem_I33(
            self.liquid_propellant_I_33,  # Inertia relative to liquid's own CoM
            self.liquid_propellant_mass,
            liquid_dist_vec_func,
        )
        grain_I_33_prop_com = parallel_axis_theorem_I33(
            self.grain_propellant_I_33,  # Inertia relative to grain's own CoM
            self.grain_propellant_mass,
            grain_dist_vec_func,
        )
        self.propellant_I_33_from_propellant_CM = (
            liquid_I_33_prop_com + grain_I_33_prop_com
        )

        # --- Products of Inertia (I_12, I_13, I_23) ---
        # Assume components PoI are 0 relative to their own CoM due to axisymmetry
        # PAT calculation will correctly handle the axisymmetry (result should be 0)

        # I_12
        liquid_I_12_prop_com = parallel_axis_theorem_I12(
            Function(0), self.liquid_propellant_mass, liquid_dist_vec_func
        )
        grain_I_12_prop_com = parallel_axis_theorem_I12(
            Function(0), self.grain_propellant_mass, grain_dist_vec_func
        )
        # Store intermediate result if needed by Motor.__init__ later, prefix with '_' if not part of public API
        self._propellant_I_12_from_propellant_CM = (
            liquid_I_12_prop_com + grain_I_12_prop_com
        )

        # I_13
        liquid_I_13_prop_com = parallel_axis_theorem_I13(
            Function(0), self.liquid_propellant_mass, liquid_dist_vec_func
        )
        grain_I_13_prop_com = parallel_axis_theorem_I13(
            Function(0), self.grain_propellant_mass, grain_dist_vec_func
        )
        self._propellant_I_13_from_propellant_CM = (
            liquid_I_13_prop_com + grain_I_13_prop_com
        )

        # I_23
        liquid_I_23_prop_com = parallel_axis_theorem_I23(
            Function(0), self.liquid_propellant_mass, liquid_dist_vec_func
        )
        grain_I_23_prop_com = parallel_axis_theorem_I23(
            Function(0), self.grain_propellant_mass, grain_dist_vec_func
        )
        self._propellant_I_23_from_propellant_CM = (
            liquid_I_23_prop_com + grain_I_23_prop_com
        )

        # IMPORTANT: Call the parent __init__ AFTER calculating component inertias
        #            because the parent __init__ uses these calculated values.
        super().__init__(
            thrust_source=thrust_source,
            dry_mass=self.dry_mass,  # Use the corrected dry mass
            dry_inertia=(
                self.dry_I_11,
                self.dry_I_22,
                self.dry_I_33,
                self.dry_I_12,
                self.dry_I_13,
                self.dry_I_23,
            ),  # Use corrected dry inertia
            nozzle_radius=nozzle_radius,
            burn_time=burn_time,
            center_of_dry_mass_position=center_of_dry_mass_position,
            nozzle_position=nozzle_position,
            reshape_thrust_curve=reshape_thrust_curve,
            interpolation_method=interpolation_method,
            coordinate_system_orientation=coordinate_system_orientation,
            reference_pressure=reference_pressure,
        )
        # The parent __init__ will now correctly use the calculated
        # self.propellant_I_xx_from_propellant_CM values.

        # Initialize plots object specific to HybridMotor
        self.plots = _HybridMotorPlots(self)

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
    @property
    @funcify_method("Time (s)", "Inertia I_11 (kg m²)")
    def propellant_I_11(self):
        """Inertia tensor 11 component of the propellant, the inertia is
        relative to the e_1 axis, centered at the instantaneous propellant
        center of mass.
        """
        # Returns the value calculated in __init__
        return self.propellant_I_11_from_propellant_CM

    @property
    @funcify_method("Time (s)", "Inertia I_22 (kg m²)")
    def propellant_I_22(self):
        """Inertia tensor 22 component of the propellant... (Identical to I_11)"""

        return self.propellant_I_22_from_propellant_CM

    @property
    @funcify_method("Time (s)", "Inertia I_33 (kg m²)")
    def propellant_I_33(self):
        """Inertia tensor 33 component of the propellant..."""

        return self.propellant_I_33_from_propellant_CM

    @property
    @funcify_method("Time (s)", "Inertia I_12 (kg m²)")
    def propellant_I_12(self):
        """Inertia tensor 12 component of the propellant..."""

        return self._propellant_I_12_from_propellant_CM

    @property
    @funcify_method("Time (s)", "Inertia I_13 (kg m²)")
    def propellant_I_13(self):
        """Inertia tensor 13 component of the propellant..."""

        return self._propellant_I_13_from_propellant_CM

    @property
    @funcify_method("Time (s)", "Inertia I_23 (kg m²)")
    def propellant_I_23(self):
        """Inertia tensor 23 component of the propellant..."""

        return self._propellant_I_23_from_propellant_CM

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

    def to_dict(self, include_outputs=False):
        data = super().to_dict(include_outputs)
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

        if include_outputs:
            data.update(
                {
                    "grain_inner_radius": self.solid.grain_inner_radius,
                    "grain_height": self.solid.grain_height,
                    "burn_area": self.solid.burn_area,
                    "burn_rate": self.solid.burn_rate,
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
            tanks_mass=data["tanks_mass"],
            oxidizer_initial_mass=data["oxidizer_initial_mass"],
            oxidizer_mass_flow_rate_curve=data["oxidizer_mass_flow_rate_curve"],
            oxidizer_density=data["oxidizer_density"],
            oxidizer_tanks_geometries=data["oxidizer_tanks_geometries"],
            oxidizer_tanks_positions=data["oxidizer_tanks_positions"],
            reference_pressure=data.get("reference_pressure"),
        )
        return motor
