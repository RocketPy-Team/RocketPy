# -*- coding: utf-8 -*-

__author__ = "Giovani Hidalgo Ceotto, Oscar Mauricio Prada Ramirez, João Lemes Gribel Soares, Lucas Kierulff Balabram, Lucas Azevedo Pezente"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

try:
    from functools import cached_property
except ImportError:
    from rocketpy.tools import cached_property

from rocketpy.Function import funcify_method

from .LiquidMotor import LiquidMotor
from .Motor import Motor
from .SolidMotor import SolidMotor


class HybridMotor(Motor):
    """Class to specify characteristics and useful operations for Hybrid
    motors.

    Attributes
    ----------

        Geometrical attributes:
        Motor.coordinateSystemOrientation : str
            Orientation of the motor's coordinate system. The coordinate system
            is defined by the motor's axis of symmetry. The origin of the
            coordinate system  may be placed anywhere along such axis, such as
            at the nozzle area, and must be kept the same for all other
            positions specified. Options are "nozzleToCombustionChamber" and
            "combustionChamberToNozzle".
        Motor.nozzleRadius : float
            Radius of motor nozzle outlet in meters.
        Motor.nozzlePosition : float
            Motor's nozzle outlet position in meters, specified in the motor's
            coordinate system. See `Motor.coordinateSystemOrientation` for
            more information.
        Motor.throatRadius : float
            Radius of motor nozzle throat in meters.
        Motor.solid : SolidMotor
            Solid motor object that composes the hybrid motor.
        Motor.liquid : LiquidMotor
            Liquid motor object that composes the hybrid motor.

        Mass and moment of inertia attributes:
        Motor.dry_mass : float
            The total mass of the motor structure, including chambers
            and tanks, when it is empty and does not contain any propellant.
        Motor.propellantInitialMass : float
            Total propellant initial mass in kg.
        Motor.totalMass : Function
            Total motor mass in kg as a function of time, defined as the sum
            of propellant and dry mass.
        Motor.propellantMass : Function
            Total propellant mass in kg as a function of time.
        Motor.totalMassFlowRate : Function
            Time derivative of propellant total mass in kg/s as a function
            of time as obtained by the thrust source.
        Motor.centerOfMass : Function
            Position of the motor center of mass in
            meters as a function of time.
            See `Motor.coordinateSystemOrientation` for more information
            regarding the motor's coordinate system.
        Motor.centerOfPropellantMass : Function
            Position of the motor propellant center of mass in meters as a
            function of time.
            See `Motor.coordinateSystemOrientation` for more information
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
        Motor.totalImpulse : float
            Total impulse of the thrust curve in N*s.
        Motor.maxThrust : float
            Maximum thrust value of the given thrust curve, in N.
        Motor.maxThrustTime : float
            Time, in seconds, in which the maximum thrust value is achieved.
        Motor.averageThrust : float
            Average thrust of the motor, given in N.
        Motor.burn_time : tuple of float
            Tuple containing the initial and final time of the motor's burn time
            in seconds.
        Motor.burnStartTime : float
            Motor burn start time, in seconds.
        Motor.burnOutTime : float
            Motor burn out time, in seconds.
        Motor.burnDuration : float
            Total motor burn duration, in seconds. It is the difference between the burnOutTime and the burnStartTime.
        Motor.exhaustVelocity : float
            Propulsion gases exhaust velocity, assumed constant, in m/s.
        Motor.burnArea : Function
            Total burn area considering all grains, made out of inner
            cylindrical burn area and grain top and bottom faces. Expressed
            in meters squared as a function of time.
        Motor.Kn : Function
            Motor Kn as a function of time. Defined as burnArea divided by
            nozzle throat cross sectional area. Has no units.
        Motor.burnRate : Function
            Propellant burn rate in meter/second as a function of time.
        Motor.interpolate : string
            Method of interpolation used in case thrust curve is given
            by data set in .csv or .eng, or as an array. Options are 'spline'
            'akima' and 'linear'. Default is "linear".
    """

    def __init__(
        self,
        thrustSource,
        dry_mass,
        center_of_dry_mass,
        dry_inertia,
        grainsCenterOfMassPosition,
        grainNumber,
        grainDensity,
        grainOuterRadius,
        grainInitialInnerRadius,
        grainInitialHeight,
        grainSeparation,
        nozzleRadius,
        burn_time=None,
        nozzlePosition=0,
        throatRadius=0.01,
        reshapeThrustCurve=False,
        interpolationMethod="linear",
        coordinateSystemOrientation="nozzleToCombustionChamber",
    ):
        """Initialize Motor class, process thrust curve and geometrical
        parameters and store results.

        Parameters
        ----------
        thrustSource : int, float, callable, string, array
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
            See `Motor.coordinateSystemOrientation`.
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
        grainNumber : int
            Number of solid grains
        grainDensity : int, float
            Solid grain density in kg/m3.
        grainOuterRadius : int, float
            Solid grain outer radius in meters.
        grainInitialInnerRadius : int, float
            Solid grain initial inner radius in meters.
        grainInitialHeight : int, float
            Solid grain initial height in meters.
        grainSeparation : int, float
            Distance between grains, in meters.
        nozzleRadius : int, float
            Motor's nozzle outlet radius in meters.
        nozzlePosition : int, float, optional
            Motor's nozzle outlet position in meters, in the motor's coordinate
            system. See `Motor.coordinateSystemOrientation` for details.
            Default is 0, in which case the origin of the coordinate system
            is placed at the motor's nozzle outlet.
        throatRadius : int, float, optional
            Motor's nozzle throat radius in meters. Used to calculate Kn curve.
            Optional if the Kn curve is not interesting. Its value does not
            impact trajectory simulation.
        reshapeThrustCurve : boolean, tuple, optional
            If False, the original thrust curve supplied is not altered. If a
            tuple is given, whose first parameter is a new burn out time and
            whose second parameter is a new total impulse in Ns, the thrust
            curve is reshaped to match the new specifications. May be useful
            for motors whose thrust curve shape is expected to remain similar
            in case the impulse and burn time varies slightly. Default is
            False.
        interpolationMethod : string, optional
            Method of interpolation to be used in case thrust curve is given
            by data set in .csv or .eng, or as an array. Options are 'spline'
            'akima' and 'linear'. Default is "linear".
        coordinateSystemOrientation : string, optional
            Orientation of the motor's coordinate system. The coordinate system
            is defined by the motor's axis of symmetry. The origin of the
            coordinate system  may be placed anywhere along such axis, such as
            at the nozzle area, and must be kept the same for all other
            positions specified. Options are "nozzleToCombustionChamber" and
            "combustionChamberToNozzle". Default is "nozzleToCombustionChamber".

        Returns
        -------
        None
        """
        super().__init__(
            thrustSource,
            dry_mass,
            center_of_dry_mass,
            dry_inertia,
            nozzleRadius,
            burn_time,
            nozzlePosition,
            reshapeThrustCurve,
            interpolationMethod,
            coordinateSystemOrientation,
        )
        self.liquid = LiquidMotor(
            thrustSource,
            dry_mass,
            center_of_dry_mass,
            dry_inertia,
            nozzleRadius,
            burn_time,
            nozzlePosition,
            reshapeThrustCurve,
            interpolationMethod,
            coordinateSystemOrientation,
        )
        self.solid = SolidMotor(
            thrustSource,
            dry_mass,
            center_of_dry_mass,
            dry_inertia,
            grainsCenterOfMassPosition,
            grainNumber,
            grainDensity,
            grainOuterRadius,
            grainInitialInnerRadius,
            grainInitialHeight,
            grainSeparation,
            nozzleRadius,
            burn_time,
            nozzlePosition,
            throatRadius,
            reshapeThrustCurve,
            interpolationMethod,
            coordinateSystemOrientation,
        )

    @funcify_method("Time (s)", "Exhaust velocity (m/s)")
    def exhaustVelocity(self):
        """Exhaust velocity by assuming it as a constant. The formula used is
        total impulse/propellant initial mass.

        Returns
        -------
        self.exhaustVelocity : Function
            Gas exhaust velocity of the motor.
        """
        return self.totalImpulse / self.propellantInitialMass

    @funcify_method("Time (s)", "Mass (kg)")
    def propellantMass(self):
        """Evaluates the total propellant mass of the motor as the sum
        of each tank mass and the grains mass.

        Returns
        -------
        Function
            Total propellant mass of the motor, in kg.
        """
        return self.solid.propellantMass + self.liquid.propellantMass

    @cached_property
    def propellantInitialMass(self):
        """Returns the initial propellant mass of the motor.

        Returns
        -------
        float
            Initial propellant mass of the motor, in kg.
        """
        return self.solid.propellantInitialMass + self.liquid.propellantInitialMass

    @funcify_method("Time (s)", "mass flow rate (kg/s)", extrapolation="zero")
    def massFlowRate(self):
        """Evaluates the mass flow rate of the motor as the sum of each tank
        mass flow rate and the grains mass flow rate.

        Returns
        -------
        Function
            Mass flow rate of the motor, in kg/s.

        See Also
        --------
        `Motor.totalMassFlowRate` :
            Calculates the total mass flow rate of the motor assuming
            constant exhaust velocity.
        """
        return self.solid.massFlowRate + self.liquid.massFlowRate

    @funcify_method("Time (s)", "center of mass (m)")
    def centerOfPropellantMass(self):
        """Position of the propellant center of mass as a function of time.
        The position is specified as a scalar, relative to the motor's
        coordinate system.

        Returns
        -------
        Function
            Position of the center of mass as a function of time.
        """
        massBalance = (
            self.solid.propellantMass * self.solid.centerOfPropellantMass
            + self.liquid.propellantMass * self.liquid.centerOfPropellantMass
        )
        return massBalance / self.propellantMass

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
            self.solid.propellantMass
            * (self.solid.centerOfPropellantMass - self.centerOfMass) ** 2
        )
        liquidCorrection = (
            self.liquid.propellantMass
            * (self.liquid.centerOfPropellantMass - self.centerOfMass) ** 2
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

    def addTank(self, tank, position):
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
        self.liquid.addTank(tank, position)
        self.solid.massFlowRate = self.totalMassFlowRate - self.liquid.massFlowRate

    def allInfo(self):
        """Prints out all data and graphs available about the Motor.

        Return
        ------
        None
        """
        # Print nozzle details
        print("Nozzle Details")
        print("Nozzle Radius: " + str(self.nozzleRadius) + " m")
        print("Nozzle Throat Radius: " + str(self.solid.throatRadius) + " m")

        # Print grain details
        print("\nGrain Details")
        print("Number of Grains: " + str(self.solid.grainNumber))
        print("Grain Spacing: " + str(self.solid.grainSeparation) + " m")
        print("Grain Density: " + str(self.solid.grainDensity) + " kg/m3")
        print("Grain Outer Radius: " + str(self.solid.grainOuterRadius) + " m")
        print("Grain Inner Radius: " + str(self.solid.grainInitialInnerRadius) + " m")
        print("Grain Height: " + str(self.solid.grainInitialHeight) + " m")
        print("Grain Volume: " + "{:.3f}".format(self.solid.grainInitialVolume) + " m3")
        print("Grain Mass: " + "{:.3f}".format(self.solid.grainInitialMass) + " kg")

        # Print motor details
        print("\nMotor Details")
        print("Total Burning Time: " + str(self.burnDuration) + " s")
        print(
            "Total Propellant Mass: "
            + "{:.3f}".format(self.propellantInitialMass)
            + " kg"
        )
        print(
            "Average Propellant Exhaust Velocity: "
            + "{:.3f}".format(self.exhaustVelocity.average(*self.burn_time))
            + " m/s"
        )
        print("Average Thrust: " + "{:.3f}".format(self.averageThrust) + " N")
        print(
            "Maximum Thrust: "
            + str(self.maxThrust)
            + " N at "
            + str(self.maxThrustTime)
            + " s after ignition."
        )
        print("Total Impulse: " + "{:.3f}".format(self.totalImpulse) + " Ns")

        # Show plots
        print("\nPlots")
        self.thrust.plot(*self.burn_time)
        self.totalMass.plot(*self.burn_time)
        self.massFlowRate.plot(*self.burn_time)
        self.solid.grainInnerRadius.plot(*self.burn_time)
        self.solid.grainHeight.plot(*self.burn_time)
        self.solid.burnRate.plot(self.burn_time[0], self.solid.grainBurnOut)
        self.solid.burnArea.plot(*self.burn_time)
        self.solid.Kn.plot(*self.burn_time)
        self.centerOfMass.plot(*self.burn_time)
        self.I_11.plot(*self.burn_time)
        self.I_22.plot(*self.burn_time)
        self.I_33.plot(*self.burn_time)
        self.I_12.plot(*self.burn_time)
        self.I_13.plot(*self.burn_time)
        self.I_23.plot(*self.burn_time)

        return None
