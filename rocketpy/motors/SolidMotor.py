# -*- coding: utf-8 -*-

__author__ = "Giovani Hidalgo Ceotto, Oscar Mauricio Prada Ramirez, João Lemes Gribel Soares, Lucas Kierulff Balabram, Lucas Azevedo Pezente"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

import numpy as np
from scipy import integrate

try:
    from functools import cached_property
except ImportError:
    from rocketpy.tools import cached_property

from rocketpy.Function import Function, funcify_method
from rocketpy.motors import Motor


class SolidMotor(Motor):
    """Class to specify characteristics and useful operations for solid motors.

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
        Motor.grainNumber : int
            Number of solid grains.
        Motor.grainsCenterOfMassPosition : float
            Position of the center of mass of the grains in meters, specified in
            the motor's coordinate system.
            See `Motor.coordinateSystemOrientation` for more information.
        Motor.grainSeparation : float
            Distance between two grains in meters.
        Motor.grainDensity : float
            Density of each grain in kg/meters cubed.
        Motor.grainOuterRadius : float
            Outer radius of each grain in meters.
        Motor.grainInitialInnerRadius : float
            Initial inner radius of each grain in meters.
        Motor.grainInitialHeight : float
            Initial height of each grain in meters.
        Motor.grainInitialVolume : float
            Initial volume of each grain in meters cubed.
        Motor.grainInnerRadius : Function
            Inner radius of each grain in meters as a function of time.
        Motor.grainHeight : Function
            Height of each grain in meters as a function of time.

        Mass and moment of inertia attributes:
        Motor.centerOfMass : Function
            Position of the center of mass in meters as a function of time.
            Constant for solid motors, as the grains are assumed to be fixed.
            See `Motor.coordinateSystemOrientation` for more information
            regarding the motor's coordinate system
        Motor.grainInitialMass : float
            Initial mass of each grain in kg.
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
        Motor.inertiaI : Function
            Propellant moment of inertia in kg*meter^2 with respect to axis
            perpendicular to axis of cylindrical symmetry of each grain,
            given as a function of time.
        Motor.inertiaIDot : Function
            Time derivative of inertiaI given in kg*meter^2/s as a function
            of time.
        Motor.inertiaZ : Function
            Propellant moment of inertia in kg*meter^2 with respect to axis of
            cylindrical symmetry of each grain, given as a function of time.
        Motor.inertiaDot : Function
            Time derivative of inertiaZ given in kg*meter^2/s as a function
            of time.

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
        Motor.burnOutTime : float
            Total motor burn out time, in seconds. Must include delay time
            when the motor takes time to ignite. Also seen as time to end thrust
            curve.
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
        burnOut,
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
        burnOut : int, float
            Motor burn out time in seconds.
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
        grainsCenterOfMassPosition : float
            Position of the center of mass of the grains in meters, specified in
            the motor's coordinate system.
            See `Motor.coordinateSystemOrientation` for more information.
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
            burnOut,
            dry_mass,
            center_of_dry_mass,
            dry_inertia,
            nozzleRadius,
            nozzlePosition,
            reshapeThrustCurve,
            interpolationMethod,
            coordinateSystemOrientation,
        )
        # Nozzle parameters
        self.throatRadius = throatRadius
        self.throatArea = np.pi * throatRadius**2

        # Grain parameters
        self.grainsCenterOfMassPosition = grainsCenterOfMassPosition
        self.grainNumber = grainNumber
        self.grainSeparation = grainSeparation
        self.grainDensity = grainDensity
        self.grainOuterRadius = grainOuterRadius
        self.grainInitialInnerRadius = grainInitialInnerRadius
        self.grainInitialHeight = grainInitialHeight

        # Grains initial geometrical parameters
        self.grainInitialVolume = (
            self.grainInitialHeight
            * np.pi
            * (self.grainOuterRadius**2 - self.grainInitialInnerRadius**2)
        )
        self.grainInitialMass = self.grainDensity * self.grainInitialVolume

        self.evaluateGeometry()

    @funcify_method("Time (s)", "mass (kg)")
    def propellantMass(self):
        """Evaluates the total propellant mass as a function of time.

        Parameters
        ----------
        t : float
            Time in seconds.

        Returns
        -------
        Function
            Mass of the motor, in kg.
        """
        return self.grainVolume * self.grainDensity * self.grainNumber

    @funcify_method("Time (s)", "grain volume (m³)")
    def grainVolume(self):
        """Evaluates the total propellant volume as a function of time. The
        propellant is assumed to be a cylindrical Bates grain under uniform
        burn.

        Parameters
        ----------
        t : float
            Time in seconds.

        Returns
        -------
        Function
            Propellant volume as a function of time.
        """
        cross_section_area = np.pi * (
            self.grainOuterRadius**2 - self.grainInnerRadius**2
        )
        return cross_section_area * self.grainHeight

    @funcify_method("Time (s)", "Exhaust velocity (m/s)")
    def exhaustVelocity(self):
        """Exhaust velocity by assuming it as a constant. The formula used is
        total impulse/propellant initial mass.

        Parameters
        ----------
        t : float
            Time in seconds.

        Returns
        -------
        self.exhaustVelocity : Function
            Gas exhaust velocity of the motor.
        """
        return self.totalImpulse / self.propellantInitialMass

    @property
    def propellantInitialMass(self):
        """Returns the initial propellant mass.

        Parameters
        ----------
        None

        Returns
        -------
        float
            Initial propellant mass in kg.
        """
        return self.grainNumber * self.grainInitialMass

    @property
    def massFlowRate(self):
        """Time derivative of propellant mass. Assumes constant exhaust
        velocity. The formula used is the opposite of thrust divided by
        exhaust velocity.

        Parameters
        ----------
        t : float
            Time in seconds.

        Returns
        -------
        self.massFlowRate : Function
            Time derivative of total propellant mass as a function of time.

        See Also
        --------
        `Motor.totalMassFlowRate` :
            Calculates the total mass flow rate of the motor assuming
            constant exhaust velocity.
        """
        try:
            return self._massFlowRate
        except AttributeError:
            self._massFlowRate = self.totalMassFlowRate
            return self._massFlowRate

    @massFlowRate.setter
    def massFlowRate(self, value):
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
        self.evaluateGeometry()

    @funcify_method("Time (s)", "center of mass (m)")
    def centerOfPropellantMass(self):
        """Position of the propellant center of mass as a function of time.
        The position is specified as a scalar, relative to the motor's
        coordinate system.

        Parameters
        ----------
        t : float
            Time in seconds.

        Returnsg
        -------
        Function
            Position of the propellant center of mass as a function of time.
        """
        return self.grainsCenterOfMassPosition

    def evaluateGeometry(self):
        """Calculates grain inner radius and grain height as a function of time
        by assuming that every propellant mass burnt is exhausted. In order to
        do that, a system of differential equations is solved using
        scipy.integrate.odeint. Furthermore, the function calculates burn area,
        burn rate and Kn as a function of time using the previous results. All
        functions are stored as objects of the class Function in
        self.grainInnerRadius, self.grainHeight, self.burnArea, self.burnRate
        and self.Kn.

        Parameters
        ----------
        None

        Returns
        -------
        geometry : list of Functions
            First element is the Function representing the inner radius of a
            grain as a function of time. Second argument is the Function
            representing the height of a grain as a function of time.
        """
        # Define initial conditions for integration
        y0 = [self.grainInitialInnerRadius, self.grainInitialHeight]

        # Define time mesh
        t = self.thrust.source[:, 0]
        t_span = (t[0], t[-1])

        density = self.grainDensity
        rO = self.grainOuterRadius

        # Define system of differential equations
        def geometryDot(t, y):
            grainMassDot = self.massFlowRate(t) / self.grainNumber
            rI, h = y
            rIDot = (
                -0.5 * grainMassDot / (density * np.pi * (rO**2 - rI**2 + rI * h))
            )
            hDot = 1.0 * grainMassDot / (density * np.pi * (rO**2 - rI**2 + rI * h))
            return [rIDot, hDot]

        def terminateBurn(t, y):
            end_function = (self.grainOuterRadius - y[0]) * y[1]
            return end_function

        terminateBurn.terminal = True

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
        self.grainInnerRadius = Function(
            np.concatenate(([sol.t], [sol.y[0]])).transpose().tolist(),
            "Time (s)",
            "Grain Inner Radius (m)",
            self.interpolate,
            "constant",
        )
        self.grainHeight = Function(
            np.concatenate(([sol.t], [sol.y[1]])).transpose().tolist(),
            "Time (s)",
            "Grain Height (m)",
            self.interpolate,
            "constant",
        )

        return [self.grainInnerRadius, self.grainHeight]

    @funcify_method("Time (s)", "burn area (m²)")
    def burnArea(self):
        """Calculates the BurnArea of the grain for each time. Assuming that
        the grains are cylindrical BATES grains.

        Parameters
        ----------
        t : float
            Time in seconds.

        Returns
        -------
        burnArea : Function
            Function representing the burn area progression with the time.
        """
        burnArea = (
            2
            * np.pi
            * (
                self.grainOuterRadius**2
                - self.grainInnerRadius**2
                + self.grainInnerRadius * self.grainHeight
            )
            * self.grainNumber
        )
        return burnArea

    @funcify_method("Time (s)", "burn rate (m/s)")
    def burnRate(self):
        """Calculates the BurnRate with respect to time. This evaluation
        assumes that it was already calculated the massDot, burnArea time
        series.

        Parameters
        ----------
        t : float
            Time in seconds.

        Returns
        -------
        burnRate : Function
            Rate of progression of the inner radius during the combustion.
        """
        return -1 * self.massFlowRate / (self.burnArea * self.grainDensity)

    @cached_property
    def Kn(self):
        """Calculates the motor Kn as a function of time. Defined as burnArea
        divided by the nozzle throat cross sectional area.

        Returns
        -------
        Kn : Function of Inner Radius and Kn
            Kn as a function of time.
        """
        KnSource = (
            np.concatenate(
                (
                    [self.grainInnerRadius.source[:, 1]],
                    [self.burnArea.source[:, 1] / self.throatArea],
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
        """Inertia tensor 11 component of the propellnat, the inertia is
        relative to the e_1 axis, centered at the instantaneous propellant
        center of mass.

        Parameters
        ----------
        t : float
            Time in seconds.

        Returns
        -------
        float
            Propellant inertia tensor 11 component at time t.

        Notes
        -----
        The e_1 direction is assumed to be the direction perpendicular to the
        motor body axis.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor
        """
        grainMass = self.propellantMass / self.grainNumber
        grainNumber = self.grainNumber
        grainInertia11 = grainMass * (
            (1 / 4) * (self.grainOuterRadius**2 + self.grainInnerRadius**2)
            + (1 / 12) * self.grainHeight**2
        )

        # Calculate each grain's distance d to propellant center of mass
        initialValue = (grainNumber - 1) / 2
        d = np.linspace(-initialValue, initialValue, grainNumber)
        d = d * (self.grainInitialHeight + self.grainSeparation)

        # Calculate inertia for all grains
        I_11 = grainNumber * grainInertia11 + grainMass * np.sum(d**2)

        return I_11

    @funcify_method("Time (s)", "Inertia I_22 (kg m²)")
    def propellant_I_22(self):
        """Inertia tensor 22 component of the propellnat, the inertia is
        relative to the e_2 axis, centered at the instantaneous propellant
        center of mass.

        Parameters
        ----------
        t : float
            Time in seconds.

        Returns
        -------
        float
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
        """Inertia tensor 33 component of the propellnat, the inertia is
        relative to the e_3 axis, centered at the instantaneous propellant
        center of mass.

        Parameters
        ----------
        t : float
            Time in seconds.

        Returns
        -------
        float
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
            * self.propellantMass
            * (self.grainOuterRadius**2 + self.grainInnerRadius**2)
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

    def allInfo(self):
        """Prints out all data and graphs available about the Motor.

        Parameters
        ----------
        None

        Return
        ------
        None
        """
        # Print nozzle details
        print("Nozzle Details")
        print("Nozzle Radius: " + str(self.nozzleRadius) + " m")
        print("Nozzle Throat Radius: " + str(self.throatRadius) + " m")

        # Print grain details
        print("\nGrain Details")
        print("Number of Grains: " + str(self.grainNumber))
        print("Grain Spacing: " + str(self.grainSeparation) + " m")
        print("Grain Density: " + str(self.grainDensity) + " kg/m3")
        print("Grain Outer Radius: " + str(self.grainOuterRadius) + " m")
        print("Grain Inner Radius: " + str(self.grainInitialInnerRadius) + " m")
        print("Grain Height: " + str(self.grainInitialHeight) + " m")
        print("Grain Volume: " + "{:.3f}".format(self.grainInitialVolume) + " m3")
        print("Grain Mass: " + "{:.3f}".format(self.grainInitialMass) + " kg")

        # Print motor details
        print("\nMotor Details")
        print("Total Burning Time: " + str(self.burnOutTime) + " s")
        print(
            "Total Propellant Mass: "
            + "{:.3f}".format(self.propellantInitialMass)
            + " kg"
        )
        print(
            "Average Propellant Exhaust Velocity: "
            + "{:.3f}".format(self.exhaustVelocity.average(0, self.burnOutTime))
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
        self.thrust()
        self.totalMass()
        self.massFlowRate()
        self.exhaustVelocity()
        self.grainInnerRadius()
        self.grainHeight()
        self.burnRate.plot(0, self.grainBurnOut)
        self.burnArea()
        self.Kn()
        self.centerOfMass.plot(0, self.burnOutTime)
        self.I_11.plot(0, self.burnOutTime)
        self.I_22.plot(0, self.burnOutTime)
        self.I_33.plot(0, self.burnOutTime)
        self.I_12.plot(0, self.burnOutTime)
        self.I_13.plot(0, self.burnOutTime)
        self.I_23.plot(0, self.burnOutTime)

        return None
