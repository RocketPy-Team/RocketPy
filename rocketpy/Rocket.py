# -*- coding: utf-8 -*-

__author__ = "Giovani Hidalgo Ceotto, Franz Masatoshi Yuri"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

import warnings
from collections import namedtuple
from inspect import getsourcelines

import numpy as np

from .Function import Function
from .Parachute import Parachute


class Rocket:

    """Keeps all rocket and parachute information.

    Attributes
    ----------
        Geometrical attributes:
        Rocket.radius : float
            Rocket's largest radius in meters.
        Rocket.area : float
            Rocket's circular cross section largest frontal area in squared
            meters.
        Rocket.positionNozzle : float
            Rocket's nozzle position, in meters. Can be relative to any
            coordinate system that is aligned with the rocket's axis.
        Rocket.positionCenterOfDryMass : float
            Rocket's center of dry mass position, in meters. Can be relative
            to any coordinate system that is aligned with the rocket's axis.
        Rocket.positionCenterOfDryMassToNozzle : float
            Position of the rocket's center of dry mass relative to the
            rocket's nozzle, in meters, considering positive direction from
            nozzle to nose cone. Always positive.
        Rocket.positionMotorReferencePositionToCenterOfDryMass : float
            Position of the rocket's motor's reference point relative to
            the rocket's center of mass, in meters.

        Mass and Inertia attributes:
        Rocket.mass : float
            Rocket's mass without propellant in kg.
        Rocket.inertiaI : float
            Rocket's moment of inertia, without propellant, with respect to
            to an axis perpendicular to the rocket's axis of cylindrical
            symmetry, in kg*m^2.
        Rocket.inertiaZ : float
            Rocket's moment of inertia, without propellant, with respect to
            the rocket's axis of cylindrical symmetry, in kg*m^2.
        Rocket.centerOfMass : Function
            Distance of the rocket's center of mass, including propellant,
            to rocket's center of mass without propellant, in meters.
            Expressed as a function of time.
        Rocket.reducedMass : Function
            Function of time expressing the reduced mass of the rocket,
            defined as the product of the propellant mass and the mass
            of the rocket without propellant, divided by the sum of the
            propellant mass and the rocket mass.
        Rocket.totalMass : Function
            Function of time expressing the total mass of the rocket,
            defined as the sum of the propellant mass and the rocket
            mass without propellant.
        Rocket.thrustToWeight : Function
            Function of time expressing the motor thrust force divided by rocket
            weight. The gravitational acceleration is assumed as 9.80665 m/s^2.

        Eccentricity attributes:
        Rocket.cpEccentricityX : float
            Center of pressure position relative to center of mass in the x
            axis, perpendicular to axis of cylindrical symmetry, in meters.
        Rocket.cpEccentricityY : float
            Center of pressure position relative to center of mass in the y
            axis, perpendicular to axis of cylindrical symmetry, in meters.
        Rocket.thrustEccentricityY : float
            Thrust vector position relative to center of mass in the y
            axis, perpendicular to axis of cylindrical symmetry, in meters.
        Rocket.thrustEccentricityX : float
            Thrust vector position relative to center of mass in the x
            axis, perpendicular to axis of cylindrical symmetry, in meters.

        Aerodynamic attributes
        Rocket.aerodynamicSurfaces : list
            List of aerodynamic surfaces of the rocket.
        Rocket.staticMargin : float
            Float value corresponding to rocket static margin when
            loaded with propellant in units of rocket diameter or
            calibers.
        Rocket.powerOffDrag : Function
            Rocket's drag coefficient as a function of Mach number when the
            motor is off.
        Rocket.powerOnDrag : Function
            Rocket's drag coefficient as a function of Mach number when the
            motor is on.

        Motor attributes:
        Rocket.motor : Motor
            Rocket's motor. See Motor class for more details.
    """

    def __init__(
        self,
        motor,
        mass,
        inertiaI,
        inertiaZ,
        radius,
        positionNozzle,
        positionCenterOfDryMass,
        powerOffDrag,
        powerOnDrag,
    ):
        """Initializes Rocket class, process inertial, geometrical and
        aerodynamic parameters.

        Parameters
        ----------
        motor : Motor
            Motor used in the rocket. See Motor class for more information.
        mass : int, float
            Unloaded rocket total mass (without propellant) in kg.
        inertiaI : int, float
            Unloaded rocket lateral (perpendicular to axis of symmetry)
            moment of inertia (without propellant) in kg m^2.
        inertiaZ : int, float
            Unloaded rocket axial moment of inertia (without propellant)
            in kg m^2.
        radius : int, float
            Rocket biggest outer radius in meters.
        positionNozzle : int, float
            Nozzle position relative to considered coordinate system. The chosen
            coordinate system must be aligned with the rocket's axis.
        positionCenterOfDryMass : int, float
            Center of dry mass position relative to considered coordinate system.
            The chosen coordinate system must be aligned with the rocket's axis.
        powerOffDrag : int, float, callable, string, array
            Rocket's drag coefficient when the motor is off. Can be given as an
            entry to the Function class. See help(Function) for more
            information. If int or float is given, it is assumed constant. If
            callable, string or array is given, it must be a function of Mach
            number only.
        powerOnDrag : int, float, callable, string, array
            Rocket's drag coefficient when the motor is on. Can be given as an
            entry to the Function class. See help(Function) for more
            information. If int or float is given, it is assumed constant. If
            callable, string or array is given, it must be a function of Mach
            number only.

        Returns
        -------
        None
        """
        # Define motor to be used
        self.motor = motor

        # Define center of mass and points of interest relative to the inputted reference axis
        self.positionNozzle = positionNozzle
        self.positionCenterOfDryMass = positionCenterOfDryMass

        # Define positions relative to nozzle
        self.positionCenterOfDryMassToNozzle = abs(
            positionCenterOfDryMass - positionNozzle
        )

        # Define positions relative to the rocket's center of dry mass
        self.positionMotorReferencePositionToCenterOfDryMass = (
            self.motor.distanceMotorReferenceToNozzle
            - self.positionCenterOfDryMassToNozzle
        )

        # Define rocket inertia attributes in SI units
        self.mass = mass
        self.inertiaI = inertiaI
        self.inertiaZ = inertiaZ

        self.centerOfMass = (
            (self.positionMotorReferencePositionToCenterOfDryMass - self.motor.zCM)
            * motor.mass
            / (mass + motor.mass)
        )

        # Define rocket geometrical parameters in SI units
        self.radius = radius
        self.area = np.pi * self.radius**2

        # Eccentricity data initialization
        self.cpEccentricityX = 0
        self.cpEccentricityY = 0
        self.thrustEccentricityY = 0
        self.thrustEccentricityX = 0

        # Parachute data initialization
        self.parachutes = []

        # Rail button data initialization
        self.railButtons = None

        # Aerodynamic data initialization
        self.aerodynamicSurfaces = []
        self.cpPosition = 0
        self.staticMargin = Function(
            lambda x: 0, inputs="Time (s)", outputs="Static Margin (c)"
        )

        # Define aerodynamic drag coefficients
        self.powerOffDrag = Function(
            powerOffDrag,
            "Mach Number",
            "Drag Coefficient with Power Off",
            "spline",
            "constant",
        )
        self.powerOnDrag = Function(
            powerOnDrag,
            "Mach Number",
            "Drag Coefficient with Power On",
            "spline",
            "constant",
        )

        # Important dynamic inertial quantities
        self.reducedMass = None
        self.totalMass = None

        # Calculate dynamic inertial quantities
        self.evaluateReducedMass()
        self.evaluateTotalMass()
        self.thrustToWeight = self.motor.thrust / (9.80665 * self.totalMass)
        self.thrustToWeight.setInputs("Time (s)")
        self.thrustToWeight.setOutputs("Thrust/Weight")

        # Evaluate static margin (even though no aerodynamic surfaces are present yet)
        self.evaluateStaticMargin()

        return None

    def evaluateReducedMass(self):
        """Calculates and returns the rocket's total reduced mass. The
        reduced mass is defined as the product of the propellant mass
        and the mass of the rocket without propellant, divided by the
        sum of the propellant mass and the rocket mass. The function
        returns an object of the Function class and is defined as a
        function of time.

        Parameters
        ----------
        None

        Returns
        -------
        self.reducedMass : Function
            Function of time expressing the reduced mass of the rocket,
            defined as the product of the propellant mass and the mass
            of the rocket without propellant, divided by the sum of the
            propellant mass and the rocket mass.
        """
        # Make sure there is a motor associated with the rocket
        if self.motor is None:
            print("Please associate this rocket with a motor!")
            return False

        # Retrieve propellant mass as a function of time
        motorMass = self.motor.mass

        # Retrieve constant rocket mass without propellant
        mass = self.mass

        # Calculate reduced mass
        self.reducedMass = motorMass * mass / (motorMass + mass)
        self.reducedMass.setOutputs("Reduced Mass (kg)")

        # Return reduced mass
        return self.reducedMass

    def evaluateTotalMass(self):
        """Calculates and returns the rocket's total mass. The total
        mass is defined as the sum of the propellant mass and the
        rocket mass without propellant. The function returns an object
        of the Function class and is defined as a function of time.

        Parameters
        ----------
        None

        Returns
        -------
        self.totalMass : Function
            Function of time expressing the total mass of the rocket,
            defined as the sum of the propellant mass and the rocket
            mass without propellant.
        """
        # Make sure there is a motor associated with the rocket
        if self.motor is None:
            print("Please associate this rocket with a motor!")
            return False

        # Calculate total mass by summing up propellant and dry mass
        self.totalMass = self.mass + self.motor.mass
        self.totalMass.setOutputs("Total Mass (Rocket + Propellant) (kg)")

        # Return total mass
        return self.totalMass

    def evaluateStaticMargin(self):
        """Calculates and returns the rocket's static margin when
        loaded with propellant. The static margin is saved and returned
        in units of rocket diameter or calibers. This function also calculates
        the rocket center of pressure and total lift coefficients.

        Parameters
        ----------
        None

        Returns
        -------
        self.staticMargin : float
            Float value corresponding to rocket static margin when
            loaded with propellant in units of rocket diameter or
            calibers.
        """
        # Initialize total lift coefficient derivative and center of pressure
        self.totalLiftCoeffDer = 0
        self.cpPosition = 0

        # Calculate total lift coefficient derivative and center of pressure
        if len(self.aerodynamicSurfaces) > 0:
            for aerodynamicSurface in self.aerodynamicSurfaces:
                self.totalLiftCoeffDer += Function(
                    lambda alpha: aerodynamicSurface["cl"](alpha, 0)
                ).differentiate(x=1e-2, dx=1e-3)
                self.cpPosition += (
                    Function(
                        lambda alpha: aerodynamicSurface["cl"](alpha, 0)
                    ).differentiate(x=1e-2, dx=1e-3)
                    * aerodynamicSurface["cp"][2]
                )
            self.cpPosition /= self.totalLiftCoeffDer

        # Calculate static margin
        self.staticMargin = (self.centerOfMass - self.cpPosition) / (2 * self.radius)
        self.staticMargin.setInputs("Time (s)")
        self.staticMargin.setOutputs("Static Margin (c)")
        self.staticMargin.setDiscrete(
            lower=0, upper=self.motor.burnOutTime, samples=200
        )

        # Return self
        return self

    def addTail(self, topRadius, bottomRadius, length, positionTail):
        """Create a new tail or rocket diameter change, storing its
        parameters as part of the aerodynamicSurfaces list. Its
        parameters are the axial position along the rocket and its
        derivative of the coefficient of lift in respect to angle of
        attack.
        Parameters
        ----------
        topRadius : int, float
            Tail top radius in meters, considering positive direction
            from center of mass to nose cone.
        bottomRadius : int, float
            Tail bottom radius in meters, considering positive direction
            from center of mass to nose cone.
        length : int, float
            Tail length or height in meters. Must be a positive value.
        positionTail : int, float
            Tail position relative to considered coordinate system.
            Consider a point belonging to the tail's top radius to
            calculate position.
        Returns
        -------
        cl : Function
            Function of the angle of attack (Alpha) and the mach number
            (Mach) expressing the tail's lift coefficient. The inputs
            are the angle of attack (in radians) and the mach number.
            The output is the tail's lift coefficient. In the current
            implementation, the tail's lift coefficient does not vary
            with mach.
        self : Rocket
            Object of the Rocket class.
        """
        # Calculate ratio between top and bottom radius
        r = topRadius / bottomRadius

        # Retrieve reference radius
        rref = self.radius

        # Calculate tail position relative to nozzle
        # Must check if the tail is set before or after the Nozzle
        tailPosition_Nozzle = self.evaluatePositionSurface_Nozzle("Tail", positionTail)

        # Calculate tail position relative to cm
        tailPosition_CM = (
            tailPosition_Nozzle - self.positionCenterOfDryMassToNozzle
        )  # tail initial position

        # Calculate cp position relative to cm
        if tailPosition_CM < 0:
            cpz = tailPosition_CM - (length / 3) * (1 + (1 - r) / (1 - r**2))
        else:
            cpz = tailPosition_CM + (length / 3) * (1 + (1 - r) / (1 - r**2))

        # Calculate clalpha
        clalpha = -2 * (1 - r ** (-2)) * (topRadius / rref) ** 2
        cl = Function(
            lambda alpha, mach: clalpha * alpha,
            ["Alpha (rad)", "Mach"],
            "Cl",
        )

        # Store values as new aerodynamic surface
        tail = {"cp": (0, 0, cpz), "cl": cl, "name": "Tail"}
        self.aerodynamicSurfaces.append(tail)

        # Refresh static margin calculation
        self.evaluateStaticMargin()

        # Return self
        return self.aerodynamicSurfaces[-1]

    def addNose(self, length, kind, positionNose):
        """Creates a nose cone, storing its parameters as part of the
        aerodynamicSurfaces list. Its parameters are the axial position
        along the rocket and its derivative of the coefficient of lift
        in respect to angle of attack.


        Parameters
        ----------
        length : int, float
            Nose cone length or height in meters. Must be a positive
            value.
        kind : string
            Nose cone type. Von Karman, conical, ogive, and lvhaack are
            supported.
        positionNose : int, float
            Nose cone position relative to considered coordinate system.
            Consider a point belonging to the nose cones's tip to calculate
            position.

        Returns
        -------
        cl : Function
            Function of the angle of attack (Alpha) and the mach number
            (Mach) expressing the nose cone's lift coefficient. The inputs
            are the angle of attack (in radians) and the mach number.
            The output is the nose cone's lift coefficient. In the current
            implementation, the nose cone's lift coefficient does not vary
            with mach
        self : Rocket
            Object of the Rocket class.
        """
        # Analyze type
        if kind == "conical":
            k = 1 - 1 / 3
        elif kind == "ogive":
            k = 1 - 0.534
        elif kind == "lvhaack":
            k = 1 - 0.437
        else:
            k = 0.5

        # Calculate nosecone tip position relative to nozzle
        # Must check if the nosecone is set before or after the Nozzle
        nosePosition_Nozzle = self.evaluatePositionSurface_Nozzle(
            "Nosecone", positionNose
        )

        # Calculate nosecone base position relative to cm
        nosePosition_CM = (
            nosePosition_Nozzle - length
        ) - self.positionCenterOfDryMassToNozzle

        # Calculate cp position relative to cm
        if nosePosition_CM > 0:
            cpz = nosePosition_CM + k * length
        else:
            cpz = nosePosition_CM - k * length

        # Calculate clalpha
        clalpha = 2
        cl = Function(
            lambda alpha, mach: clalpha * alpha,
            ["Alpha (rad)", "Mach"],
            "Cl",
        )

        # Store values
        nose = {"cp": (0, 0, cpz), "cl": cl, "name": "Nose Cone"}
        self.aerodynamicSurfaces.append(nose)

        # Refresh static margin calculation
        self.evaluateStaticMargin()

        # Return self
        return self.aerodynamicSurfaces[-1]

    def addFins(
        self,
        n,
        span,
        rootChord,
        tipChord,
        positionFins,
        radius=0,
        cantAngle=0,
        airfoil=None,
    ):
        """Create a fin set, storing its parameters as part of the
        aerodynamicSurfaces list. Its parameters are the axial position
        along the rocket and its derivative of the coefficient of lift
        in respect to angle of attack.
        Parameters
        ----------
        n : int
            Number of fins, from 2 to infinity.
        span : int, float
            Fin span in meters.
        rootChord : int, float
            Fin root chord in meters.
        tipChord : int, float
            Fin tip chord in meters.
        positionFins : int, float
            Fins position relative to considered coordinate system.
            Consider the center point belonging to the top of the
            fins to calculate position.
        radius : int, float, optional
            Reference radius to calculate lift coefficient. If 0, which
            is default, use rocket radius. Otherwise, enter the radius
            of the rocket in the section of the fins, as this impacts
            its lift coefficient.
        cantAngle : int, float, optional
            Fins cant angle with respect to the rocket centerline. Must
            be given in degrees.
        airfoil : tuple, optional
            Default is null, in which case fins will be treated as flat plates.
            Otherwise, if tuple, fins will be considered as airfoils. The
            tuple's first item specifies the airfoil's lift coefficient
            by angle of attack and must be either a .csv, .txt, ndarray
            or callable. The .csv and .txt files must contain no headers
            and the first column must specify the angle of attack, while
            the second column must specify the lift coefficient. The
            ndarray should be as [(x0, y0), (x1, y1), (x2, y2), ...]
            where x0 is the angle of attack and y0 is the lift coefficient.
            If callable, it should take an angle of attack as input and
            return the lift coefficient at that angle of attack.
            The tuple's second item is the unit of the angle of attack,
            accepting either "radians" or "degrees".
        Returns
        -------
        cl : Function
            Function of the angle of attack (Alpha) and the mach number
            (Mach) expressing the fin's lift coefficient. The inputs
            are the angle of attack (in radians) and the mach number.
            The output is the fin's lift coefficient.
        self : Rocket
            Object of the Rocket class.
        """

        # Retrieve parameters for calculations
        Cr = rootChord
        Ct = tipChord
        Yr = rootChord + tipChord
        s = span
        Af = Yr * s / 2  # fin area
        Yma = (
            (s / 3) * (Cr + 2 * Ct) / Yr
        )  # span wise position of fin's mean aerodynamic chord
        gamac = np.arctan((Cr - Ct) / (2 * s))
        Lf = np.sqrt((Cr / 2 - Ct / 2) ** 2 + s**2)
        radius = self.radius if radius == 0 else radius
        d = 2 * radius
        Aref = np.pi * radius**2
        AR = 2 * s**2 / Af  # Barrowman's convention for fin's aspect ratio
        cantAngleRad = np.radians(cantAngle)
        trapezoidalConstant = (
            (Cr + 3 * Ct) * s**3
            + 4 * (Cr + 2 * Ct) * radius * s**2
            + 6 * (Cr + Ct) * s * radius**2
        ) / 12

        # Fin–body interference correction parameters
        τ = (s + radius) / radius
        λ = Ct / Cr
        liftInterferenceFactor = 1 + 1 / τ
        rollForcingInterferenceFactor = (1 / np.pi**2) * (
            (np.pi**2 / 4) * ((τ + 1) ** 2 / τ**2)
            + ((np.pi * (τ**2 + 1) ** 2) / (τ**2 * (τ - 1) ** 2))
            * np.arcsin((τ**2 - 1) / (τ**2 + 1))
            - (2 * np.pi * (τ + 1)) / (τ * (τ - 1))
            + ((τ**2 + 1) ** 2)
            / (τ**2 * (τ - 1) ** 2)
            * (np.arcsin((τ**2 - 1) / (τ**2 + 1))) ** 2
            - (4 * (τ + 1)) / (τ * (τ - 1)) * np.arcsin((τ**2 - 1) / (τ**2 + 1))
            + (8 / (τ - 1) ** 2) * np.log((τ**2 + 1) / (2 * τ))
        )
        rollDampingInterferenceFactor = 1 + (
            ((τ - λ) / (τ)) - ((1 - λ) / (τ - 1)) * np.log(τ)
        ) / (((τ + 1) * (τ - λ)) / (2) - ((1 - λ) * (τ**3 - 1)) / (3 * (τ - 1)))

        # Save geometric parameters for later Fin Flutter Analysis and Roll Moment Calculation
        self.rootChord = Cr
        self.tipChord = Ct
        self.span = s
        # self.distanceRocketFins = distanceToCM

        # Auxiliary functions

        # Defines beta parameter
        def beta(mach):
            """Defines a parameter that is commonly used in aerodynamic
            equations. It is commonly used in the Prandtl factor which
            corrects subsonic force coefficients for compressible flow.

            Parameters
            ----------
            mach : int, float
                Number of mach.

            Returns
            -------
            beta : int, float
                Value that characterizes flow speed based on the mach number.
            """

            if mach < 0.8:
                return np.sqrt(1 - mach**2)
            elif mach < 1.1:
                return np.sqrt(1 - 0.8**2)
            else:
                return np.sqrt(mach**2 - 1)

        # Defines number of fins correction
        def finNumCorrection(n):
            """Calculates a corrector factor for the lift coefficient of multiple fins.
            The specifics  values are documented at:
            Niskanen, S. (2013). “OpenRocket technical documentation”. In: Development
            of an Open Source model rocket simulation software.

            Parameters
            ----------
            n : int
                Number of fins.

            Returns
            -------
            Corrector factor : int
                Factor that accounts for the number of fins.
            """
            correctorFactor = [2.37, 2.74, 2.99, 3.24]
            if n >= 5 and n <= 8:
                return correctorFactor[n - 5]
            else:
                return n / 2

        # Calculate fins position relative to Nozzle
        # Must check if the fins are set before or after the Nozzle
        finsPosition_Nozzle = self.evaluatePositionSurface_Nozzle("Fins", positionFins)

        # Calculate fins position relative to cm
        finsPosition_CM = finsPosition_Nozzle - self.positionCenterOfDryMassToNozzle

        # Calculate cp position relative to cm
        if finsPosition_CM < 0:
            cpz = finsPosition_CM - (
                ((Cr - Ct) / 3) * ((Cr + 2 * Ct) / (Cr + Ct))
                + (1 / 6) * (Cr + Ct - Cr * Ct / (Cr + Ct))
            )
        else:
            cpz = finsPosition_CM + (
                ((Cr - Ct) / 3) * ((Cr + 2 * Ct) / (Cr + Ct))
                + (1 / 6) * (Cr + Ct - Cr * Ct / (Cr + Ct))
            )

        if not airfoil:
            # Defines clalpha2D as 2*pi for planar fins
            clalpha2D = Function(lambda mach: 2 * np.pi / beta(mach))
        else:
            # Defines clalpha2D as the derivative of the
            # lift coefficient curve for a specific airfoil
            airfoilCl = Function(
                airfoil[0],
                interpolation="linear",
            )

            # Differentiating at x = 0 to get cl_alpha
            clalpha2D_Mach0 = airfoilCl.differentiate(x=1e-3, dx=1e-3)

            # Convert to radians if needed
            if airfoil[1] == "degrees":
                clalpha2D_Mach0 *= 180 / np.pi

            # Correcting for compressible flow
            clalpha2D = Function(lambda mach: clalpha2D_Mach0 / beta(mach))
        # Diederich's Planform Correlation Parameter
        FD = 2 * np.pi * AR / (clalpha2D * np.cos(gamac))

        # Lift coefficient derivative for a single fin
        clalphaSingleFin = Function(
            lambda mach: (clalpha2D(mach) * FD(mach) * (Af / Aref) * np.cos(gamac))
            / (2 + FD(mach) * np.sqrt(1 + (2 / FD(mach)) ** 2))
        )

        # Lift coefficient derivative for a number of n fins corrected for Fin-Body interference
        clalphaMultipleFins = (
            liftInterferenceFactor * finNumCorrection(n) * clalphaSingleFin
        )  # Function of mach number

        # Calculates clalpha * alpha
        cl = Function(
            lambda alpha, mach: alpha * clalphaMultipleFins(mach),
            ["Alpha (rad)", "Mach"],
            "Cl",
        )

        # Parameters for Roll Moment.
        # Documented at: https://github.com/Projeto-Jupiter/RocketPy/blob/develop/docs/technical/aerodynamics/Roll_Equations.pdf
        clfDelta = (
            rollForcingInterferenceFactor * n * (Yma + radius) * clalphaSingleFin / d
        )  # Function of mach number
        cldOmega = (
            2
            * rollDampingInterferenceFactor
            * n
            * clalphaSingleFin
            * np.cos(cantAngleRad)
            * trapezoidalConstant
            / (Aref * d**2)
        )
        # Function of mach number
        rollParameters = [clfDelta, cldOmega, cantAngleRad]

        # Store values
        fin = {
            "cp": (0, 0, cpz),
            "cl": cl,
            "roll parameters": rollParameters,
            "name": "Fins",
        }
        self.aerodynamicSurfaces.append(fin)

        # Refresh static margin calculation
        self.evaluateStaticMargin()

        # Return self
        return self.aerodynamicSurfaces[-1]

    def addParachute(
        self, name, CdS, trigger, samplingRate=100, lag=0, noise=(0, 0, 0)
    ):
        """Creates a new parachute, storing its parameters such as
        opening delay, drag coefficients and trigger function.

        Parameters
        ----------
        name : string
            Parachute name, such as drogue and main. Has no impact in
            simulation, as it is only used to display data in a more
            organized matter.
        CdS : float
            Drag coefficient times reference area for parachute. It is
            used to compute the drag force exerted on the parachute by
            the equation F = ((1/2)*rho*V^2)*CdS, that is, the drag
            force is the dynamic pressure computed on the parachute
            times its CdS coefficient. Has units of area and must be
            given in squared meters.
        trigger : function
            Function which defines if the parachute ejection system is
            to be triggered. It must take as input the freestream
            pressure in pascal and the state vector of the simulation,
            which is defined by [x, y, z, vx, vy, vz, e0, e1, e2, e3, wx, wy, wz].
            It will be called according to the sampling rate given next.
            It should return True if the parachute ejection system is
            to be triggered and False otherwise.
        samplingRate : float, optional
            Sampling rate in which the trigger function works. It is used to
            simulate the refresh rate of onboard sensors such as barometers.
            Default value is 100. Value must be given in hertz.
        lag : float, optional
            Time between the parachute ejection system is triggered and the
            parachute is fully opened. During this time, the simulation will
            consider the rocket as flying without a parachute. Default value
            is 0. Must be given in seconds.
        noise : tuple, list, optional
            List in the format (mean, standard deviation, time-correlation).
            The values are used to add noise to the pressure signal which is
            passed to the trigger function. Default value is (0, 0, 0). Units
            are in pascal.

        Returns
        -------
        parachute : Parachute
            Parachute  containing trigger, samplingRate, lag, CdS, noise
            and name. Furthermore, it stores cleanPressureSignal,
            noiseSignal and noisyPressureSignal which are filled in during
            Flight simulation.
        """
        # Create a parachute
        parachute = Parachute(name, CdS, trigger, samplingRate, lag, noise)

        # Add parachute to list of parachutes
        self.parachutes.append(parachute)

        # Return self
        return self.parachutes[-1]

    def setRailButtons(self, positionRailButtons, angularPosition=45):
        """Adds rail buttons to the rocket, allowing for the
        calculation of forces exerted by them when the rocket is
        sliding in the launch rail. Furthermore, rail buttons are
        also needed for the simulation of the planar flight phase,
        when the rocket experiences 3 degrees of freedom motion while
        only one rail button is still in the launch rail.

        Parameters
        ----------
        positionRailButtons : tuple, list, array
            Two values organized in a tuple, list or array which
            represent the position of each of the two rail buttons
            relative to the considered coordinate system. The order
            does not matter. All values should be in meters.
        angularPosition : float
            Angular position of the rail buttons in degrees measured
            as the rotation around the symmetry axis of the rocket
            relative to one of the other principal axis.
            Default value is 45 degrees, generally used in rockets with
            4 fins.

        Returns
        -------
        None
        """
        # Calculate rail buttons position relative to cm
        railButtonsPosition_CM = [
            positionRailButton - self.positionCenterOfDryMass
            for positionRailButton in positionRailButtons
        ]

        # Order distance to CM
        if railButtonsPosition_CM[0] < railButtonsPosition_CM[1]:
            railButtonsPosition_CM.reverse()
        # Save
        self.railButtons = self.railButtonPair(
            railButtonsPosition_CM, positionRailButtons, angularPosition
        )

        return None

    def addCMEccentricity(self, x, y):
        """Moves line of action of aerodynamic and thrust forces by
        equal translation amount to simulate an eccentricity in the
        position of the center of mass of the rocket relative to its
        geometrical center line. Should not be used together with
        addCPEccentricity and addThrustEccentricity.

        Parameters
        ----------
        x : float
            Distance in meters by which the CM is to be translated in
            the x direction relative to geometrical center line.
        y : float
            Distance in meters by which the CM is to be translated in
            the y direction relative to geometrical center line.

        Returns
        -------
        self : Rocket
            Object of the Rocket class.
        """
        # Move center of pressure to -x and -y
        self.cpEccentricityX = -x
        self.cpEccentricityY = -y

        # Move thrust center by -x and -y
        self.thrustEccentricityY = -x
        self.thrustEccentricityX = -y

        # Return self
        return self

    def addCPEccentricity(self, x, y):
        """Moves line of action of aerodynamic forces to simulate an
        eccentricity in the position of the center of pressure relative
        to the center of mass of the rocket.

        Parameters
        ----------
        x : float
            Distance in meters by which the CP is to be translated in
            the x direction relative to the center of mass axial line.
        y : float
            Distance in meters by which the CP is to be translated in
            the y direction relative to the center of mass axial line.

        Returns
        -------
        self : Rocket
            Object of the Rocket class.
        """
        # Move center of pressure by x and y
        self.cpEccentricityX = x
        self.cpEccentricityY = y

        # Return self
        return self

    def addThrustEccentricity(self, x, y):
        """Moves line of action of thrust forces to simulate a
        misalignment of the thrust vector and the center of mass.

        Parameters
        ----------
        x : float
            Distance in meters by which the line of action of the
            thrust force is to be translated in the x direction
            relative to the center of mass axial line.
        y : float
            Distance in meters by which the line of action of the
            thrust force is to be translated in the x direction
            relative to the center of mass axial line.

        Returns
        -------
        self : Rocket
            Object of the Rocket class.
        """
        # Move thrust line by x and y
        self.thrustEccentricityY = x
        self.thrustEccentricityX = y

        # Return self
        return self

    def info(self):
        """Prints out a summary of the data and graphs available about
        the Rocket.

        Parameters
        ----------
        None

        Return
        ------
        None
        """
        # Print inertia details
        print("Inertia Details")
        print("Rocket Dry Mass: " + str(self.mass) + " kg (No Propellant)")
        print("Rocket Total Mass: " + str(self.totalMass(0)) + " kg (With Propellant)")

        # Print rocket geometrical parameters
        print("\nGeometrical Parameters")
        print("Rocket Radius: " + str(self.radius) + " m")

        # Print rocket aerodynamics quantities
        print("\nAerodynamics Stability")
        print("Initial Static Margin: " + "{:.3f}".format(self.staticMargin(0)) + " c")
        print(
            "Final Static Margin: "
            + "{:.3f}".format(self.staticMargin(self.motor.burnOutTime))
            + " c"
        )

        # Print parachute data
        for chute in self.parachutes:
            print("\n" + chute.name.title() + " Parachute")
            print("CdS Coefficient: " + str(chute.CdS) + " m2")

        # Show plots
        print("\nAerodynamics Plots")
        self.powerOnDrag()

        # Return None
        return None

    def allInfo(self):
        """Prints out all data and graphs available about the Rocket.

        Parameters
        ----------
        None

        Return
        ------
        None
        """
        # Print inertia details
        print("Inertia Details")
        print("Rocket Mass: {:.3f} kg (No Propellant)".format(self.mass))
        print("Rocket Mass: {:.3f} kg (With Propellant)".format(self.totalMass(0)))
        print("Rocket Inertia I: {:.3f} kg*m2".format(self.inertiaI))
        print("Rocket Inertia Z: {:.3f} kg*m2".format(self.inertiaZ))

        # Print rocket geometrical parameters
        print("\nGeometrical Parameters")
        print("Rocket Maximum Radius: " + str(self.radius) + " m")
        print("Rocket Frontal Area: " + "{:.6f}".format(self.area) + " m2")
        print("\nRocket Distances")
        print(
            "Rocket Center of Mass - Nozzle Exit Distance: "
            + str(-self.positionCenterOfDryMassToNozzle)
            + " m"
        )
        print(
            "Rocket Center of Mass - Motor reference point: "
            + str(self.positionMotorReferencePositionToCenterOfDryMass)
            + " m"
        )
        print(
            "Rocket Center of Mass - Rocket Loaded Center of Mass: "
            + "{:.3f}".format(self.centerOfMass(0))
            + " m"
        )
        print("\nAerodynamic Components Parameters")
        print("Currently not implemented.")

        # Print rocket aerodynamics quantities
        print("\nAerodynamics Lift Coefficient Derivatives")
        for aerodynamicSurface in self.aerodynamicSurfaces:
            name = aerodynamicSurface["name"]
            clalpha = Function(
                lambda alpha: aerodynamicSurface["cl"](alpha, 0),
            ).differentiate(x=1e-2, dx=1e-3)
            print(
                name + " Lift Coefficient Derivative: {:.3f}".format(clalpha) + "/rad"
            )

        print("\nAerodynamics Center of Pressure")
        for aerodynamicSurface in self.aerodynamicSurfaces:
            name = aerodynamicSurface["name"]
            cpz = aerodynamicSurface["cp"][2]
            print(name + " Center of Pressure to CM: {:.3f}".format(cpz) + " m")
        print(
            "Distance - Center of Pressure to CM: "
            + "{:.3f}".format(self.cpPosition)
            + " m"
        )
        print("Initial Static Margin: " + "{:.3f}".format(self.staticMargin(0)) + " c")
        print(
            "Final Static Margin: "
            + "{:.3f}".format(self.staticMargin(self.motor.burnOutTime))
            + " c"
        )

        # Print parachute data
        for chute in self.parachutes:
            print("\n" + chute.name.title() + " Parachute")
            print("CdS Coefficient: " + str(chute.CdS) + " m2")
            if chute.trigger.__name__ == "<lambda>":
                line = getsourcelines(chute.trigger)[0][0]
                print(
                    "Ejection signal trigger: "
                    + line.split("lambda ")[1].split(",")[0].split("\n")[0]
                )
            else:
                print("Ejection signal trigger: " + chute.trigger.__name__)
            print("Ejection system refresh rate: " + str(chute.samplingRate) + " Hz.")
            print(
                "Time between ejection signal is triggered and the "
                "parachute is fully opened: " + str(chute.lag) + " s"
            )

        # Show plots
        print("\nMass Plots")
        self.totalMass()
        self.reducedMass()
        print("\nAerodynamics Plots")
        self.staticMargin()
        self.powerOnDrag()
        self.powerOffDrag()
        self.thrustToWeight.plot(lower=0, upper=self.motor.burnOutTime)

        # ax = plt.subplot(415)
        # ax.plot(  , self.rocket.motor.thrust()/(self.env.g() * self.rocket.totalMass()))
        # ax.set_xlim(0, self.rocket.motor.burnOutTime)
        # ax.set_xlabel("Time (s)")
        # ax.set_ylabel("Thrust/Weight")
        # ax.set_title("Thrust-Weight Ratio")

        # Return None
        return None

    def addFin(
        self,
        numberOfFins=4,
        cl=2 * np.pi,
        cpr=1,
        cpz=1,
        gammas=[0, 0, 0, 0],
        angularPositions=None,
    ):
        "Hey! I will document this function later"
        self.aerodynamicSurfaces = []
        pi = np.pi
        # Calculate angular positions if not given
        if angularPositions is None:
            angularPositions = np.array(range(numberOfFins)) * 2 * pi / numberOfFins
        else:
            angularPositions = np.array(angularPositions) * pi / 180
        # Convert gammas to degree
        if isinstance(gammas, (int, float)):
            gammas = [(pi / 180) * gammas for i in range(numberOfFins)]
        else:
            gammas = [(pi / 180) * gamma for gamma in gammas]
        for i in range(numberOfFins):
            # Get angular position and inclination for current fin
            angularPosition = angularPositions[i]
            gamma = gammas[i]
            # Calculate position vector
            cpx = cpr * np.cos(angularPosition)
            cpy = cpr * np.sin(angularPosition)
            positionVector = np.array([cpx, cpy, cpz])
            # Calculate chord vector
            auxVector = np.array([cpy, -cpx, 0]) / (cpr)
            chordVector = (
                np.cos(gamma) * np.array([0, 0, 1]) - np.sin(gamma) * auxVector
            )
            self.aerodynamicSurfaces.append([positionVector, chordVector])
        return None

    # Variables
    railButtonPair = namedtuple(
        "railButtonPair", "distanceToCM distanceToReference angularPosition"
    )

    # Helper functions
    def evaluatePositionSurface_Nozzle(self, surfaceName, positionSurface):
        """Calculates and returns the position of an aerodynamic surface
        relative to the nozzle exit. The relative position to the Nozzle
        considers the direction towards the rocket tip to be positive.
        The calculations take into account the possibility of the surface
        to be set behind the nozzle, meaning its relative position must
        be negative.

        Parameters
        ----------
        surfaceName : string
            Name of the aerodynamic surface.
        positionSurface : float
            Position of the aerodynamic surface relative to the coordinate
            system considered for the inputs.

        Returns
        -------
        surfacePosition_Nozzle : float
            The relative position of the aerodynamic surface relative to the nozzle
        """
        if self.positionNozzle == self.positionCenterOfDryMass:
            # Nozzle and Center of Mass are at the same position
            # Impossible to know if Surface is in front or behind the Nozzle
            # Unless Surface is also at the same position
            # Surface is then assumed to be in front of the Nozzle and a warning is raised
            if positionSurface != self.positionNozzle:
                warnings.warn(
                    "Can not determine if ",
                    surfaceName,
                    " are in front or behind the nozzle.\n",
                    "Calculations will assume they are in front.\n",
                    "This happens when the reference point is at the ",
                    surfaceName,
                    " Surface position ",
                    "and when the center of dry mass is at the same position as the nozzle.",
                )
            return abs(positionSurface - self.positionNozzle)

        elif positionSurface == 0:
            # Surface is at the coordinate system origin

            if np.sign(self.positionCenterOfDryMass * self.positionNozzle) == 1:
                # Nozzle and Center of Mass at the same side of the Surface
                # Meaning Surface is either behind the Nozzle but closer to CM
                # Or behind the Nozzle and further away from CM

                if abs(self.positionCenterOfDryMass) < abs(self.positionNozzle):
                    # Surface is closer to Center of dry mass, therefore in front of the Nozzle
                    return abs(
                        positionSurface - self.positionNozzle
                    )  # positive value since Surface is before the Nozzle

                elif abs(self.positionCenterOfDryMass) > abs(self.positionNozzle):
                    # Surface is closer to Nozzle, therefore behind the Nozzle
                    return -abs(
                        positionSurface - self.positionNozzle
                    )  # negative value since Surface is after the Nozzle

            elif np.sign(self.positionCenterOfDryMass * self.positionNozzle) == -1:
                # Surface is in between the Center of dry mass and the Nozzle
                # Meaning Surface is in front of the Nozzle
                return abs(positionSurface - self.positionNozzle)
            else:
                # Nozzle or Center of Mass are at the coordinate system origin
                # Meaning Surface is either at the Center of Mass or at the Nozzle
                return abs(positionSurface - self.positionNozzle)

        elif np.sign(positionSurface * self.positionNozzle) == 1:
            # Surface and Nozzle are at the same side of the coordinate system origin

            if np.sign(positionSurface * self.positionCenterOfDryMass) == 1:
                # Surface and Center of Mass at the same side of the coordinate system origin
                # Therefore Center of Mass is at the same side of the Nozzle and Surface

                if abs(self.positionCenterOfDryMass) < abs(self.positionNozzle):
                    # Center of Mass is closer to coordinate system then the Nozzle
                    # Meaning coordinate system is set behind the Nozzle

                    if abs(positionSurface) <= abs(self.positionNozzle):
                        # Surface is set before or at the Nozzle
                        return abs(positionSurface - self.positionNozzle)

                    else:  # Surface is set after the Nozzle
                        return -abs(positionSurface - self.positionNozzle)

                elif abs(self.positionCenterOfDryMass) > abs(self.positionNozzle):
                    # Center of Mass is further from coordinate system then the Nozzle
                    # Meaning coordinate system is set after the Nozzle

                    if abs(positionSurface) >= abs(self.positionNozzle):
                        # Surface is set before or at the Nozzle
                        return abs(positionSurface - self.positionNozzle)

                    else:  # Surface is set after the Nozzle
                        return -abs(positionSurface - self.positionNozzle)

            elif np.sign(positionSurface * self.positionCenterOfDryMass) == -1:
                # Surface and Center of Mass at different sides of the coordinate system
                # origin (therefore Center of Mass is at a different side of the Nozzle).
                # Meaning the coordinate system is before the Nozzle

                if abs(positionSurface) <= abs(self.positionNozzle):
                    # Surface is set before or at the Nozzle
                    return abs(positionSurface - self.positionNozzle)

                else:  # Surface is set after the Nozzle
                    return -abs(positionSurface - self.positionNozzle)

            else:  # Center of mass is set at the coordinate system origin
                if abs(positionSurface) <= abs(self.positionNozzle):
                    # Surface is set before or at the Nozzle
                    return abs(positionSurface - self.positionNozzle)
                else:  # Surface is set after the Nozzle
                    return -abs(positionSurface - self.positionNozzle)

        elif np.sign(positionSurface * self.positionNozzle) == -1:
            # Surface and Nozzle at different sides of the coordinate system origin

            if np.sign(positionSurface * self.positionCenterOfDryMass) == -1:
                # Surface and Center of Mass at different sides (Center of Mass at the same side of Nozzle)

                if abs(self.positionCenterOfDryMass) < abs(self.positionNozzle):
                    # Center of Mass closer to the origin (and therefore the Surface) than the Nozzle
                    # Meaning Surface must be set before the Nozzle
                    return abs(positionSurface - self.positionNozzle)

                elif abs(self.positionCenterOfDryMass) > abs(self.positionNozzle):
                    # Center of Mass closer to the origin (and therefore the Surface) than the Nozzle
                    # Meaning Surface must be set after the Nozzle
                    return -abs(positionSurface - self.positionNozzle)

            else:
                # Surface and Center of Mass at the same side of the coordinate system origin
                # Meaning Surface must be set before the Nozzle
                return abs(positionSurface - self.positionNozzle)

        else:  # Nozzle is at the coordinate system origin

            if np.sign(positionSurface * self.positionCenterOfDryMass) == 1:
                # Surface and Center of Mass at the same side of the coordinate system origin
                # Meaning Surface is set before the Nozzle
                return abs(positionSurface - self.positionNozzle)

            elif np.sign(positionSurface * self.positionCenterOfDryMass) == -1:
                # Surface and Center of Mass are at different sides of the coordinate system origin
                # Meaning Surface is set after the Nozzle
                return -abs(positionSurface - self.positionNozzle)
