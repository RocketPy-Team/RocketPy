__author__ = "Guilherme Fernandes Alves"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

import numpy as np
from .Function import Function


class NoseCone:
    """Keeps nose cone information.

    Attributes
    ----------
    NoseCone.length : float
        Nose cone length. Has units of length and must be given in meters.
    NoseCone.kind : string
        Nose cone kind. Can be "conical", "ogive" or "lvhaack".
    NoseCone.distanceToCM : float
        Distance between nose cone tip and rocket center of mass. Has units of
        length and must be given in meters.
    NoseCone.name : string
        Nose cone name. Has no impact in simulation, as it is only used to
        display data in a more organized matter.
    NoseCone.cp : tuple
        Tuple with the x, y and z coordinates of the nose cone center of pressure
        relative to the rocket center of mass. Has units of length and must be
        given in meters.
    NoseCone.cl : Function
        Function which defines the lift coefficient as a function of the angle of
        attack and the Mach number. It must take as input the angle of attack in
        radians and the Mach number. It should return the lift coefficient.
    """

    def __init__(self, length, kind, distanceToCM, name="Nose Cone"):
        """Initializes the nose cone. It is used to define the nose cone
        length, kind, distance to center of mass and name.

        Parameters
        ----------
        length : float
            Nose cone length. Has units of length and must be given in meters.
        kind : string
            Nose cone kind. Can be "conical", "ogive" or "lvhaack".
        distanceToCM : _type_
            Distance between nose cone tip and rocket center of mass. Has units of
            length and must be given in meters.
        name : str, optional
            Nose cone name. Has no impact in simulation, as it is only used to
            display data in a more organized matter.

        Returns
        -------
        None
        """
        self.length = length
        self.kind = kind
        self.distanceToCM = distanceToCM
        self.name = name

        # Analyze type
        if self.kind == "conical":
            self.k = 1 - 1 / 3
        elif self.kind == "ogive":
            self.k = 1 - 0.534
        elif self.kind == "lvhaack":
            self.k = 1 - 0.437
        else:
            self.k = 0.5
        # Calculate cp position relative to cm
        self.cpz = self.distanceToCM + np.sign(self.distanceToCM) * self.k * length
        self.cpy = 0
        self.cpx = 0
        self.cp = (self.cpx, self.cpy, self.cpz)

        # Calculate clalpha
        self.clalpha = 2
        self.cl = Function(
            lambda alpha, mach: self.clalpha * alpha,
            ["Alpha (rad)", "Mach"],
            "Cl",
        )
        # # Store values
        # nose = {"cp": (0, 0, self.cpz), "cl": self.cl, "name": name}

        return None


class TrapezoidalFins:
    """Keeps trapezoidal fins information.

    Attributes
    ----------

    """

    def __init__(
        self,
        n,
        rootChord,
        tipChord,
        span,
        distanceToCM,
        cantAngle=0,
        sweepLength=None,
        sweepAngle=None,
        radius=None,
        airfoil=None,
        name="Fins",
    ):
        """Initializes the trapezoidal fins. It is used to define the number of
        fins, root chord, tip chord, span, distance to center of mass, cant angle
        and name.

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
        distanceToCM : int, float
            Fin set position relative to rocket unloaded center of
            mass, considering positive direction from center of mass to
            nose cone. Consider the center point belonging to the top
            of the fins to calculate distance.
        cantAngle : int, float, optional
            Fins cant angle with respect to the rocket centerline. Must
            be given in degrees.
        radius : int, float, optional
            Reference radius to calculate lift coefficient. If None, which
            is default, use rocket radius. Otherwise, enter the radius
            of the rocket in the section of the fins, as this impacts
            its lift coefficient.
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
        None
        """
        # Store values
        self.numberOfFins = n
        self.finRadius = radius
        self.finAirfoil = airfoil
        self.finDistanceToCM = distanceToCM
        self.finCantAngle = cantAngle
        self.finRootChord = rootChord
        self.finTipChord = tipChord
        self.finSpan = span
        self.name = name

        # get some nicknames
        Cr, Ct = self.finRootChord, self.finTipChord
        s = self.finSpan
        cantAngleRad = np.radians(cantAngle)

        # Check if sweep angle or sweep length is given
        if sweepLength is not None and sweepAngle is not None:
            raise ValueError("Cannot use sweepLength and sweepAngle together")
        elif sweepAngle is not None:
            sweepLength = np.tan(sweepAngle * np.pi / 180) * span
        elif sweepLength is None:
            sweepLength = Cr - Ct
        else:
            # Sweep length is given
            pass

        # Compute auxiliary geometrical parameters
        d = 2 * radius
        Aref = np.pi * radius**2
        Yr = Cr + Ct
        Af = Yr * s / 2  # Fin area
        AR = 2 * s**2 / Af  # Fin aspect ratio
        gamma_c = np.arctan(
            (sweepLength + 0.5 * Ct - 0.5 * Cr) / (span)
        )  # Mid chord angle
        Yma = (s / 3) * (Cr + 2 * Ct) / Yr  # Span wise coord of mean aero chord

        # Center of pressure position relative to CDM (center of dry mass)
        cpz = distanceToCM + np.sign(distanceToCM) * (
            ((Cr - Ct) / 3) * ((Cr + 2 * Ct) / (Cr + Ct))
            + (1 / 6) * (Cr + Ct - Cr * Ct / (Cr + Ct))
        )

        # Fin–body interference correction parameters
        tau = (s + radius) / radius
        liftInterferenceFactor = 1 + 1 / tau
        λ = Ct / Cr

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

        # Defines number of fins  factor
        def finNumCorrection(n):
            """Calculates a correction factor for the lift coefficient of multiple fins.
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
        # Documented at: https://github.com/RocketPy-Team/RocketPy/blob/master/docs/technical/aerodynamics/Roll_Equations.pdf
        rollDampingInterferenceFactor = 1 + (
            ((tau - λ) / (tau)) - ((1 - λ) / (tau - 1)) * np.log(tau)
        ) / (
            ((tau + 1) * (tau - λ)) / (2) - ((1 - λ) * (tau**3 - 1)) / (3 * (tau - 1))
        )
        rollForcingInterferenceFactor = (1 / np.pi**2) * (
            (np.pi**2 / 4) * ((tau + 1) ** 2 / tau**2)
            + ((np.pi * (tau**2 + 1) ** 2) / (tau**2 * (tau - 1) ** 2))
            * np.arcsin((tau**2 - 1) / (tau**2 + 1))
            - (2 * np.pi * (tau + 1)) / (tau * (tau - 1))
            + ((tau**2 + 1) ** 2)
            / (tau**2 * (tau - 1) ** 2)
            * (np.arcsin((tau**2 - 1) / (tau**2 + 1))) ** 2
            - (4 * (tau + 1))
            / (tau * (tau - 1))
            * np.arcsin((tau**2 - 1) / (tau**2 + 1))
            + (8 / (tau - 1) ** 2) * np.log((tau**2 + 1) / (2 * tau))
        )
        clfDelta = (
            rollForcingInterferenceFactor * n * (Yma + radius) * clalphaSingleFin / d
        )  # Function of mach number
        cldOmega = (
            2
            * rollDampingInterferenceFactor
            * n
            * clalphaSingleFin
            * np.cos(cantAngleRad)
            * rollGeometricalConstant
            / (Aref * d**2)
        )  # Function of mach number
        rollParameters = [clfDelta, cldOmega, cantAngleRad]

        # Save and store parameters
        self.rollParameters = rollParameters
        self.cl = cl
        self.clalphaSingleFin = clalphaSingleFin
        self.clalphaMultipleFins = clalphaMultipleFins
        self.cpx = 0
        self.cpy = 0
        self.cpz = cpz
        self.cp = (self.cpx, self.cpy, self.cpz)

        def info(self):
            "Still not implemented. Must print important information."

            return None

        return None


class EllipticalFins:
    """Class that defines the aerodynamic model of an elliptical fin.

    Parameters
    ----------
    n : int
        Number of fins.
    radius : int, float
        Fin radius.
    span : int, float
        Fin span.
    cantAngle : int, float
        Cant angle of the fin.

    Returns
    -------
    None
    """

    def __init__(
        self,
        n,
        rootChord,
        span,
        distanceToCM,
        cantAngle=0,
        radius=None,
        airfoil=None,
        name="Fins",
    ):
        """Initializes the class, defining the parameters of the fins.

        Parameters
        ----------
        n : int
            Number of fins.
        rootChord : _type_
            _description_
        span : _type_
            _description_
        distanceToCM : _type_
            _description_
        cantAngle : int, optional
            _description_, by default 0
        radius : _type_, optional
            _description_, by default None
        airfoil : _type_, optional
            _description_, by default None
        name : str, optional
            _description_, by default "Fins"

        Returns
        -------
        None

        """

        # Save attributes
        self.numberOfFins = n
        self.rootChord = rootChord
        self.span = span
        self.distanceToCM = distanceToCM
        self.cantAngle = cantAngle
        self.radius = radius
        self.airfoil = airfoil
        self.name = name

        # Get some nicknames
        Cr = self.rootChord
        s = self.span
        cantAngleRad = np.radians(cantAngle)

        # Compute auxiliary geometrical parameters
        d = 2 * radius
        Aref = np.pi * radius**2  # Reference area for coefficients
        Af = (np.pi * Cr / 2 * s) / 2  # Fin area
        AR = 2 * s**2 / Af  # Fin aspect ratio
        Yma = (
            s / (3 * np.pi) * np.sqrt(9 * np.pi**2 - 16)
        )  # Span wise coord of mean aero chord
        rollGeometricalConstant = (
            Cr
            * s
            * (3 * np.pi * s**2 + 32 * radius * s + 12 * np.pi * radius**2)
            / 48
        )

        # Center of pressure position relative to CDM (center of dry mass)
        cpz = distanceToCM + np.sign(distanceToCM) * (0.288 * Cr)

        # Fin–body interference correction parameters
        tau = (s + radius) / radius
        liftInterferenceFactor = 1 + 1 / tau
        rollDampingInterferenceFactor = 1 + (
            (radius**2)
            * (
                2
                * (radius**2)
                * np.sqrt(s**2 - radius**2)
                * np.log((2 * s * np.sqrt(s**2 - radius**2) + 2 * s**2) / radius)
                - 2 * (radius**2) * np.sqrt(s**2 - radius**2) * np.log(2 * s)
                + 2 * s**3
                - np.pi * radius * s**2
                - 2 * (radius**2) * s
                + np.pi * radius**3
            )
        ) / (2 * (s**2) * (s / 3 + np.pi * radius / 4) * (s**2 - radius**2))
        rollForcingInterferenceFactor = (1 / np.pi**2) * (
            (np.pi**2 / 4) * ((tau + 1) ** 2 / tau**2)
            + ((np.pi * (tau**2 + 1) ** 2) / (tau**2 * (tau - 1) ** 2))
            * np.arcsin((tau**2 - 1) / (tau**2 + 1))
            - (2 * np.pi * (tau + 1)) / (tau * (tau - 1))
            + ((tau**2 + 1) ** 2)
            / (tau**2 * (tau - 1) ** 2)
            * (np.arcsin((tau**2 - 1) / (tau**2 + 1))) ** 2
            - (4 * (tau + 1))
            / (tau * (tau - 1))
            * np.arcsin((tau**2 - 1) / (tau**2 + 1))
            + (8 / (tau - 1) ** 2) * np.log((tau**2 + 1) / (2 * tau))
        )

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
        FD = 2 * np.pi * AR / (clalpha2D)

        # Lift coefficient derivative for a single fin
        clalphaSingleFin = Function(
            lambda mach: (clalpha2D(mach) * FD(mach) * (Af / Aref))
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
        # Documented at: https://github.com/RocketPy-Team/RocketPy/blob/develop/docs/technical/aerodynamics/Roll_Equations.pdf
        clfDelta = (
            rollForcingInterferenceFactor * n * (Yma + radius) * clalphaSingleFin / d
        )  # Function of mach number
        cldOmega = (
            2
            * rollDampingInterferenceFactor
            * n
            * clalphaSingleFin
            * np.cos(cantAngleRad)
            * rollGeometricalConstant
            / (Aref * d**2)
        )
        # Function of mach number
        rollParameters = [clfDelta, cldOmega, cantAngleRad]

        # Store values
        self.cpx = 0
        self.cpy = 0
        self.cpz = cpz
        self.cp = (self.cpx, self.cpy, self.cpz)
        self.cl = cl
        self.rollParameters = rollParameters
        self.clalphaMultipleFins = clalphaMultipleFins
        self.clalphaSingleFin = clalphaSingleFin

        return None


class Tail:
    """Class that defines a tail for the rocket.

    Parameters
    ----------
    length : int, float
        Length of the tail.
    ...


    """

    def __init__(
        self, topRadius, bottomRadius, length, distanceToCM, radius, name="Tail"
    ):
        """_summary_

        Parameters
        ----------
        topRadius : _type_
            _description_
        bottomRadius : _type_
            _description_
        length : _type_
            _description_
        distanceToCM : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """

        # Store arguments as attributes
        self.tailTopRadius = topRadius
        self.tailBottomRadius = bottomRadius
        self.tailLength = length
        self.tailDistanceToCM = distanceToCM
        self.name = name
        self.radius = radius

        # Calculate ratio between top and bottom radius
        r = topRadius / bottomRadius

        # Retrieve reference radius
        rref = self.radius

        # Calculate cp position relative to center of dry mass
        if distanceToCM < 0:
            cpz = distanceToCM - (length / 3) * (1 + (1 - r) / (1 - r**2))
        else:
            cpz = distanceToCM + (length / 3) * (1 + (1 - r) / (1 - r**2))

        # Calculate clalpha
        clalpha = -2 * (1 - r ** (-2)) * (topRadius / rref) ** 2
        cl = Function(
            lambda alpha, mach: clalpha * alpha,
            ["Alpha (rad)", "Mach"],
            "Cl",
        )

        # Store values as class attributes
        self.cpx = 0
        self.cpy = 0
        self.cpz = cpz
        self.cp = (self.cpx, self.cpy, self.cpz)
        self.cl = cl
        self.clalpha = clalpha

        return None
