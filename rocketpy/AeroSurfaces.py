__author__ = "Guilherme Fernandes Alves"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
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


class Fins:
    def evaluateLiftCoefficient(self):
        if not self.airfoil:
            # Defines clalpha2D as 2*pi for planar fins
            clalpha2D = Function(lambda mach: 2 * np.pi / self._beta(mach))
        else:
            # Defines clalpha2D as the derivative of the
            # lift coefficient curve for a specific airfoil
            airfoilCl = Function(
                self.airfoil[0],
                interpolation="linear",
            )

            # Differentiating at x = 0 to get cl_alpha
            clalpha2D_Mach0 = airfoilCl.differentiate(x=1e-3, dx=1e-3)

            # Convert to radians if needed
            if self.airfoil[1] == "degrees":
                clalpha2D_Mach0 *= 180 / np.pi

            # Correcting for compressible flow
            clalpha2D = Function(lambda mach: clalpha2D_Mach0 / self._beta(mach))

        # Diederich's Planform Correlation Parameter
        FD = 2 * np.pi * self.AR / (clalpha2D * np.cos(self.gamma_c))

        # Lift coefficient derivative for a single fin
        self.clalphaSingleFin = Function(
            lambda mach: (
                clalpha2D(mach)
                * FD(mach)
                * (self.Af / self.Aref)
                * np.cos(self.gamma_c)
            )
            / (2 + FD(mach) * np.sqrt(1 + (2 / FD(mach)) ** 2))
        )

        # Lift coefficient derivative for a number of n fins corrected for Fin-Body interference
        self.clalphaMultipleFins = (
            self.liftInterferenceFactor
            * self._finNumCorrection(self.n)
            * self.clalphaSingleFin
        )  # Function of mach number

        # Calculates clalpha * alpha
        self.cl = Function(
            lambda alpha, mach: alpha * self.clalphaMultipleFins(mach),
            ["Alpha (rad)", "Mach"],
            "Cl",
        )

        return self

    def evaluateRollCoefficients(self):
        clfDelta = (
            self.rollForcingInterferenceFactor
            * self.n
            * (self.Yma + self.radius)
            * self.clalphaSingleFin
            / self.d
        )  # Function of mach number
        cldOmega = (
            2
            * self.rollDampingInterferenceFactor
            * self.n
            * self.clalphaSingleFin
            * np.cos(self.cantAngleRad)
            * self.rollGeometricalConstant
            / (self.Aref * self.d**2)
        )  # Function of mach number
        self.rollParameters = [clfDelta, cldOmega, self.cantAngleRad]
        return self

    def changeCantAngle(self, cantAngle):
        self.cantAngleList.append(cantAngle)

        self.cantAngle = cantAngle
        self.cantAngleRad = np.radians(cantAngle)

        self.evaluateRollCoefficients()

        return self

    # Defines beta parameter
    def _beta(_, mach):
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
    def _finNumCorrection(_, n):
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


class TrapezoidalFins(Fins):
    def __init__(
        self,
        n,
        rootChord,
        tipChord,
        span,
        distanceToCM,
        radius,
        cantAngle=0,
        sweepLength=None,
        sweepAngle=None,
        airfoil=None,
        name="Fins",
    ):

        # Check if sweep angle or sweep length is given
        if sweepLength is not None and sweepAngle is not None:
            raise ValueError("Cannot use sweepLength and sweepAngle together")
        elif sweepAngle is not None:
            sweepLength = np.tan(sweepAngle * np.pi / 180) * span
        elif sweepLength is None:
            sweepLength = rootChord - tipChord
        else:
            # Sweep length is given
            pass

        # Compute auxiliary geometrical parameters
        d = 2 * radius
        Aref = np.pi * radius**2  # Reference area
        Yr = rootChord + tipChord
        Af = Yr * span / 2  # Fin area
        AR = 2 * span**2 / Af  # Fin aspect ratio
        gamma_c = np.arctan(
            (sweepLength + 0.5 * tipChord - 0.5 * rootChord) / (span)
        )  # Mid chord angle
        Yma = (
            (span / 3) * (rootChord + 2 * tipChord) / Yr
        )  # Span wise coord of mean aero chord

        rollGeometricalConstant = (
            (rootChord + 3 * tipChord) * span**3
            + 4 * (rootChord + 2 * tipChord) * radius * span**2
            + 6 * (rootChord + tipChord) * span * radius**2
        ) / 12

        # Fin–body interference correction parameters
        tau = (span + radius) / radius
        liftInterferenceFactor = 1 + 1 / tau

        # Store values
        self.n = n
        self.radius = radius
        self.airfoil = airfoil
        self.distanceToCM = distanceToCM
        self.cantAngle = cantAngle
        self.cantAngleList = [cantAngle]
        self.cantAngleRad = np.radians(cantAngle)
        self.rootChord = rootChord
        self.tipChord = tipChord
        self.span = span
        self.name = name
        self.sweepLength = sweepLength
        self.sweepAngle = sweepAngle
        self.d = d
        self.Aref = Aref  # Reference area
        self.Yr = Yr
        self.Af = Af * span / 2  # Fin area
        self.AR = AR  # Fin aspect ratio
        self.gamma_c = gamma_c  # Mid chord angle
        self.Yma = Yma  # Span wise coord of mean aero chord
        self.rollGeometricalConstant = rollGeometricalConstant
        self.tau = tau
        self.liftInterferenceFactor = liftInterferenceFactor

        self.evaluateCenterOfPressure()
        self.evaluateLiftCoefficient()

        if cantAngle:
            # Parameters for Roll Moment.
            # Documented at: https://github.com/RocketPy-Team/RocketPy/blob/master/docs/technical/aerodynamics/Roll_Equations.pdf
            self.λ = tipChord / rootChord
            self.rollDampingInterferenceFactor = 1 + (
                ((tau - self.λ) / (tau)) - ((1 - self.λ) / (tau - 1)) * np.log(tau)
            ) / (
                ((tau + 1) * (tau - self.λ)) / (2)
                - ((1 - self.λ) * (tau**3 - 1)) / (3 * (tau - 1))
            )
            self.rollForcingInterferenceFactor = (1 / np.pi**2) * (
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
            self.evaluateRollCoefficients()

    def evaluateCenterOfPressure(self):
        # Center of pressure position relative to CDM (center of dry mass)
        cpz = self.distanceToCM + np.sign(self.distanceToCM) * (
            (self.sweepLength / 3)
            * ((self.rootChord + 2 * self.tipChord) / (self.rootChord + self.tipChord))
            + (1 / 6)
            * (
                self.rootChord
                + self.tipChord
                - self.rootChord * self.tipChord / (self.rootChord + self.tipChord)
            )
        )
        self.cpx = 0
        self.cpy = 0
        self.cpz = cpz
        self.cp = (self.cpx, self.cpy, self.cpz)
        return self.cp

    def draw(self):
        # Color cycle [#348ABD, #A60628, #7A68A6, #467821, #D55E00, #CC79A7, #56B4E9, #009E73, #F0E442, #0072B2]
        # Fin
        leadingEdge = plt.Line2D((0, self.sweepLength), (0, self.span), color="#A60628")
        tip = plt.Line2D(
            (self.sweepLength, self.sweepLength + self.tipChord),
            (self.span, self.span),
            color="#A60628",
        )
        backEdge = plt.Line2D(
            (self.sweepLength + self.tipChord, self.rootChord),
            (self.span, 0),
            color="#A60628",
        )
        root = plt.Line2D((self.rootChord, 0), (0, 0), color="#A60628")

        # Center and Quarter line
        center_line = plt.Line2D(
            (self.rootChord / 2, self.sweepLength + self.tipChord / 2),
            (0, self.span),
            color="#7A68A6",
            alpha=0.35,
            linestyle="--",
            label="Center Line",
        )
        quarter_line = plt.Line2D(
            (self.rootChord / 4, self.sweepLength + self.tipChord / 4),
            (0, self.span),
            color="#7A68A6",
            alpha=1,
            linestyle="--",
            label="Quarter Line",
        )

        # Center of pressure
        cp_point = [abs(self.distanceToCM - self.cpz), self.Yma]

        # Mean Aerodynamic Chord
        Yma_start = (
            self.sweepLength
            * (self.rootChord + 2 * self.tipChord)
            / (3 * (self.rootChord + self.tipChord))
        )
        Yma_end = (
            2 * self.rootChord**2
            + self.rootChord * self.sweepLength
            + 2 * self.rootChord * self.tipChord
            + 2 * self.sweepLength * self.tipChord
            + 2 * self.tipChord**2
        ) / (3 * (self.rootChord + self.tipChord))
        Yma_line = plt.Line2D(
            (Yma_start, Yma_end),
            (self.Yma, self.Yma),
            color="#467821",
            linestyle="--",
            label="Mean Aerodynamic Chord",
        )

        # Plotting
        fig3 = plt.figure(figsize=(4, 4))
        with plt.style.context("bmh"):
            ax1 = fig3.add_subplot(111)

        # Fin
        ax1.add_line(leadingEdge)
        ax1.add_line(tip)
        ax1.add_line(backEdge)
        ax1.add_line(root)

        ax1.add_line(center_line)
        ax1.add_line(quarter_line)
        ax1.add_line(Yma_line)
        ax1.scatter(
            *cp_point, label="Center Of Pressure", color="red", s=100, zorder=10
        )
        ax1.scatter(*cp_point, facecolors="none", edgecolors="red", s=500, zorder=10)

        # Plot settings
        xlim = (
            self.rootChord
            if self.sweepLength + self.tipChord < self.rootChord
            else self.sweepLength + self.tipChord
        )
        ax1.set_xlim(0, xlim * 1.1)
        ax1.set_ylim(0, self.span * 1.1)
        ax1.set_xlabel("Root Chord")
        ax1.set_ylabel("Span")
        ax1.set_title("Trapezoidal Fin")
        ax1.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")


class EllipticalFins(Fins):
    def __init__(
        self,
        n,
        rootChord,
        span,
        distanceToCM,
        radius,
        cantAngle=0,
        airfoil=None,
        name="Fins",
    ):

        # Compute auxiliary geometrical parameters
        d = 2 * radius
        Aref = np.pi * radius**2  # Reference area for coefficients
        Af = (np.pi * rootChord / 2 * span) / 2  # Fin area
        gamma_c = 0  # Zero for elliptical fins
        AR = 2 * span**2 / Af  # Fin aspect ratio
        Yma = (
            span / (3 * np.pi) * np.sqrt(9 * np.pi**2 - 64)
        )  # Span wise coord of mean aero chord
        rollGeometricalConstant = (
            rootChord
            * span
            * (3 * np.pi * span**2 + 32 * radius * span + 12 * np.pi * radius**2)
            / 48
        )

        # Fin–body interference correction parameters
        tau = (span + radius) / radius
        liftInterferenceFactor = 1 + 1 / tau

        # Store values
        self.n = n
        self.radius = radius
        self.airfoil = airfoil
        self.distanceToCM = distanceToCM
        self.cantAngle = cantAngle
        self.cantAngleList = [cantAngle]
        self.cantAngleRad = np.radians(cantAngle)
        self.rootChord = rootChord
        self.span = span
        self.name = name
        self.d = d
        self.Aref = Aref  # Reference area
        self.Af = Af * span / 2  # Fin area
        self.AR = AR  # Fin aspect ratio
        self.gamma_c = gamma_c  # Mid chord angle
        self.Yma = Yma  # Span wise coord of mean aero chord
        self.rollGeometricalConstant = rollGeometricalConstant
        self.tau = tau
        self.liftInterferenceFactor = liftInterferenceFactor

        self.evaluateCenterOfPressure()
        self.evaluateLiftCoefficient()

        if cantAngle:
            self.rollDampingInterferenceFactor = 1 + (
                (radius**2)
                * (
                    2
                    * (radius**2)
                    * np.sqrt(span**2 - radius**2)
                    * np.log(
                        (2 * span * np.sqrt(span**2 - radius**2) + 2 * span**2)
                        / radius
                    )
                    - 2
                    * (radius**2)
                    * np.sqrt(span**2 - radius**2)
                    * np.log(2 * span)
                    + 2 * span**3
                    - np.pi * radius * span**2
                    - 2 * (radius**2) * span
                    + np.pi * radius**3
                )
            ) / (
                2
                * (span**2)
                * (span / 3 + np.pi * radius / 4)
                * (span**2 - radius**2)
            )
            self.rollForcingInterferenceFactor = (1 / np.pi**2) * (
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
            self.evaluateRollCoefficients()

    def evaluateCenterOfPressure(self):
        # Center of pressure position relative to CDM (center of dry mass)
        cpz = self.distanceToCM + np.sign(self.distanceToCM) * (0.288 * self.rootChord)
        self.cpx = 0
        self.cpy = 0
        self.cpz = cpz
        self.cp = (self.cpx, self.cpy, self.cpz)
        return self

    def draw(self):
        # Color cycle [#348ABD, #A60628, #7A68A6, #467821, #D55E00, #CC79A7, #56B4E9, #009E73, #F0E442, #0072B2]
        # Ellipse
        el = Ellipse(
            (self.rootChord / 2, 0),
            self.rootChord,
            self.span * 2,
            fill=False,
            edgecolor="#A60628",
            linewidth=2,
        )

        # Mean Aerodynamic Chord
        Yma_length = 8 * self.rootChord / (3 * np.pi)  # From barrowman
        Yma_start = (self.rootChord - Yma_length) / 2
        Yma_end = self.rootChord - (self.rootChord - Yma_length) / 2
        Yma_line = plt.Line2D(
            (Yma_start, Yma_end),
            (self.Yma, self.Yma),
            label="Mean Aerodynamic Chord",
            color="#467821",
        )

        # Center Line
        center_line = plt.Line2D(
            (self.rootChord / 2, self.rootChord / 2),
            (0, self.span),
            color="#7A68A6",
            alpha=0.35,
            linestyle="--",
            label="Center Line",
        )

        # Center of pressure
        cp_point = [abs(self.distanceToCM - self.cpz), self.Yma]

        # Plotting
        fig3 = plt.figure(figsize=(4, 4))
        with plt.style.context("bmh"):
            ax1 = fig3.add_subplot(111)
        ax1.add_patch(el)
        ax1.add_line(Yma_line)
        ax1.add_line(center_line)
        ax1.scatter(
            *cp_point, label="Center Of Pressure", color="red", s=100, zorder=10
        )
        ax1.scatter(*cp_point, facecolors="none", edgecolors="red", s=500, zorder=10)

        # Plot settings
        ax1.set_xlim(0, self.rootChord)
        ax1.set_ylim(0, self.span * 1.1)
        ax1.set_xlabel("Root Chord")
        ax1.set_ylabel("Span")
        ax1.set_title("Elliptical Fin")
        ax1.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")

        return self


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
        self.topRadius = topRadius
        self.bottomRadius = bottomRadius
        self.length = length
        self.distanceToCM = distanceToCM
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
