__author__ = "Guilherme Fernandes Alves, Mateus Stano Junqueira"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from .Function import Function
from abc import ABC, abstractmethod, abstractproperty


class AeroSurfaces:
    """Class used to hold multiple aerodynamic surfaces and their positions."""

    def __init__(self):
        self._aeroSurfaces = []

    def append(self, aeroSurface, position):
        self._aeroSurfaces.append((aeroSurface, position))

    def remove(self, aeroSurface):
        for surface, position in self._aeroSurfaces:
            if surface == aeroSurface:
                self._aeroSurfaces.remove((aeroSurface, position))

    def pop(self, index=-1):
        return self._aerosurfaces.pop(index)

    def __len__(self):
        return len(self._aeroSurfaces)

    def __getitem__(self, index):
        return self._aeroSurfaces[index]

    def __iter__(self):
        return iter(self._aeroSurfaces)


class NoseCone:
    """Keeps nose cone information.

    Local coordinate system: Z axis along the longitudinal axis of symmetry, positive
    downwards (top -> bottom). Origin located at the tip of the nose cone.

    Attributes
    ----------
    NoseCone.length : float
        Nose cone length. Has units of length and must be given in meters.
    NoseCone.rocketRadius : float
        The reference rocket radius used for lift coefficient normalization, in meters.
    NoseCone.kind : string
        Nose cone kind. Can be "conical", "ogive" or "lvhaack".
    NoseCone.name : string
        Nose cone name. Has no impact in simulation, as it is only used to
        display data in a more organized matter.
    NoseCone.cp : tuple
        Tuple with the x, y and z local coordinates of the nose cone center of pressure.
        Has units of length and is given in meters.
    NoseCone.cpx : float
        Nose cone local center of pressure x coordinate. Has units of length and is
        given in meters.
    NoseCone.cpy : float
        Nose cone local center of pressure y coordinate. Has units of length and is
        given in meters.
    NoseCone.cpz : float
        Nose cone local center of pressure z coordinate. Has units of length and is
        given in meters.
    NoseCone.cl : Function
        Function which defines the lift coefficient as a function of the angle of
        attack and the Mach number. Takes as input the angle of attack in radians and
        the Mach number. Returns the lift coefficient.
    NoseCone.clalpha : float
        Lift coefficient slope. Has units of 1/rad.
    """

    def __init__(
        self, length, kind, baseRadius=None, rocketRadius=None, name="Nose Cone"
    ):
        """Initializes the nose cone. It is used to define the nose cone
        length, kind, center of pressure and lift coefficient curve.

        Parameters
        ----------
        length : float
            Nose cone length. Has units of length and must be given in meters.
        kind : string
            Nose cone kind. Can be "conical", "ogive" or "lvhaack".
        baseRadius : float, optional
            Nose cone base radius. Has units of length and must be given in meters.
            If not given, the ratio between baseRadius and rocketRadius will be
            assumed as 1.
        rocketRadius : int, float, optional
            The reference rocket radius used for lift coefficient normalization.
            If not given, the ratio between baseRadius and rocketRadius will be
            assumed as 1.
        name : str, optional
            Nose cone name. Has no impact in simulation, as it is only used to
            display data in a more organized matter.

        Returns
        -------
        None
        """
        self.length = length
        self.kind = kind
        self.name = name
        self.baseRadius = baseRadius
        self.rocketRadius = rocketRadius

        if self.baseRadius is None or self.rocketRadius is None:
            self.radiusRatio = 1
        else:
            self.radiusRatio = self.baseRadius / self.rocketRadius

        # Analyze type
        if self.kind == "conical":
            self.k = 2 / 3
        elif self.kind == "ogive":
            self.k = 0.466
        elif self.kind == "lvhaack":
            self.k = 0.563
        else:
            self.k = 0.5

        # Calculate cp position in local coordinates
        # Local coordinate origin is found at the tip of the nose cone
        self.cpz = self.k * length
        self.cpy = 0
        self.cpx = 0
        self.cp = (self.cpx, self.cpy, self.cpz)

        # Calculate clalpha
        self.clalpha = 2 * self.radiusRatio**2
        self.cl = Function(
            lambda alpha, mach: self.clalpha * alpha,
            ["Alpha (rad)", "Mach"],
            "Cl",
        )
        # # Store values
        # nose = {"cp": (0, 0, self.cpz), "cl": self.cl, "name": name}

        return None

    def geometricInfo(self):
        """Prints out all the geometric information of the nose cone.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        print("Nose Cone Geometric Information of Nose: {}".format(self.name))
        print("-------------------------------")
        print(f"Length: {self.length:.3f} m")
        print(f"Kind: {self.kind}")
        print(f"Base Radius: {self.baseRadius:.3f} m")
        print(f"Reference Rocket Radius: {self.rocketRadius:.3f} m")
        print(f"Radius Ratio: {self.radiusRatio:.3f}")

        return None

    def aerodynamicInfo(self):
        """Prints out all the aerodynamic information of the nose cone.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        print(f"Nose Cone Aerodynamic Information of Nose: {self.name}")
        print("-------------------------------")
        print(f"Center of Pressure Position in Local Coordinates: {self.cp} m")
        print(f"Lift Coefficient Slope: {self.clalpha:.3f} 1/rad")
        print("Lift Coefficient as a Function of Alpha and Mach:")
        self.cl()

        return None

    def allInfo(self):
        """Prints out all the geometric and aerodynamic information of the nose cone.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.geometricInfo()
        self.aerodynamicInfo()

        return None


class Fins(ABC):
    """Abstract class that holds common methods for the fin classes.
    Cannot be instantiated.

    Local coordinate system: Z axis along the longitudinal axis of symmetry, positive
    downwards (top -> bottom). Origin located at the top of the root chord.

    Attributes
    ----------
    Fins.n : int
        Number of fins in fin set.
    Fins.rocketRadius : float
        The reference rocket radius used for lift coefficient normalization, in meters.
    Fins.airfoil : tuple
        Tuple of two items. First is the airfoil lift curve.
        Second is the unit of the curve (radians or degrees).
    Fins.cantAngle : float
        Fins cant angle with respect to the rocket centerline, in degrees.
    Fins.changingAttributeDict : dict
        Dictionary that stores the name and the values of the attributes that may
        be changed during a simulation. Useful for control systems.
    Fins.cantAngleRad : float
        Fins cant angle with respect to the rocket centerline, in radians.
    Fins.rootChord : float
        Fin root chord in meters.
    Fins.tipChord : float
        Fin tip chord in meters.
    Fins.span : float
        Fin span in meters.
    Fins.name : string
        Name of fin set.
    Fins.sweepLength : float
        Fins sweep length in meters. By sweep length, understand the axial distance
        between the fin root leading edge and the fin tip leading edge measured
        parallel to the rocket centerline.
    Fins.sweepAngle : float
        Fins sweep angle with respect to the rocket centerline. Must
        be given in degrees.
    Fins.d : float
        Reference diameter of the rocket. Has units of length and is given in meters.
    Fins.Aref : float
        Reference area of the rocket.
    Fins.Af : float
        Area of the longitudinal section of each fin in the set.
    Fins.AR : float
        Aspect ratio of each fin in the set.
    Fins.gamma_c : float
        Fin mid-chord sweep angle.
    Fins.Yma : float
        Span wise position of the mean aerodynamic chord.
    Fins.rollGeometricalConstant : float
        Geometrical constant used in roll calculations.
    Fins.tau : float
        Geometrical relation used to simplify lift and roll calculations.
    Fins.liftInterferenceFactor : float
        Factor of Fin-Body interference in the lift coefficient.
    Fins.cp : tuple
        Tuple with the x, y and z local coordinates of the fin set center of pressure.
        Has units of length and is given in meters.
    Fins.cpx : float
        Fin set local center of pressure x coordinate. Has units of length and is
        given in meters.
    Fins.cpy : float
        Fin set local center of pressure y coordinate. Has units of length and is
        given in meters.
    Fins.cpz : float
        Fin set local center of pressure z coordinate. Has units of length and is
        given in meters.
    Fins.cl : Function
        Function which defines the lift coefficient as a function of the angle of
        attack and the Mach number. Takes as input the angle of attack in radians and
        the Mach number. Returns the lift coefficient.
    Fins.clalpha : float
        Lift coefficient slope. Has units of 1/rad.
    """

    def __init__(
        self,
        n,
        rootChord,
        span,
        rocketRadius,
        cantAngle=0,
        airfoil=None,
        name="Fins",
    ):
        """Initialize Fins class.

        Parameters
        ----------
        n : int
            Number of fins, from 2 to infinity.
        rootChord : int, float
            Fin root chord in meters.
        span : int, float
            Fin span in meters.
        rocketRadius : int, float
            Reference rocket radius used for lift coefficient normalization.
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
        name : str
            Name of fin set.

        Returns
        -------
        None
        """

        # Compute auxiliary geometrical parameters
        d = 2 * rocketRadius
        Aref = np.pi * rocketRadius**2  # Reference area

        # Store values
        self.n = n
        self.rocketRadius = rocketRadius
        self.airfoil = airfoil
        self.cantAngle = cantAngle
        self.rootChord = rootChord
        self.span = span
        self.name = name
        self.d = d
        self.Aref = Aref  # Reference area

        return None

    @abstractmethod
    def evaluateCenterOfPressure(self):
        """Calculates and returns the fin set's center of pressure position in local
        coordinates. The center of pressure position is saved and stored as a tuple.

        Parameters
        ----------
        None

        Returns
        -------
        self.cp : tuple
            Tuple containing cpx, cpy, cpz.
        """

        pass

    def evaluateLiftCoefficient(self):
        """Calculates and returns the finset's lift coefficient.
        The lift coefficient is saved and returned. This function
        also calculates and saves the lift coefficient derivative
        for a single fin and the lift coefficient derivative for
        a number of n fins corrected for Fin-Body interference.

        Parameters
        ----------
        None

        Returns
        -------
        self.cl : Function
            Function of the angle of attack (Alpha) and the mach number
            (Mach) expressing the lift coefficient of the fin set. The inputs
            are the angle of attack (in radians) and the mach number.
            The output is the lift coefficient of the fin set.
        """
        if not self.airfoil:
            # Defines clalpha2D as 2*pi for planar fins
            clalpha2D = Function(lambda mach: 2 * np.pi / self.__beta(mach))
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
            clalpha2D = Function(lambda mach: clalpha2D_Mach0 / self.__beta(mach))

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
            / (2 + FD(mach) * np.sqrt(1 + (2 / FD(mach)) ** 2)),
            "Mach",
            "Lift coefficient derivative for a single fin",
        )

        # Lift coefficient derivative for a number of n fins corrected for Fin-Body interference
        self.clalphaMultipleFins = (
            self.liftInterferenceFactor
            * self.__finNumCorrection(self.n)
            * self.clalphaSingleFin
        )  # Function of mach number
        self.clalphaMultipleFins.setInputs("Mach")
        self.clalphaMultipleFins.setOutputs(
            "Lift coefficient derivative for {:.0f} fins".format(self.n)
        )
        self.clalpha = self.clalphaMultipleFins

        # Calculates clalpha * alpha
        self.cl = Function(
            lambda alpha, mach: alpha * self.clalphaMultipleFins(mach),
            ["Alpha (rad)", "Mach"],
            "Lift coefficient",
        )

        return self.cl

    def evaluateRollParameters(self):
        """Calculates and returns the finset's roll coefficients.
        The roll coefficients are saved in a list.

        Parameters
        ----------
        None

        Returns
        -------
        self.rollParameters : list
            List containing the roll moment lift coefficient, the
            roll moment damping coefficient and the cant angle in
            radians
        """

        self.cantAngleRad = np.radians(self.cantAngle)

        clfDelta = (
            self.rollForcingInterferenceFactor
            * self.n
            * (self.Yma + self.rocketRadius)
            * self.clalphaSingleFin
            / self.d
        )  # Function of mach number
        clfDelta.setInputs("Mach")
        clfDelta.setOutputs("Roll moment forcing coefficient derivative")
        cldOmega = (
            2
            * self.rollDampingInterferenceFactor
            * self.n
            * self.clalphaSingleFin
            * np.cos(self.cantAngleRad)
            * self.rollGeometricalConstant
            / (self.Aref * self.d**2)
        )  # Function of mach number
        cldOmega.setInputs("Mach")
        cldOmega.setOutputs("Roll moment damping coefficient derivative")
        self.rollParameters = [clfDelta, cldOmega, self.cantAngleRad]
        return self.rollParameters

    # Defines beta parameter
    def __beta(_, mach):
        """Defines a parameter that is often used in aerodynamic
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
    def __finNumCorrection(_, n):
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

    @abstractmethod
    def draw():
        """Draw the fin shape along with some important
        information. These being, the center line, the
        quarter line and the center of pressure position.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        pass

    def geometricalInfo(self):
        """Prints out information about geometrical parameters
        of the fin set.

        Parameters
        ----------
        None

        Return
        ------
        None
        """

        print("\n\nGeometrical Parameters\n")
        if isinstance(self, TrapezoidalFins):
            print("Fin Type: Trapezoidal")
            print("Tip Chord: {:.3f} m".format(self.tipChord))
        else:
            print("Fin Type: Elliptical")

        print("Root Chord: {:.3f} m".format(self.rootChord))
        print("Span: {:.3f} m".format(self.span))
        print("Cant Angle: {:.3f} °".format(self.cantAngle))
        print("Longitudinal Section Area: {:.3f} m".format(self.Af))
        print("Aspect Ratio: {:.3f} m".format(self.AR))
        print("Gamma_c: {:.3f} m".format(self.gamma_c))
        print("Mean Aerodynamic Chord: {:.3f} m".format(self.Yma))
        print(
            "Roll Geometrical Constant: {:.3f} m".format(self.rollGeometricalConstant)
        )

        return None

    def aerodynamicInfo(self):
        """Prints out information about lift parameters
        of the fin set.

        Parameters
        ----------
        None

        Return
        ------
        None
        """
        print("\n\nAerodynamic Information")
        print("----------------")
        print("Lift Interference Factor: {:.3f} m".format(self.liftInterferenceFactor))
        print(
            "Center of Pressure position in Local Coordinates: ({:.3f},{:.3f},{:.3f}) (x, y, z)".format(
                self.cpx, self.cpy, self.cpz
            )
        )
        print(
            "Lift Coefficient derivative as a Function of Alpha and Mach for Single Fin"
        )
        self.clalphaSingleFin()
        print(
            "Lift Coefficient derivative as a Function of Alpha and Mach for the Fin Set"
        )
        self.clalphaMultipleFins()
        print("Lift Coefficient as a Function of Alpha and Mach for the Fin Set")
        self.cl()

        return None

    def rollInfo(self):
        """Prints out information about roll parameters
        of the fin set.

        Parameters
        ----------
        None

        Return
        ------
        None
        """
        print("\nRoll Information")
        print("----------------")
        print(
            "Cant Angle: {:.3f} ° or {:.3f} rad".format(
                self.cantAngle, self.cantAngleRad
            )
        )
        print(
            "Roll Damping Interference Factor: {:.3f} rad".format(
                self.rollDampingInterferenceFactor
            )
        )
        print(
            "Roll Forcing Interference Factor: {:.3f} rad".format(
                self.rollForcingInterferenceFactor
            )
        )
        #lacks a title for the plot
        self.rollParameters[0]()
        #lacks a title for the plot
        self.rollParameters[1]()

        return None

    def airfoilPlots(self):
        
        if self.airfoil is not None:
            airfoilCl = Function(
                self.airfoil[0],
                inputs = "angle of attack",
                outputs = "lift coefficient",
                interpolation="linear",
                extrapolation = None,
            )
            airfoilCl.plot1D()

    def allInfo(self):
        """Prints out all data and graphs available about the Rocket.

        Parameters
        ----------
        None

        Return
        ------
        None
        """

        print("Fin set information\n")

        self.draw()

        print("Basic Information\n")

        print("Number of fins: {:.0f}".format(self.n))
        print("Reference rocket radius: {:.3f} m".format(self.rocketRadius))

        self.geometricalInfo()
        self.aerodynamicInfo()
        self.rollInfo()
        self.airfoilPlots()

        return None


class TrapezoidalFins(Fins):
    """Class that defines and holds information for a trapezoidal fin set.

    Attributes
    ----------

        Geometrical attributes:
        Fins.n : int
            Number of fins in fin set.
        Fins.rocketRadius : float
            The reference rocket radius used for lift coefficient normalization, in
            meters.
        Fins.airfoil : tuple
            Tuple of two items. First is the airfoil lift curve.
            Second is the unit of the curve (radians or degrees).
        Fins.cantAngle : float
            Fins cant angle with respect to the rocket centerline, in degrees.
        Fins.changingAttributeDict : dict
            Dictionary that stores the name and the values of the attributes that may
            be changed during a simulation. Useful for control systems.
        Fins.cantAngleRad : float
            Fins cant angle with respect to the rocket centerline, in radians.
        Fins.rootChord : float
            Fin root chord in meters.
        Fins.tipChord : float
            Fin tip chord in meters.
        Fins.span : float
            Fin span in meters.
        Fins.name : string
            Name of fin set.
        Fins.sweepLength : float
            Fins sweep length in meters. By sweep length, understand the axial distance
            between the fin root leading edge and the fin tip leading edge measured
            parallel to the rocket centerline.
        Fins.sweepAngle : float
            Fins sweep angle with respect to the rocket centerline. Must
            be given in degrees.
        Fins.d : float
            Reference diameter of the rocket, in meters.
        Fins.Aref : float
            Reference area of the rocket, in m².
        Fins.Af : float
            Area of the longitudinal section of each fin in the set.
        Fins.AR : float
            Aspect ratio of each fin in the set
        Fins.gamma_c : float
            Fin mid-chord sweep angle.
        Fins.Yma : float
            Span wise position of the mean aerodynamic chord.
        Fins.rollGeometricalConstant : float
            Geometrical constant used in roll calculations.
        Fins.tau : float
            Geometrical relation used to simplify lift and roll calculations.
        Fins.liftInterferenceFactor : float
            Factor of Fin-Body interference in the lift coefficient.
        Fins.cp : tuple
            Tuple with the x, y and z local coordinates of the fin set center of
            pressure. Has units of length and is given in meters.
        Fins.cpx : float
            Fin set local center of pressure x coordinate. Has units of length and is
            given in meters.
        Fins.cpy : float
            Fin set local center of pressure y coordinate. Has units of length and is
            given in meters.
        Fins.cpz : float
            Fin set local center of pressure z coordinate. Has units of length and is
            given in meters.
        Fins.cl : Function
            Function which defines the lift coefficient as a function of the angle of
            attack and the Mach number. Takes as input the angle of attack in radians
            and the Mach number. Returns the lift coefficient.
        Fins.clalpha : float
            Lift coefficient slope. Has units of 1/rad.
    """

    def __init__(
        self,
        n,
        rootChord,
        tipChord,
        span,
        rocketRadius,
        cantAngle=0,
        sweepLength=None,
        sweepAngle=None,
        airfoil=None,
        name="Fins",
    ):
        """Initialize TrapezoidalFins class.

        Parameters
        ----------
        n : int
            Number of fins, from 2 to infinity.
        rootChord : int, float
            Fin root chord in meters.
        tipChord : int, float
            Fin tip chord in meters.
        span : int, float
            Fin span in meters.
        rocketRadius : int, float
            Reference radius to calculate lift coefficient, in meters.
        cantAngle : int, float, optional
            Fins cant angle with respect to the rocket centerline. Must
            be given in degrees.
        sweepLength : int, float, optional
            Fins sweep length in meters. By sweep length, understand the axial distance
            between the fin root leading edge and the fin tip leading edge measured
            parallel to the rocket centerline. If not given, the sweep length is
            assumed to be equal the root chord minus the tip chord, in which case the
            fin is a right trapezoid with its base perpendicular to the rocket's axis.
            Cannot be used in conjunction with sweepAngle.
        sweepAngle : int, float, optional
            Fins sweep angle with respect to the rocket centerline. Must
            be given in degrees. If not given, the sweep angle is automatically
            calculated, in which case the fin is assumed to be a right trapezoid with
            its base perpendicular to the rocket's axis.
            Cannot be used in conjunction with sweepLength.
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
        name : str
            Name of fin set.

        Returns
        -------
        None
        """

        super().__init__(
            n,
            rootChord,
            span,
            rocketRadius,
            cantAngle,
            airfoil,
            name,
        )

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

        Yr = rootChord + tipChord
        Af = Yr * span / 2  # Fin area
        AR = 2 * span**2 / Af  # Fin aspect ratio
        gamma_c = np.arctan((sweepLength + 0.5 * tipChord - 0.5 * rootChord) / (span))
        Yma = (
            (span / 3) * (rootChord + 2 * tipChord) / Yr
        )  # Span wise coord of mean aero chord

        # Fin–body interference correction parameters
        tau = (span + rocketRadius) / rocketRadius
        liftInterferenceFactor = 1 + 1 / tau
        λ = tipChord / rootChord

        # Parameters for Roll Moment.
        # Documented at: https://github.com/RocketPy-Team/RocketPy/blob/master/docs/technical/aerodynamics/Roll_Equations.pdf
        rollGeometricalConstant = (
            (rootChord + 3 * tipChord) * span**3
            + 4 * (rootChord + 2 * tipChord) * rocketRadius * span**2
            + 6 * (rootChord + tipChord) * span * rocketRadius**2
        ) / 12
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

        self.tipChord = tipChord
        self.sweepLength = sweepLength
        self.sweepAngle = sweepAngle
        self.Yr = Yr
        self.Af = Af  # Fin area
        self.AR = AR  # Aspect Ratio
        self.gamma_c = gamma_c  # Mid chord angle
        self.Yma = Yma  # Span wise coord of mean aero chord
        self.rollGeometricalConstant = rollGeometricalConstant
        self.tau = tau
        self.liftInterferenceFactor = liftInterferenceFactor
        self.λ = λ
        self.rollDampingInterferenceFactor = rollDampingInterferenceFactor
        self.rollForcingInterferenceFactor = rollForcingInterferenceFactor

        self.evaluateCenterOfPressure()
        self.evaluateLiftCoefficient()
        self.evaluateRollParameters()

    def evaluateCenterOfPressure(self):
        """Calculates and returns the center of pressure of the fin set in local
        coordinates. The center of pressure position is saved and stored as a tuple.

        Parameters
        ----------
        None

        Returns
        -------
        self.cp : tuple
            Tuple containing cpx, cpy, cpz.
        """
        # Center of pressure position in local coordinates
        cpz = (self.sweepLength / 3) * (
            (self.rootChord + 2 * self.tipChord) / (self.rootChord + self.tipChord)
        ) + (1 / 6) * (
            self.rootChord
            + self.tipChord
            - self.rootChord * self.tipChord / (self.rootChord + self.tipChord)
        )
        self.cpx = 0
        self.cpy = 0
        self.cpz = cpz
        self.cp = (self.cpx, self.cpy, self.cpz)
        return self.cp

    def draw(self):
        """Draw the fin shape along with some important
        information. These being, the center line, the
        quarter line and the center of pressure position.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
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
        cp_point = [self.cpz, self.Yma]

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
        fig3 = plt.figure(figsize=(7, 4))
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

        plt.tight_layout()
        plt.show()

        return None


class EllipticalFins(Fins):
    """Class that defines and holds information for an elliptical fin set.

    Attributes
    ----------

        Geometrical attributes:
        Fins.n : int
            Number of fins in fin set.
        Fins.rocketRadius : float
            The reference rocket radius used for lift coefficient normalization, in
            meters.
        Fins.airfoil : tuple
            Tuple of two items. First is the airfoil lift curve.
            Second is the unit of the curve (radians or degrees)
        Fins.cantAngle : float
            Fins cant angle with respect to the rocket centerline, in degrees.
        Fins.changingAttributeDict : dict
            Dictionary that stores the name and the values of the attributes that may
            be changed during a simulation. Useful for control systems.
        Fins.cantAngleRad : float
            Fins cant angle with respect to the rocket centerline, in radians.
        Fins.rootChord : float
            Fin root chord in meters.
        Fins.span : float
            Fin span in meters.
        Fins.name : string
            Name of fin set.
        Fins.sweepLength : float
            Fins sweep length in meters. By sweep length, understand the axial distance
            between the fin root leading edge and the fin tip leading edge measured
            parallel to the rocket centerline.
        Fins.sweepAngle : float
            Fins sweep angle with respect to the rocket centerline. Must
            be given in degrees.
        Fins.d : float
            Reference diameter of the rocket, in meters.
        Fins.Aref : float
            Reference area of the rocket.
        Fins.Af : float
            Area of the longtudinal section of each fin in the set.
        Fins.AR : float
            Aspect ratio of each fin in the set.
        Fins.gamma_c : float
            Fin mid-chord sweep angle.
        Fins.Yma : float
            Span wise position of the mean aerodynamic chord.
        Fins.rollGeometricalConstant : float
            Geometrical constant used in roll calculations.
        Fins.tau : float
            Geometrical relation used to simplify lift and roll calculations.
        Fins.liftInterferenceFactor : float
            Factor of Fin-Body interference in the lift coefficient.
        Fins.cp : tuple
            Tuple with the x, y and z local coordinates of the fin set center of
            pressure. Has units of length and is given in meters.
        Fins.cpx : float
            Fin set local center of pressure x coordinate. Has units of length and is
            given in meters.
        Fins.cpy : float
            Fin set local center of pressure y coordinate. Has units of length and is
            given in meters.
        Fins.cpz : float
            Fin set local center of pressure z coordinate. Has units of length and is
            given in meters.
        Fins.cl : Function
            Function which defines the lift coefficient as a function of the angle of
            attack and the Mach number. Takes as input the angle of attack in radians
            and the Mach number. Returns the lift coefficient.
        Fins.clalpha : float
            Lift coefficient slope. Has units of 1/rad.
    """

    def __init__(
        self,
        n,
        rootChord,
        span,
        rocketRadius,
        cantAngle=0,
        airfoil=None,
        name="Fins",
    ):
        """Initialize EllipticalFins class.

        Parameters
        ----------
        n : int
            Number of fins, from 2 to infinity.
        rootChord : int, float
            Fin root chord in meters.
        span : int, float
            Fin span in meters.
        rocketRadius : int, float
            Reference radius to calculate lift coefficient, in meters.
        cantAngle : int, float, optional
            Fins cant angle with respect to the rocket centerline. Must
            be given in degrees.
        sweepLength : int, float, optional
            Fins sweep length in meters. By sweep length, understand the axial distance
            between the fin root leading edge and the fin tip leading edge measured
            parallel to the rocket centerline. If not given, the sweep length is
            assumed to be equal the root chord minus the tip chord, in which case the
            fin is a right trapezoid with its base perpendicular to the rocket's axis.
            Cannot be used in conjunction with sweepAngle.
        sweepAngle : int, float, optional
            Fins sweep angle with respect to the rocket centerline. Must
            be given in degrees. If not given, the sweep angle is automatically
            calculated, in which case the fin is assumed to be a right trapezoid with
            its base perpendicular to the rocket's axis.
            Cannot be used in conjunction with sweepLength.
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
        name : str
            Name of fin set.

        Returns
        -------
        None
        """

        super().__init__(
            n,
            rootChord,
            span,
            rocketRadius,
            cantAngle,
            airfoil,
            name,
        )

        # Compute auxiliary geometrical parameters
        Af = (np.pi * rootChord / 2 * span) / 2  # Fin area
        gamma_c = 0  # Zero for elliptical fins
        AR = 2 * span**2 / Af  # Fin aspect ratio
        Yma = (
            span / (3 * np.pi) * np.sqrt(9 * np.pi**2 - 64)
        )  # Span wise coord of mean aero chord
        rollGeometricalConstant = (
            rootChord
            * span
            * (
                3 * np.pi * span**2
                + 32 * rocketRadius * span
                + 12 * np.pi * rocketRadius**2
            )
            / 48
        )

        # Fin–body interference correction parameters
        tau = (span + rocketRadius) / rocketRadius
        liftInterferenceFactor = 1 + 1 / tau
        rollDampingInterferenceFactor = 1 + (
            (rocketRadius**2)
            * (
                2
                * (rocketRadius**2)
                * np.sqrt(span**2 - rocketRadius**2)
                * np.log(
                    (2 * span * np.sqrt(span**2 - rocketRadius**2) + 2 * span**2)
                    / rocketRadius
                )
                - 2
                * (rocketRadius**2)
                * np.sqrt(span**2 - rocketRadius**2)
                * np.log(2 * span)
                + 2 * span**3
                - np.pi * rocketRadius * span**2
                - 2 * (rocketRadius**2) * span
                + np.pi * rocketRadius**3
            )
        ) / (
            2
            * (span**2)
            * (span / 3 + np.pi * rocketRadius / 4)
            * (span**2 - rocketRadius**2)
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

        # Store values
        self.Af = Af  # Fin area
        self.AR = AR  # Fin aspect ratio
        self.gamma_c = gamma_c  # Mid chord angle
        self.Yma = Yma  # Span wise coord of mean aero chord
        self.rollGeometricalConstant = rollGeometricalConstant
        self.tau = tau
        self.liftInterferenceFactor = liftInterferenceFactor
        self.rollDampingInterferenceFactor = rollDampingInterferenceFactor
        self.rollForcingInterferenceFactor = rollForcingInterferenceFactor

        self.evaluateCenterOfPressure()
        self.evaluateLiftCoefficient()
        self.evaluateRollParameters()

        return None

    def evaluateCenterOfPressure(self):
        """Calculates and returns the center of pressure of the fin set in local
        coordinates. The center of pressure position is saved and stored as a tuple.

        Parameters
        ----------
        None

        Returns
        -------
        self.cp : tuple
            Tuple containing cpx, cpy, cpz.
        """
        # Center of pressure position in local coordinates
        cpz = 0.288 * self.rootChord
        self.cpx = 0
        self.cpy = 0
        self.cpz = cpz
        self.cp = (self.cpx, self.cpy, self.cpz)
        return self

    def draw(self):
        """Draw the fin shape along with some important information.
        These being, the center line and the center of pressure position.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
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
        Yma_length = 8 * self.rootChord / (3 * np.pi)  # From Barrowman's theory
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
        cp_point = [self.cpz, self.Yma]

        # Plotting
        fig3 = plt.figure(figsize=(7, 4))
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

        plt.tight_layout()
        plt.show()

        return None


class Tail:
    """Class that defines a tail. Currently only accepts conical tails.

    Local coordinate system: Z axis along the longitudinal axis of symmetry, positive
    downwards (top -> bottom). Origin located at top of the tail (generally the portion
    closest to the rocket's nose).

    Parameters
    ----------
    Tail.topRadius : int, float
        Radius of the top of the tail. The top radius is defined as the radius
        of the transversal section that is closest to the rocket's nose.
    Tail.bottomRadius : int, float
        Radius of the bottom of the tail.
    Tail.length : int, float
        Length of the tail. The length is defined as the distance between the
        top and bottom of the tail. The length is measured along the rocket's
        longitudinal axis. Has the unit of meters.
    Tail.rocketRadius: int, float
        The reference rocket radius used for lift coefficient normalization in meters.
    Tail.name : str
        Name of the tail. Default is 'Tail'.

    Attributes
    ----------
    Tail.cpx : int, float
        x local coordinate of the center of pressure of the tail.
    Tail.cpy : int, float
        y local coordinate of the center of pressure of the tail.
    Tail.cpz : int, float
        z local coordinate of the center of pressure of the tail.
    Tail.cp : tuple
        Tuple containing the coordinates of the center of pressure of the tail.
    Tail.cl : Function
        Function that returns the lift coefficient of the tail. The function
        is defined as a function of the angle of attack and the mach number.
    Tail.clalpha : float
        Lift coefficient slope. Has the unit of 1/rad.
    Tail.slantLength : float
        Slant length of the tail. The slant length is defined as the distance
        between the top and bottom of the tail. The slant length is measured
        along the tail's slant axis. Has the unit of meters.
    Tail.surfaceArea : float
        Surface area of the tail. Has the unit of meters squared.

    """

    def __init__(self, topRadius, bottomRadius, length, rocketRadius, name="Tail"):
        """Initializes the tail object by computing and storing the most
        important values.

        Parameters
        ----------
        topRadius : int, float
            Radius of the top of the tail. The top radius is defined as the radius
            of the transversal section that is closest to the rocket's nose.
        bottomRadius : int, float
            Radius of the bottom of the tail.
        length : int, float
            Length of the tail.
        rocketRadius : int, float
            The reference rocket radius used for lift coefficient normalization.
        name : str
            Name of the tail. Default is 'Tail'.

        Returns
        -------
        None
        """

        # Store arguments as attributes
        self.topRadius = topRadius
        self.bottomRadius = bottomRadius
        self.length = length
        self.name = name
        self.rocketRadius = rocketRadius

        # Calculate ratio between top and bottom radius

        # Calculate tail slant length
        self.slantLength = np.sqrt(
            (self.length) ** 2 + (self.topRadius - self.bottomRadius) ** 2
        )
        # Calculate the surface area of the tail
        self.surfaceArea = (
            np.pi * self.slantLength * (self.topRadius + self.bottomRadius)
        )

        # Calculate cp position in local coordinates
        r = topRadius / bottomRadius
        cpz = (length / 3) * (1 + (1 - r) / (1 - r**2))

        # Calculate clalpha
        clalpha = 2 * (
            (bottomRadius / rocketRadius) ** 2 - (topRadius / rocketRadius) ** 2
        )
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

    def geometricInfo(self):
        """Prints out all the geometric information of the tail.

        Returns
        -------
        None
        """

        print(f"\nTail name: {self.name}")
        print(f"Tail Top Radius: {self.topRadius:.3f} m")
        print(f"Tail Bottom Radius: {self.bottomRadius:.3f} m")
        print(f"Tail Length: {self.length:.3f} m")
        print(f"Reference Radius: {2*self.rocketRadius:.3f} m")
        print(f"Tail Slant Length: {self.slantLength:.3f} m")
        print(f"Tail Surface Area: {self.surfaceArea:.6f} m^2")

        return None

    def aerodynamicInfo(self):

        print(f"\nTail name: {self.name}")
        print(f"Tail Center of Pressure Position in Local Coordinates: {self.cp} m")
        print(f"Tail Lift Coefficient Slope: {self.clalpha:.3f} 1/rad")
        print("Tail Lift Coefficient as a function of Alpha and Mach:")
        self.cl()

        return None

    def allInfo(self):
        """Prints all the information about the tail object.

        Returns
        -------
        None
        """
        self.geometricInfo()
        self.aerodynamicInfo()

        return None
