__author__ = "Guilherme Fernandes Alves, Mateus Stano Junqueira"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

from abc import ABC, abstractmethod

import numpy as np

from .Function import Function
from .plots.aero_surface_plots import (
    _EllipticalFinsPlots,
    _NoseConePlots,
    _TailPlots,
    _TrapezoidalFinsPlots,
)
from .prints.aero_surface_prints import (
    _EllipticalFinsPrints,
    _NoseConePrints,
    _RailButtonsPrints,
    _TailPrints,
    _TrapezoidalFinsPrints,
)


class AeroSurface(ABC):
    """Abstract class used to define aerodynamic surfaces."""

    def __init__(self, name):
        self.cpx = 0
        self.cpy = 0
        self.cpz = 0
        self.name = name
        return None

    @abstractmethod
    def evaluateCenterOfPressure(self):
        """Evaluates the center of pressure of the aerodynamic surface in local
        coordinates.

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def evaluateLiftCoefficient(self):
        """Evaluates the lift coefficient curve of the aerodynamic surface.

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def evaluateGeometricalParameters(self):
        """Evaluates the geometrical parameters of the aerodynamic surface.

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def info(self):
        """Prints and plots summarized information of the aerodynamic surface.

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def allInfo(self):
        """Prints and plots all the available information of the aero surface.

        Returns
        -------
        None
        """
        pass


class NoseCone(AeroSurface):
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
    NoseCone.plots : rocketpy.plots._NoseConePlots
        This contains all the plots methods. Use help(NoseCone.plots) to know
        more about it.
    NoseCone.prints : rocketpy.prints._NoseConePrints
        This contains all the prints methods. Use help(NoseCone.prints) to know
        more about it.
    """

    def __init__(
        self,
        length,
        kind,
        baseRadius=None,
        rocketRadius=None,
        name="Nose Cone",
    ):
        """Initializes the nose cone. It is used to define the nose cone
        length, kind, center of pressure and lift coefficient curve.

        Parameters
        ----------
        length : float
            Nose cone length. Has units of length and must be given in meters.
        kind : string
            Nose cone kind. Can be "conical", "ogive", "elliptical", "tangent",
            "von karman", "parabolic" or "lvhaack".
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
        super().__init__(name)

        self._rocketRadius = rocketRadius
        self._baseRadius = baseRadius
        self._length = length
        self.kind = kind

        self.evaluateGeometricalParameters()
        self.evaluateLiftCoefficient()
        self.evaluateCenterOfPressure()

        self.plots = _NoseConePlots(self)
        self.prints = _NoseConePrints(self)

        return None

    @property
    def rocketRadius(self):
        return self._rocketRadius

    @rocketRadius.setter
    def rocketRadius(self, value):
        self._rocketRadius = value
        self.evaluateGeometricalParameters()
        self.evaluateLiftCoefficient()

    @property
    def baseRadius(self):
        return self._baseRadius

    @baseRadius.setter
    def baseRadius(self, value):
        self._baseRadius = value
        self.evaluateGeometricalParameters()
        self.evaluateLiftCoefficient()

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        self._length = value
        self.evaluateCenterOfPressure()

    @property
    def kind(self):
        return self._kind

    @kind.setter
    def kind(self, value):
        # Analyze type
        self._kind = value
        if value == "conical":
            self.k = 2 / 3
        elif value == "ogive":
            self.k = 0.466
        elif value == "lvhaack":
            self.k = 0.563
        elif value == "tangent":
            rho = (self.baseRadius**2 + self.length**2) / (2 * self.baseRadius)
            volume = np.pi * (
                self.length * rho**2
                - (self.length**3) / 3
                - (rho - self.baseRadius) * rho**2 * np.arcsin(self.length / rho)
            )
            area = np.pi * self.baseRadius**2
            self.k = 1 - volume / (area * self.length)
        elif value == "elliptical":
            self.k = 1 / 3
        else:
            self.k = 0.5  # Parabolic and Von Karman
        self.evaluateCenterOfPressure()

    def evaluateGeometricalParameters(self):
        """Calculates and saves nose cone's radius ratio.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.baseRadius is None or self.rocketRadius is None:
            self.radiusRatio = 1
        else:
            self.radiusRatio = self.baseRadius / self.rocketRadius

    def evaluateLiftCoefficient(self):
        """Calculates and returns nose cone's lift coefficient.
        The lift coefficient is saved and returned. This function
        also calculates and saves its lift coefficient derivative.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Calculate clalpha
        # clalpha is currently a constant, meaning it is independent of Mach
        # number. This is only valid for subsonic speeds.
        # It must be set as a Function because it will be called and treated
        # as a function of mach in the simulation.
        self.clalpha = Function(
            lambda mach: 2 * self.radiusRatio**2,
            "Mach",
            f"Lift coefficient derivative for {self.name}",
        )
        self.cl = Function(
            lambda alpha, mach: self.clalpha(mach) * alpha,
            ["Alpha (rad)", "Mach"],
            "Cl",
        )
        return None

    def evaluateCenterOfPressure(self):
        """Calculates and returns the center of pressure of the nose cone in local
        coordinates. The center of pressure position is saved and stored as a tuple.
        Local coordinate origin is found at the tip of the nose cone.
        Parameters
        ----------
        None

        Returns
        -------
        self.cp : tuple
            Tuple containing cpx, cpy, cpz.
        """

        self.cpz = self.k * self.length
        self.cpy = 0
        self.cpx = 0
        self.cp = (self.cpx, self.cpy, self.cpz)
        return self.cp

    def info(self):
        """Prints and plots summarized information of the nose cone.

        Parameters
        ----------
        None

        Return
        ------
        None
        """
        self.prints.geometry()
        self.prints.lift()
        return None

    def allInfo(self):
        """Prints and plots all the available information of the nose cone.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.prints.all()
        self.plots.all()
        return None


class Fins(AeroSurface):
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
    Fins.rollParameters : list
        List containing the roll moment lift coefficient, the roll moment damping
        coefficient and the cant angle in radians.
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

        super().__init__(name)

        # Compute auxiliary geometrical parameters
        d = 2 * rocketRadius
        Aref = np.pi * rocketRadius**2  # Reference area

        # Store values
        self._n = n
        self._rocketRadius = rocketRadius
        self._airfoil = airfoil
        self._cantAngle = cantAngle
        self._rootChord = rootChord
        self._span = span
        self.name = name
        self.d = d
        self.Aref = Aref  # Reference area

        return None

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, value):
        self._n = value
        self.evaluateGeometricalParameters()
        self.evaluateCenterOfPressure()
        self.evaluateLiftCoefficient()
        self.evaluateRollParameters()

    @property
    def rootChord(self):
        return self._rootChord

    @rootChord.setter
    def rootChord(self, value):
        self._rootChord = value
        self.evaluateGeometricalParameters()
        self.evaluateCenterOfPressure()
        self.evaluateLiftCoefficient()
        self.evaluateRollParameters()

    @property
    def span(self):
        return self._span

    @span.setter
    def span(self, value):
        self._span = value
        self.evaluateGeometricalParameters()
        self.evaluateCenterOfPressure()
        self.evaluateLiftCoefficient()
        self.evaluateRollParameters()

    @property
    def rocketRadius(self):
        return self._rocketRadius

    @rocketRadius.setter
    def rocketRadius(self, value):
        self._rocketRadius = value
        self.evaluateGeometricalParameters()
        self.evaluateCenterOfPressure()
        self.evaluateLiftCoefficient()
        self.evaluateRollParameters()

    @property
    def cantAngle(self):
        return self._cantAngle

    @cantAngle.setter
    def cantAngle(self, value):
        self._cantAngle = value
        self.evaluateGeometricalParameters()
        self.evaluateCenterOfPressure()
        self.evaluateLiftCoefficient()
        self.evaluateRollParameters()

    @property
    def airfoil(self):
        return self._airfoil

    @airfoil.setter
    def airfoil(self, value):
        self._airfoil = value
        self.evaluateGeometricalParameters()
        self.evaluateCenterOfPressure()
        self.evaluateLiftCoefficient()
        self.evaluateRollParameters()

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
        None
        """
        if not self.airfoil:
            # Defines clalpha2D as 2*pi for planar fins
            clalpha2D = Function(lambda mach: 2 * np.pi / self.__beta(mach))
        else:
            # Defines clalpha2D as the derivative of the
            # lift coefficient curve for a specific airfoil
            self.airfoilCl = Function(
                self.airfoil[0],
                interpolation="linear",
            )

            # Differentiating at x = 0 to get cl_alpha
            clalpha2D_Mach0 = self.airfoilCl.differentiate(x=1e-3, dx=1e-3)

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

        self._tipChord = tipChord
        self._sweepLength = sweepLength
        self._sweepAngle = sweepAngle

        self.evaluateGeometricalParameters()
        self.evaluateCenterOfPressure()
        self.evaluateLiftCoefficient()
        self.evaluateRollParameters()

        self.prints = _TrapezoidalFinsPrints(self)
        self.plots = _TrapezoidalFinsPlots(self)

    @property
    def tipChord(self):
        return self._tipChord

    @tipChord.setter
    def tipChord(self, value):
        self._tipChord = value
        self.evaluateGeometricalParameters()
        self.evaluateCenterOfPressure()
        self.evaluateLiftCoefficient()
        self.evaluateRollParameters()

    @property
    def sweepAngle(self):
        return self._sweepAngle

    @sweepAngle.setter
    def sweepAngle(self, value):
        self._sweepAngle = value
        self._sweepLength = np.tan(value * np.pi / 180) * self.span
        self.evaluateGeometricalParameters()
        self.evaluateCenterOfPressure()
        self.evaluateLiftCoefficient()
        self.evaluateRollParameters()

    @property
    def sweepLength(self):
        return self._sweepLength

    @sweepLength.setter
    def sweepLength(self, value):
        self._sweepLength = value
        self.evaluateGeometricalParameters()
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
        None
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
        return None

    def evaluateGeometricalParameters(self):
        """Calculates and saves fin set's geometrical parameters such as the
        fins' area, aspect ratio and parameters for roll movement.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        Yr = self.rootChord + self.tipChord
        Af = Yr * self.span / 2  # Fin area
        AR = 2 * self.span**2 / Af  # Fin aspect ratio
        gamma_c = np.arctan(
            (self.sweepLength + 0.5 * self.tipChord - 0.5 * self.rootChord)
            / (self.span)
        )
        Yma = (
            (self.span / 3) * (self.rootChord + 2 * self.tipChord) / Yr
        )  # Span wise coord of mean aero chord

        # Fin–body interference correction parameters
        tau = (self.span + self.rocketRadius) / self.rocketRadius
        liftInterferenceFactor = 1 + 1 / tau
        λ = self.tipChord / self.rootChord

        # Parameters for Roll Moment.
        # Documented at: https://github.com/RocketPy-Team/RocketPy/blob/master/docs/technical/aerodynamics/Roll_Equations.pdf
        rollGeometricalConstant = (
            (self.rootChord + 3 * self.tipChord) * self.span**3
            + 4
            * (self.rootChord + 2 * self.tipChord)
            * self.rocketRadius
            * self.span**2
            + 6 * (self.rootChord + self.tipChord) * self.span * self.rocketRadius**2
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

        # Store values
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

    def info(self):
        self.prints.geometry()
        self.prints.lift()
        return None

    def allInfo(self):
        self.prints.all()
        self.plots.all()
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

        self.evaluateGeometricalParameters()
        self.evaluateCenterOfPressure()
        self.evaluateLiftCoefficient()
        self.evaluateRollParameters()

        self.prints = _EllipticalFinsPrints(self)
        self.plots = _EllipticalFinsPlots(self)

        return None

    def evaluateCenterOfPressure(self):
        """Calculates and returns the center of pressure of the fin set in local
        coordinates. The center of pressure position is saved and stored as a tuple.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Center of pressure position in local coordinates
        cpz = 0.288 * self.rootChord
        self.cpx = 0
        self.cpy = 0
        self.cpz = cpz
        self.cp = (self.cpx, self.cpy, self.cpz)
        return None

    def evaluateGeometricalParameters(self):
        """Calculates and saves fin set's geometrical parameters such as the
        fins' area, aspect ratio and parameters for roll movement.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        # Compute auxiliary geometrical parameters
        Af = (np.pi * self.rootChord / 2 * self.span) / 2  # Fin area
        gamma_c = 0  # Zero for elliptical fins
        AR = 2 * self.span**2 / Af  # Fin aspect ratio
        Yma = (
            self.span / (3 * np.pi) * np.sqrt(9 * np.pi**2 - 64)
        )  # Span wise coord of mean aero chord
        rollGeometricalConstant = (
            self.rootChord
            * self.span
            * (
                3 * np.pi * self.span**2
                + 32 * self.rocketRadius * self.span
                + 12 * np.pi * self.rocketRadius**2
            )
            / 48
        )

        # Fin–body interference correction parameters
        tau = (self.span + self.rocketRadius) / self.rocketRadius
        liftInterferenceFactor = 1 + 1 / tau
        rollDampingInterferenceFactor = 1 + (
            (self.rocketRadius**2)
            * (
                2
                * (self.rocketRadius**2)
                * np.sqrt(self.span**2 - self.rocketRadius**2)
                * np.log(
                    (
                        2 * self.span * np.sqrt(self.span**2 - self.rocketRadius**2)
                        + 2 * self.span**2
                    )
                    / self.rocketRadius
                )
                - 2
                * (self.rocketRadius**2)
                * np.sqrt(self.span**2 - self.rocketRadius**2)
                * np.log(2 * self.span)
                + 2 * self.span**3
                - np.pi * self.rocketRadius * self.span**2
                - 2 * (self.rocketRadius**2) * self.span
                + np.pi * self.rocketRadius**3
            )
        ) / (
            2
            * (self.span**2)
            * (self.span / 3 + np.pi * self.rocketRadius / 4)
            * (self.span**2 - self.rocketRadius**2)
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

    def info(self):
        self.prints.geometry()
        self.prints.lift()
        return None

    def allInfo(self):
        self.prints.all()
        self.plots.all()
        return None


class Tail(AeroSurface):
    """Class that defines a tail. Currently only accepts conical tails.

    Local coordinate system: Z axis along the longitudinal axis of symmetry, positive
    downwards (top -> bottom). Origin located at top of the tail (generally the portion
    closest to the rocket's nose).

    Attributes
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
        super().__init__(name)

        # Store arguments as attributes
        self._topRadius = topRadius
        self._bottomRadius = bottomRadius
        self._length = length
        self._rocketRadius = rocketRadius

        # Calculate geometrical parameters
        self.evaluateGeometricalParameters()
        self.evaluateLiftCoefficient()
        self.evaluateCenterOfPressure()

        self.plots = _TailPlots(self)
        self.prints = _TailPrints(self)

        return None

    @property
    def topRadius(self):
        return self._topRadius

    @topRadius.setter
    def topRadius(self, value):
        self._topRadius = value
        self.evaluateGeometricalParameters()
        self.evaluateLiftCoefficient()
        self.evaluateCenterOfPressure()

    @property
    def bottomRadius(self):
        return self._bottomRadius

    @bottomRadius.setter
    def bottomRadius(self, value):
        self._bottomRadius = value
        self.evaluateGeometricalParameters()
        self.evaluateLiftCoefficient()
        self.evaluateCenterOfPressure()

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        self._length = value
        self.evaluateGeometricalParameters()
        self.evaluateCenterOfPressure()

    @property
    def rocketRadius(self):
        return self._rocketRadius

    @rocketRadius.setter
    def rocketRadius(self, value):
        self._rocketRadius = value
        self.evaluateLiftCoefficient()

    def evaluateGeometricalParameters(self):
        """Calculates and saves tail's slant length and surface area.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Calculate tail slant length
        self.slantLength = np.sqrt(
            (self.length) ** 2 + (self.topRadius - self.bottomRadius) ** 2
        )
        # Calculate the surface area of the tail
        self.surfaceArea = (
            np.pi * self.slantLength * (self.topRadius + self.bottomRadius)
        )

    def evaluateLiftCoefficient(self):
        """Calculates and returns tail's lift coefficient.
        The lift coefficient is saved and returned. This function
        also calculates and saves its lift coefficient derivative.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Calculate clalpha
        # clalpha is currently a constant, meaning it is independent of Mach
        # number. This is only valid for subsonic speeds.
        # It must be set as a Function because it will be called and treated
        # as a function of mach in the simulation.
        self.clalpha = Function(
            lambda mach: 2
            * (
                (self.bottomRadius / self.rocketRadius) ** 2
                - (self.topRadius / self.rocketRadius) ** 2
            ),
            "Mach",
            f"Lift coefficient derivative for {self.name}",
        )
        self.cl = Function(
            lambda alpha, mach: self.clalpha(mach) * alpha,
            ["Alpha (rad)", "Mach"],
            "Cl",
        )
        return None

    def evaluateCenterOfPressure(self):
        """Calculates and returns the center of pressure of the tail in local
        coordinates. The center of pressure position is saved and stored as a tuple.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Calculate cp position in local coordinates
        r = self.topRadius / self.bottomRadius
        cpz = (self.length / 3) * (1 + (1 - r) / (1 - r**2))

        # Store values as class attributes
        self.cpx = 0
        self.cpy = 0
        self.cpz = cpz
        self.cp = (self.cpx, self.cpy, self.cpz)
        return None

    def info(self):
        self.prints.geometry()
        self.prints.lift()
        return None

    def allInfo(self):
        self.prints.all()
        self.plots.all()
        return None


class RailButtons(AeroSurface):
    """Class that defines a rail button pair or group.

    Attributes
    ----------
    RailButtons.buttons_distance : int, float
        Distance between the two rail buttons closest to the nozzle.
    RailButtons.angular_position : int, float
        Angular position of the rail buttons in degrees measured
        as the rotation around the symmetry axis of the rocket
        relative to one of the other principal axis.
    """

    def __init__(self, buttons_distance, angular_position=45, name="Rail Buttons"):
        """Initializes RailButtons Class.

        Parameters
        ----------
        buttons_distance : int, float
            Distance between the first and the last rail button in meters.
        angular_position : int, float, optional
            Angular position of the rail buttons in degrees measured
            as the rotation around the symmetry axis of the rocket
            relative to one of the other principal axis.
        name : string, optional
            Name of the rail buttons. Default is "Rail Buttons".

        Returns
        -------
        None

        """
        super().__init__(name)
        self.buttons_distance = buttons_distance
        self.angular_position = angular_position
        self.name = name

        self.evaluateLiftCoefficient()
        self.evaluateCenterOfPressure()

        self.prints = _RailButtonsPrints(self)
        return None

    def evaluateCenterOfPressure(self):
        """Evaluates the center of pressure of the rail buttons. Rail buttons
        do not contribute to the center of pressure of the rocket.

        Returns
        -------
        None
        """
        self.cpx = 0
        self.cpy = 0
        self.cpz = 0
        self.cp = (self.cpx, self.cpy, self.cpz)
        return None

    def evaluateLiftCoefficient(self):
        """Evaluates the lift coefficient curve of the rail buttons. Rail
        buttons do not contribute to the lift coefficient of the rocket.

        Returns
        -------
        None
        """
        self.clalpha = Function(
            lambda mach: 0,
            "Mach",
            f"Lift coefficient derivative for {self.name}",
        )
        self.cl = Function(
            lambda alpha, mach: 0,
            ["Alpha (rad)", "Mach"],
            "Cl",
        )
        return None

    def evaluateGeometricalParameters(self):
        """Evaluates the geometrical parameters of the rail buttons. Rail
        buttons do not contribute to the geometrical parameters of the rocket.

        Returns
        -------
        None
        """
        return None

    def info(self):
        """Prints out all the information about the Rail Buttons.

        Returns
        -------
        None
        """
        self.prints.geometry()
        return None

    def allInfo(self):
        """Returns all info of the Rail Buttons.

        Returns
        -------
        None
        """
        self.prints.all()
        return None
