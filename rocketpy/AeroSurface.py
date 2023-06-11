__author__ = "Guilherme Fernandes Alves, Mateus Stano Junqueira, Giovani Hidalgo Ceotto, Franz MasatoshiYuri, Calebe Gomes Teles"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np

from .Function import Function
from abc import ABC, abstractmethod, abstractproperty
from matplotlib.patches import Ellipse
from scipy.optimize import fsolve


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
    def geometricalInfo(self):
        """Returns the geometrical info of the aerodynamic surface.

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def aerodynamicInfo(self):
        """Returns the aerodynamic info of the aerodynamic surface.

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def allInfo(self):
        """Returns all info of the aerodynamic surface.

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
        Nose cone kind. Can be "conical", "ogive", "elliptical", "tangent",
        "von karman", "parabolic" or "lvhaack".
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
        self,
        length,
        kind,
        baseRadius=None,
        bluffiness=0,
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
            Nose cone base radius. Has units of length and must be given in
            meters.
            If not given, the ratio between baseRadius and rocketRadius will be
            assumed as 1.
        bluffiness : float, optional
            Ratio between the radius of the circle on the tip of the ogive and
            the radius of the base of the ogive.
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
        self.bluffiness = bluffiness
        self.kind = kind

        self.evaluateGeometricalParameters()
        self.evaluateLiftCoefficient()
        self.evaluateCenterOfPressure()

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
        # Analyzes nosecone type
        # Sets the k for Cp calculation
        # Sets the function which creates the respective curve
        self._kind = value
        value = (value.replace(" ", "")).lower()

        if value == "conical":
            self.k = 2 / 3
            self.y_nosecone = Function(lambda x: x * self.baseRadius / self.length)

        elif value == "lvhaack":
            self.k = 0.563
            theta = lambda x: np.arccos(1 - 2 * (x / self.length))
            self.y_nosecone = Function(
                lambda x: self.baseRadius
                * (theta(x) - np.sin(2 * theta(x)) / 2 + (np.sin(theta) ** 3) / 3)
                ** (0.5)
                / (np.pi**0.5)
            )

        elif value in ["tangent", "tangentogive", "ogive"]:
            rho = (self.baseRadius**2 + self.length**2) / (2 * self.baseRadius)
            volume = np.pi * (
                self.length * rho**2
                - (self.length**3) / 3
                - (rho - self.baseRadius) * rho**2 * np.arcsin(self.length / rho)
            )
            area = np.pi * self.baseRadius**2
            self.k = 1 - volume / (area * self.length)
            self.y_nosecone = Function(
                lambda x: np.sqrt(rho**2 - (x - self.length) ** 2)
                + (self.baseRadius - rho)
            )

        elif value == "elliptical":
            self.k = 1 / 3
            self.y_nosecone = Function(
                lambda x: self.baseRadius
                * np.sqrt(1 - ((x - self.length) / self.length) ** 2)
            )

        elif value == "vonkarman":
            self.k = 0.5
            theta = lambda x: np.arccos(1 - 2 * (x / self.length))
            self.y_nosecone = Function(
                lambda x: self.baseRadius
                * (theta(x) - np.sin(2 * theta(x)) / 2) ** (0.5)
                / (np.pi**0.5)
            )
        elif value == "parabolic":
            self.k = 0.5
            self.y_nosecone = Function(
                lambda x: self.baseRadius
                * ((2 * x / self.length - (x / self.length) ** 2) / (2 - 1))
            )

        else:
            raise ValueError(
                f"Nose Cone kind '{self.kind}' not found, "
                + "please use one of the following Nose Cone kinds:"
                + '\n\t"conical"'
                + '\n\t"ogive"'
                + '\n\t"lvhaack"'
                + '\n\t"tangent"'
                + '\n\t"vonkarman"'
                + '\n\t"elliptical"'
                + '\n\t"parabolic"\n'
            )

        n = 127  # Points on the final curve.
        p = 3  # Density modifier. Greater n makes more points closer to 0. n=1 -> points equally spaced.

        # Finds the tangential intersection point between the circle and nosecone curve.
        yPrimeNosecone = lambda x: self.y_nosecone.differentiate(x)
        xIntercept = lambda x: x + self.y_nosecone(x) * yPrimeNosecone(x)
        radius = lambda x: (self.y_nosecone(x) ** 2 + (x - xIntercept(x)) ** 2) ** 0.5
        xInit = fsolve(
            lambda x: radius(x) - self.bluffiness * self.baseRadius if x > 3e-7 else 0,
            self.bluffiness * self.baseRadius,
            xtol=1e-7,
        )[0]

        # Corrects circle radius if it's too small.
        if xInit > 0:
            r = self.bluffiness * self.baseRadius
        else:
            r = 0
            print(
                "ATTENTION: The chosen bluffiness ratio is insufficient for the selected nosecone category, thereby the effective bluffiness will be 0."
            )

        # Creates the circle at correct position.
        circleCenter = xIntercept(xInit)
        circle = lambda x: abs(r**2 - (x - circleCenter) ** 2) ** 0.5

        # Function defining final shape of curve with circle o the tip.
        finalShape = Function(lambda x: self.y_nosecone(x) if x >= xInit else circle(x))
        finalShapeVec = np.vectorize(finalShape)

        # Creates the vectors X and Y with the points of the curve.
        self.nosecone_Xs = (self.length - (circleCenter - r)) * (
            np.linspace(0, 1, n) ** p
        )
        self.nosecone_Ys = finalShapeVec(self.nosecone_Xs + (circleCenter - r))

        # Evaluates final geometry parameters.
        self.length = self.nosecone_Xs[-1]
        self.FinenessRatio = self.length / (2 * self.baseRadius)
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

    def draw(self):
        # Figure creation and set up
        fig_Ogive, ax = plt.subplots()
        ax.set_xlim(-0.05, self.length * 1.02)  # Horizontal size
        ax.set_ylim(-self.baseRadius * 1.05, self.baseRadius * 1.05)  # Vertical size
        ax.set_aspect("equal")  # Makes the graduation be the same on both axis
        ax.set_facecolor("#EEEEEE")  # Background colour
        ax.grid(True, linestyle="--", linewidth=0.5)

        cp_plot = (self.cpz, 0)
        # Plotting
        ax.plot(
            self.nosecone_Xs, self.nosecone_Ys, linestyle="-", color="#A60628"
        )  # Ogive's upper side
        ax.plot(
            self.nosecone_Xs, -self.nosecone_Ys, linestyle="-", color="#A60628"
        )  # Ogive's lower side
        ax.scatter(
            *cp_plot, label="Center Of Pressure", color="red", s=100, zorder=10
        )  # Center of pressure inner circle
        ax.scatter(
            *cp_plot, facecolors="none", edgecolors="red", s=500, zorder=10
        )  # Center of pressure outer circle
        # Center Line
        ax.plot(
            [0, self.nosecone_Xs[len(self.nosecone_Xs) - 1]],
            [0, 0],
            linestyle="--",
            color="#7A68A6",
            linewidth=1.5,
            label="Center Line",
        )
        # Vertical base line
        ax.plot(
            [
                self.nosecone_Xs[len(self.nosecone_Xs) - 1],
                self.nosecone_Xs[len(self.nosecone_Xs) - 1],
            ],
            [
                self.nosecone_Ys[len(self.nosecone_Ys) - 1],
                -self.nosecone_Ys[len(self.nosecone_Ys) - 1],
            ],
            linestyle="-",
            color="#A60628",
            linewidth=1.5,
        )

        # Labels and legend
        ax.set_xlabel("Length")
        ax.set_ylabel("Radius")
        ax.set_title(self.kind + " Nose Cone")
        ax.legend(bbox_to_anchor=(1, -0.2))
        # Show Plot
        plt.show()
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
        print(f"\nGeometric Information of {self.name}")
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
        print(f"\nAerodynamic Information of {self.name}")
        print("-------------------------------")
        print(f"Center of Pressure Position in Local Coordinates: {self.cp} m")
        print(f"Lift Coefficient Slope at Mach 0: {self.clalpha(0):.3f} 1/rad")
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
        self.geometricalInfo()
        self.aerodynamicInfo()

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

        print("\nGeometrical Parameters\n")
        if isinstance(self, TrapezoidalFins):
            print("Fin Type: Trapezoidal")
            print("Tip Chord: {:.3f} m".format(self.tipChord))
        else:
            print("Fin Type: Elliptical")
        print("Root Chord: {:.3f} m".format(self.rootChord))
        print("Span: {:.3f} m".format(self.span))
        print("Cant Angle: {:.3f} °".format(self.cantAngle))
        print("Longitudinal Section Area: {:.3f} m²".format(self.Af))
        print("Aspect Ratio: {:.3f} ".format(self.AR))
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
        print("\nAerodynamic Information")
        print("----------------")
        print("Lift Interference Factor: {:.3f} m".format(self.liftInterferenceFactor))
        print(
            "Center of Pressure position in Local Coordinates: ({:.3f},{:.3f},{:.3f}) (x, y, z)".format(
                self.cpx, self.cpy, self.cpz
            )
        )
        print()
        print(
            "Lift Coefficient derivative as a Function of Alpha and Mach for Single Fin"
        )
        print()
        self.clalphaSingleFin()
        print()
        print(
            "Lift Coefficient derivative as a Function of Alpha and Mach for the Fin Set"
        )
        print()
        self.clalphaMultipleFins()
        print()
        print("Lift Coefficient as a Function of Alpha and Mach for the Fin Set")
        print()
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
        # lacks a title for the plot
        self.rollParameters[0]()
        # lacks a title for the plot
        self.rollParameters[1]()

        return None

    def airfoilInfo(self):
        """Prints out airfoil related information of the
        fin set.

        Parameters
        ----------
        None

        Return
        ------
        None
        """
        if self.airfoil is not None:
            print("\n\nAerodynamic Information\n")
            print(
                "Airfoil's Lift Curve as a Function of Alpha ({}))".format(
                    self.airfoil[1]
                )
            )
            self.airfoilCl.plot1D()

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

        print("Fin set information\n")

        self.draw()

        print("Basic Information\n")

        print("Number of fins: {:.0f}".format(self.n))
        print("Reference rocket radius: {:.3f} m".format(self.rocketRadius))

        self.geometricalInfo()
        self.aerodynamicInfo()
        self.rollInfo()
        self.airfoilInfo()

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

        self._tipChord = tipChord
        self._sweepLength = sweepLength
        self._sweepAngle = sweepAngle

        self.evaluateGeometricalParameters()
        self.evaluateCenterOfPressure()
        self.evaluateLiftCoefficient()
        self.evaluateRollParameters()

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

    def geometricalInfo(self):
        """Prints out all the geometric information of the tail.

        Returns
        -------
        None
        """

        print(f"\nGeometric Information of {self.name}")
        print("-------------------------------")
        print(f"Tail Top Radius: {self.topRadius:.3f} m")
        print(f"Tail Bottom Radius: {self.bottomRadius:.3f} m")
        print(f"Tail Length: {self.length:.3f} m")
        print(f"Reference Radius: {2*self.rocketRadius:.3f} m")
        print(f"Tail Slant Length: {self.slantLength:.3f} m")
        print(f"Tail Surface Area: {self.surfaceArea:.6f} m²")

        return None

    def aerodynamicInfo(self):
        print(f"\nAerodynamic Information of {self.name}")
        print("-------------------------------")
        print(f"Tail Center of Pressure Position in Local Coordinates: {self.cp} m")
        print(f"Tail Lift Coefficient Slope at Mach 0: {self.clalpha(0):.3f} 1/rad")
        print("Tail Lift Coefficient as a function of Alpha and Mach:")
        self.cl()

        return None

    def allInfo(self):
        """Prints all the information about the tail object.

        Returns
        -------
        None
        """
        self.geometricalInfo()
        self.aerodynamicInfo()

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
        self.buttons_distance = buttons_distance
        self.angular_position = angular_position
        self.name = name

        self.evaluateLiftCoefficient()
        self.evaluateCenterOfPressure()
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

    def geometricalInfo(self):
        """Returns the geometrical info of the rail buttons. Rail buttons
        do not have geometrical parameters.

        Returns
        -------
        None
        """
        return None

    def aerodynamicInfo(self):
        """Returns the aerodynamic info of the aerodynamic surface.

        Returns
        -------
        None
        """
        return None

    def allInfo(self):
        """Returns all info of the aerodynamic surface.

        Returns
        -------
        None
        """
        return None
