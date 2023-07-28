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
    def evaluate_center_of_pressure(self):
        """Evaluates the center of pressure of the aerodynamic surface in local
        coordinates.

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def evaluate_lift_coefficient(self):
        """Evaluates the lift coefficient curve of the aerodynamic surface.

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def evaluate_geometrical_parameters(self):
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
    def all_info(self):
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
    NoseCone.rocket_radius : float
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
        base_radius=None,
        rocket_radius=None,
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
        base_radius : float, optional
            Nose cone base radius. Has units of length and must be given in meters.
            If not given, the ratio between base_radius and rocket_radius will be
            assumed as 1.
        rocket_radius : int, float, optional
            The reference rocket radius used for lift coefficient normalization.
            If not given, the ratio between base_radius and rocket_radius will be
            assumed as 1.
        name : str, optional
            Nose cone name. Has no impact in simulation, as it is only used to
            display data in a more organized matter.

        Returns
        -------
        None
        """
        super().__init__(name)

        self._rocket_radius = rocket_radius
        self._base_radius = base_radius
        self._length = length
        self.kind = kind

        self.evaluate_geometrical_parameters()
        self.evaluate_lift_coefficient()
        self.evaluate_center_of_pressure()

        self.plots = _NoseConePlots(self)
        self.prints = _NoseConePrints(self)

        return None

    @property
    def rocket_radius(self):
        return self._rocket_radius

    @rocket_radius.setter
    def rocket_radius(self, value):
        self._rocket_radius = value
        self.evaluate_geometrical_parameters()
        self.evaluate_lift_coefficient()

    @property
    def base_radius(self):
        return self._base_radius

    @base_radius.setter
    def base_radius(self, value):
        self._base_radius = value
        self.evaluate_geometrical_parameters()
        self.evaluate_lift_coefficient()

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        self._length = value
        self.evaluate_center_of_pressure()

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
            rho = (self.base_radius**2 + self.length**2) / (2 * self.base_radius)
            volume = np.pi * (
                self.length * rho**2
                - (self.length**3) / 3
                - (rho - self.base_radius) * rho**2 * np.arcsin(self.length / rho)
            )
            area = np.pi * self.base_radius**2
            self.k = 1 - volume / (area * self.length)
        elif value == "elliptical":
            self.k = 1 / 3
        else:
            self.k = 0.5  # Parabolic and Von Karman
        self.evaluate_center_of_pressure()

    def evaluate_geometrical_parameters(self):
        """Calculates and saves nose cone's radius ratio.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.base_radius is None or self.rocket_radius is None:
            self.radius_ratio = 1
        else:
            self.radius_ratio = self.base_radius / self.rocket_radius

    def evaluate_lift_coefficient(self):
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
            lambda mach: 2 * self.radius_ratio**2,
            "Mach",
            f"Lift coefficient derivative for {self.name}",
        )
        self.cl = Function(
            lambda alpha, mach: self.clalpha(mach) * alpha,
            ["Alpha (rad)", "Mach"],
            "Cl",
        )
        return None

    def evaluate_center_of_pressure(self):
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

    def all_info(self):
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
    Fins.rocket_radius : float
        The reference rocket radius used for lift coefficient normalization, in meters.
    Fins.airfoil : tuple
        Tuple of two items. First is the airfoil lift curve.
        Second is the unit of the curve (radians or degrees).
    Fins.cant_angle : float
        Fins cant angle with respect to the rocket centerline, in degrees.
    Fins.changing_attribute_dict : dict
        Dictionary that stores the name and the values of the attributes that may
        be changed during a simulation. Useful for control systems.
    Fins.cant_angle_rad : float
        Fins cant angle with respect to the rocket centerline, in radians.
    Fins.root_chord : float
        Fin root chord in meters.
    Fins.tip_chord : float
        Fin tip chord in meters.
    Fins.span : float
        Fin span in meters.
    Fins.name : string
        Name of fin set.
    Fins.sweep_length : float
        Fins sweep length in meters. By sweep length, understand the axial distance
        between the fin root leading edge and the fin tip leading edge measured
        parallel to the rocket centerline.
    Fins.sweep_angle : float
        Fins sweep angle with respect to the rocket centerline. Must
        be given in degrees.
    Fins.d : float
        Reference diameter of the rocket. Has units of length and is given in meters.
    Fins.ref_area : float
        Reference area of the rocket.
    Fins.Af : float
        Area of the longitudinal section of each fin in the set.
    Fins.AR : float
        Aspect ratio of each fin in the set.
    Fins.gamma_c : float
        Fin mid-chord sweep angle.
    Fins.Yma : float
        Span wise position of the mean aerodynamic chord.
    Fins.roll_geometrical_constant : float
        Geometrical constant used in roll calculations.
    Fins.tau : float
        Geometrical relation used to simplify lift and roll calculations.
    Fins.lift_interference_factor : float
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
    Fins.roll_parameters : list
        List containing the roll moment lift coefficient, the roll moment damping
        coefficient and the cant angle in radians.
    """

    def __init__(
        self,
        n,
        root_chord,
        span,
        rocket_radius,
        cant_angle=0,
        airfoil=None,
        name="Fins",
    ):
        """Initialize Fins class.

        Parameters
        ----------
        n : int
            Number of fins, from 2 to infinity.
        root_chord : int, float
            Fin root chord in meters.
        span : int, float
            Fin span in meters.
        rocket_radius : int, float
            Reference rocket radius used for lift coefficient normalization.
        cant_angle : int, float, optional
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
        d = 2 * rocket_radius
        ref_area = np.pi * rocket_radius**2  # Reference area

        # Store values
        self._n = n
        self._rocket_radius = rocket_radius
        self._airfoil = airfoil
        self._cant_angle = cant_angle
        self._root_chord = root_chord
        self._span = span
        self.name = name
        self.d = d
        self.ref_area = ref_area  # Reference area

        return None

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, value):
        self._n = value
        self.evaluate_geometrical_parameters()
        self.evaluate_center_of_pressure()
        self.evaluate_lift_coefficient()
        self.evaluate_roll_parameters()

    @property
    def root_chord(self):
        return self._root_chord

    @root_chord.setter
    def root_chord(self, value):
        self._root_chord = value
        self.evaluate_geometrical_parameters()
        self.evaluate_center_of_pressure()
        self.evaluate_lift_coefficient()
        self.evaluate_roll_parameters()

    @property
    def span(self):
        return self._span

    @span.setter
    def span(self, value):
        self._span = value
        self.evaluate_geometrical_parameters()
        self.evaluate_center_of_pressure()
        self.evaluate_lift_coefficient()
        self.evaluate_roll_parameters()

    @property
    def rocket_radius(self):
        return self._rocket_radius

    @rocket_radius.setter
    def rocket_radius(self, value):
        self._rocket_radius = value
        self.evaluate_geometrical_parameters()
        self.evaluate_center_of_pressure()
        self.evaluate_lift_coefficient()
        self.evaluate_roll_parameters()

    @property
    def cant_angle(self):
        return self._cant_angle

    @cant_angle.setter
    def cant_angle(self, value):
        self._cant_angle = value
        self.evaluate_geometrical_parameters()
        self.evaluate_center_of_pressure()
        self.evaluate_lift_coefficient()
        self.evaluate_roll_parameters()

    @property
    def airfoil(self):
        return self._airfoil

    @airfoil.setter
    def airfoil(self, value):
        self._airfoil = value
        self.evaluate_geometrical_parameters()
        self.evaluate_center_of_pressure()
        self.evaluate_lift_coefficient()
        self.evaluate_roll_parameters()

    def evaluate_lift_coefficient(self):
        """Calculates and returns the fin set's lift coefficient.
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
            self.airfoil_cl = Function(
                self.airfoil[0],
                interpolation="linear",
            )

            # Differentiating at x = 0 to get cl_alpha
            clalpha2D_Mach0 = self.airfoil_cl.differentiate(x=1e-3, dx=1e-3)

            # Convert to radians if needed
            if self.airfoil[1] == "degrees":
                clalpha2D_Mach0 *= 180 / np.pi

            # Correcting for compressible flow
            clalpha2D = Function(lambda mach: clalpha2D_Mach0 / self.__beta(mach))

        # Diederich's Planform Correlation Parameter
        FD = 2 * np.pi * self.AR / (clalpha2D * np.cos(self.gamma_c))

        # Lift coefficient derivative for a single fin
        self.clalpha_single_fin = Function(
            lambda mach: (
                clalpha2D(mach)
                * FD(mach)
                * (self.Af / self.ref_area)
                * np.cos(self.gamma_c)
            )
            / (2 + FD(mach) * np.sqrt(1 + (2 / FD(mach)) ** 2)),
            "Mach",
            "Lift coefficient derivative for a single fin",
        )

        # Lift coefficient derivative for a number of n fins corrected for Fin-Body interference
        self.clalpha_multiple_fins = (
            self.lift_interference_factor
            * self.__fin_num_correction(self.n)
            * self.clalpha_single_fin
        )  # Function of mach number
        self.clalpha_multiple_fins.set_inputs("Mach")
        self.clalpha_multiple_fins.set_outputs(
            "Lift coefficient derivative for {:.0f} fins".format(self.n)
        )
        self.clalpha = self.clalpha_multiple_fins

        # Calculates clalpha * alpha
        self.cl = Function(
            lambda alpha, mach: alpha * self.clalpha_multiple_fins(mach),
            ["Alpha (rad)", "Mach"],
            "Lift coefficient",
        )

        return self.cl

    def evaluate_roll_parameters(self):
        """Calculates and returns the fin set's roll coefficients.
        The roll coefficients are saved in a list.

        Parameters
        ----------
        None

        Returns
        -------
        self.roll_parameters : list
            List containing the roll moment lift coefficient, the
            roll moment damping coefficient and the cant angle in
            radians
        """

        self.cant_angle_rad = np.radians(self.cant_angle)

        clf_delta = (
            self.roll_forcing_interference_factor
            * self.n
            * (self.Yma + self.rocket_radius)
            * self.clalpha_single_fin
            / self.d
        )  # Function of mach number
        clf_delta.set_inputs("Mach")
        clf_delta.set_outputs("Roll moment forcing coefficient derivative")
        cld_omega = (
            2
            * self.roll_damping_interference_factor
            * self.n
            * self.clalpha_single_fin
            * np.cos(self.cant_angle_rad)
            * self.roll_geometrical_constant
            / (self.ref_area * self.d**2)
        )  # Function of mach number
        cld_omega.set_inputs("Mach")
        cld_omega.set_outputs("Roll moment damping coefficient derivative")
        self.roll_parameters = [clf_delta, cld_omega, self.cant_angle_rad]
        return self.roll_parameters

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
    def __fin_num_correction(_, n):
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
        corrector_factor = [2.37, 2.74, 2.99, 3.24]
        if n >= 5 and n <= 8:
            return corrector_factor[n - 5]
        else:
            return n / 2

    def draw(self):
        self.plots.draw()


class TrapezoidalFins(Fins):
    """Class that defines and holds information for a trapezoidal fin set.

    Attributes
    ----------

        Geometrical attributes:
        Fins.n : int
            Number of fins in fin set.
        Fins.rocket_radius : float
            The reference rocket radius used for lift coefficient normalization, in
            meters.
        Fins.airfoil : tuple
            Tuple of two items. First is the airfoil lift curve.
            Second is the unit of the curve (radians or degrees).
        Fins.cant_angle : float
            Fins cant angle with respect to the rocket centerline, in degrees.
        Fins.changing_attribute_dict : dict
            Dictionary that stores the name and the values of the attributes that may
            be changed during a simulation. Useful for control systems.
        Fins.cant_angle_rad : float
            Fins cant angle with respect to the rocket centerline, in radians.
        Fins.root_chord : float
            Fin root chord in meters.
        Fins.tip_chord : float
            Fin tip chord in meters.
        Fins.span : float
            Fin span in meters.
        Fins.name : string
            Name of fin set.
        Fins.sweep_length : float
            Fins sweep length in meters. By sweep length, understand the axial distance
            between the fin root leading edge and the fin tip leading edge measured
            parallel to the rocket centerline.
        Fins.sweep_angle : float
            Fins sweep angle with respect to the rocket centerline. Must
            be given in degrees.
        Fins.d : float
            Reference diameter of the rocket, in meters.
        Fins.ref_area : float
            Reference area of the rocket, in m².
        Fins.Af : float
            Area of the longitudinal section of each fin in the set.
        Fins.AR : float
            Aspect ratio of each fin in the set
        Fins.gamma_c : float
            Fin mid-chord sweep angle.
        Fins.Yma : float
            Span wise position of the mean aerodynamic chord.
        Fins.roll_geometrical_constant : float
            Geometrical constant used in roll calculations.
        Fins.tau : float
            Geometrical relation used to simplify lift and roll calculations.
        Fins.lift_interference_factor : float
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
        root_chord,
        tip_chord,
        span,
        rocket_radius,
        cant_angle=0,
        sweep_length=None,
        sweep_angle=None,
        airfoil=None,
        name="Fins",
    ):
        """Initialize TrapezoidalFins class.

        Parameters
        ----------
        n : int
            Number of fins, from 2 to infinity.
        root_chord : int, float
            Fin root chord in meters.
        tip_chord : int, float
            Fin tip chord in meters.
        span : int, float
            Fin span in meters.
        rocket_radius : int, float
            Reference radius to calculate lift coefficient, in meters.
        cant_angle : int, float, optional
            Fins cant angle with respect to the rocket centerline. Must
            be given in degrees.
        sweep_length : int, float, optional
            Fins sweep length in meters. By sweep length, understand the axial distance
            between the fin root leading edge and the fin tip leading edge measured
            parallel to the rocket centerline. If not given, the sweep length is
            assumed to be equal the root chord minus the tip chord, in which case the
            fin is a right trapezoid with its base perpendicular to the rocket's axis.
            Cannot be used in conjunction with sweep_angle.
        sweep_angle : int, float, optional
            Fins sweep angle with respect to the rocket centerline. Must
            be given in degrees. If not given, the sweep angle is automatically
            calculated, in which case the fin is assumed to be a right trapezoid with
            its base perpendicular to the rocket's axis.
            Cannot be used in conjunction with sweep_length.
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
            root_chord,
            span,
            rocket_radius,
            cant_angle,
            airfoil,
            name,
        )

        # Check if sweep angle or sweep length is given
        if sweep_length is not None and sweep_angle is not None:
            raise ValueError("Cannot use sweep_length and sweep_angle together")
        elif sweep_angle is not None:
            sweep_length = np.tan(sweep_angle * np.pi / 180) * span
        elif sweep_length is None:
            sweep_length = root_chord - tip_chord
        else:
            # Sweep length is given
            pass

        self._tip_chord = tip_chord
        self._sweep_length = sweep_length
        self._sweep_angle = sweep_angle

        self.evaluate_geometrical_parameters()
        self.evaluate_center_of_pressure()
        self.evaluate_lift_coefficient()
        self.evaluate_roll_parameters()

        self.prints = _TrapezoidalFinsPrints(self)
        self.plots = _TrapezoidalFinsPlots(self)

    @property
    def tip_chord(self):
        return self._tip_chord

    @tip_chord.setter
    def tip_chord(self, value):
        self._tip_chord = value
        self.evaluate_geometrical_parameters()
        self.evaluate_center_of_pressure()
        self.evaluate_lift_coefficient()
        self.evaluate_roll_parameters()

    @property
    def sweep_angle(self):
        return self._sweep_angle

    @sweep_angle.setter
    def sweep_angle(self, value):
        self._sweep_angle = value
        self._sweep_length = np.tan(value * np.pi / 180) * self.span
        self.evaluate_geometrical_parameters()
        self.evaluate_center_of_pressure()
        self.evaluate_lift_coefficient()
        self.evaluate_roll_parameters()

    @property
    def sweep_length(self):
        return self._sweep_length

    @sweep_length.setter
    def sweep_length(self, value):
        self._sweep_length = value
        self.evaluate_geometrical_parameters()
        self.evaluate_center_of_pressure()
        self.evaluate_lift_coefficient()
        self.evaluate_roll_parameters()

    def evaluate_center_of_pressure(self):
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
        cpz = (self.sweep_length / 3) * (
            (self.root_chord + 2 * self.tip_chord) / (self.root_chord + self.tip_chord)
        ) + (1 / 6) * (
            self.root_chord
            + self.tip_chord
            - self.root_chord * self.tip_chord / (self.root_chord + self.tip_chord)
        )
        self.cpx = 0
        self.cpy = 0
        self.cpz = cpz
        self.cp = (self.cpx, self.cpy, self.cpz)
        return None

    def evaluate_geometrical_parameters(self):
        """Calculates and saves fin set's geometrical parameters such as the
        fins' area, aspect ratio and parameters for roll movement.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        Yr = self.root_chord + self.tip_chord
        Af = Yr * self.span / 2  # Fin area
        AR = 2 * self.span**2 / Af  # Fin aspect ratio
        gamma_c = np.arctan(
            (self.sweep_length + 0.5 * self.tip_chord - 0.5 * self.root_chord)
            / (self.span)
        )
        Yma = (
            (self.span / 3) * (self.root_chord + 2 * self.tip_chord) / Yr
        )  # Span wise coord of mean aero chord

        # Fin–body interference correction parameters
        tau = (self.span + self.rocket_radius) / self.rocket_radius
        lift_interference_factor = 1 + 1 / tau
        λ = self.tip_chord / self.root_chord

        # Parameters for Roll Moment.
        # Documented at: https://github.com/RocketPy-Team/RocketPy/blob/master/docs/technical/aerodynamics/Roll_Equations.pdf
        roll_geometrical_constant = (
            (self.root_chord + 3 * self.tip_chord) * self.span**3
            + 4
            * (self.root_chord + 2 * self.tip_chord)
            * self.rocket_radius
            * self.span**2
            + 6
            * (self.root_chord + self.tip_chord)
            * self.span
            * self.rocket_radius**2
        ) / 12
        roll_damping_interference_factor = 1 + (
            ((tau - λ) / (tau)) - ((1 - λ) / (tau - 1)) * np.log(tau)
        ) / (
            ((tau + 1) * (tau - λ)) / (2) - ((1 - λ) * (tau**3 - 1)) / (3 * (tau - 1))
        )
        roll_forcing_interference_factor = (1 / np.pi**2) * (
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
        self.roll_geometrical_constant = roll_geometrical_constant
        self.tau = tau
        self.lift_interference_factor = lift_interference_factor
        self.λ = λ
        self.roll_damping_interference_factor = roll_damping_interference_factor
        self.roll_forcing_interference_factor = roll_forcing_interference_factor

    def info(self):
        self.prints.geometry()
        self.prints.lift()
        return None

    def all_info(self):
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
        Fins.rocket_radius : float
            The reference rocket radius used for lift coefficient normalization, in
            meters.
        Fins.airfoil : tuple
            Tuple of two items. First is the airfoil lift curve.
            Second is the unit of the curve (radians or degrees)
        Fins.cant_angle : float
            Fins cant angle with respect to the rocket centerline, in degrees.
        Fins.changing_attribute_dict : dict
            Dictionary that stores the name and the values of the attributes that may
            be changed during a simulation. Useful for control systems.
        Fins.cant_angle_rad : float
            Fins cant angle with respect to the rocket centerline, in radians.
        Fins.root_chord : float
            Fin root chord in meters.
        Fins.span : float
            Fin span in meters.
        Fins.name : string
            Name of fin set.
        Fins.sweep_length : float
            Fins sweep length in meters. By sweep length, understand the axial distance
            between the fin root leading edge and the fin tip leading edge measured
            parallel to the rocket centerline.
        Fins.sweep_angle : float
            Fins sweep angle with respect to the rocket centerline. Must
            be given in degrees.
        Fins.d : float
            Reference diameter of the rocket, in meters.
        Fins.ref_area : float
            Reference area of the rocket.
        Fins.Af : float
            Area of the longitudinal section of each fin in the set.
        Fins.AR : float
            Aspect ratio of each fin in the set.
        Fins.gamma_c : float
            Fin mid-chord sweep angle.
        Fins.Yma : float
            Span wise position of the mean aerodynamic chord.
        Fins.roll_geometrical_constant : float
            Geometrical constant used in roll calculations.
        Fins.tau : float
            Geometrical relation used to simplify lift and roll calculations.
        Fins.lift_interference_factor : float
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
        root_chord,
        span,
        rocket_radius,
        cant_angle=0,
        airfoil=None,
        name="Fins",
    ):
        """Initialize EllipticalFins class.

        Parameters
        ----------
        n : int
            Number of fins, from 2 to infinity.
        root_chord : int, float
            Fin root chord in meters.
        span : int, float
            Fin span in meters.
        rocket_radius : int, float
            Reference radius to calculate lift coefficient, in meters.
        cant_angle : int, float, optional
            Fins cant angle with respect to the rocket centerline. Must
            be given in degrees.
        sweep_length : int, float, optional
            Fins sweep length in meters. By sweep length, understand the axial distance
            between the fin root leading edge and the fin tip leading edge measured
            parallel to the rocket centerline. If not given, the sweep length is
            assumed to be equal the root chord minus the tip chord, in which case the
            fin is a right trapezoid with its base perpendicular to the rocket's axis.
            Cannot be used in conjunction with sweep_angle.
        sweep_angle : int, float, optional
            Fins sweep angle with respect to the rocket centerline. Must
            be given in degrees. If not given, the sweep angle is automatically
            calculated, in which case the fin is assumed to be a right trapezoid with
            its base perpendicular to the rocket's axis.
            Cannot be used in conjunction with sweep_length.
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
            root_chord,
            span,
            rocket_radius,
            cant_angle,
            airfoil,
            name,
        )

        self.evaluate_geometrical_parameters()
        self.evaluate_center_of_pressure()
        self.evaluate_lift_coefficient()
        self.evaluate_roll_parameters()

        self.prints = _EllipticalFinsPrints(self)
        self.plots = _EllipticalFinsPlots(self)

        return None

    def evaluate_center_of_pressure(self):
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
        cpz = 0.288 * self.root_chord
        self.cpx = 0
        self.cpy = 0
        self.cpz = cpz
        self.cp = (self.cpx, self.cpy, self.cpz)
        return None

    def evaluate_geometrical_parameters(self):
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
        Af = (np.pi * self.root_chord / 2 * self.span) / 2  # Fin area
        gamma_c = 0  # Zero for elliptical fins
        AR = 2 * self.span**2 / Af  # Fin aspect ratio
        Yma = (
            self.span / (3 * np.pi) * np.sqrt(9 * np.pi**2 - 64)
        )  # Span wise coord of mean aero chord
        roll_geometrical_constant = (
            self.root_chord
            * self.span
            * (
                3 * np.pi * self.span**2
                + 32 * self.rocket_radius * self.span
                + 12 * np.pi * self.rocket_radius**2
            )
            / 48
        )

        # Fin–body interference correction parameters
        tau = (self.span + self.rocket_radius) / self.rocket_radius
        lift_interference_factor = 1 + 1 / tau
        roll_damping_interference_factor = 1 + (
            (self.rocket_radius**2)
            * (
                2
                * (self.rocket_radius**2)
                * np.sqrt(self.span**2 - self.rocket_radius**2)
                * np.log(
                    (
                        2
                        * self.span
                        * np.sqrt(self.span**2 - self.rocket_radius**2)
                        + 2 * self.span**2
                    )
                    / self.rocket_radius
                )
                - 2
                * (self.rocket_radius**2)
                * np.sqrt(self.span**2 - self.rocket_radius**2)
                * np.log(2 * self.span)
                + 2 * self.span**3
                - np.pi * self.rocket_radius * self.span**2
                - 2 * (self.rocket_radius**2) * self.span
                + np.pi * self.rocket_radius**3
            )
        ) / (
            2
            * (self.span**2)
            * (self.span / 3 + np.pi * self.rocket_radius / 4)
            * (self.span**2 - self.rocket_radius**2)
        )
        roll_forcing_interference_factor = (1 / np.pi**2) * (
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
        self.roll_geometrical_constant = roll_geometrical_constant
        self.tau = tau
        self.lift_interference_factor = lift_interference_factor
        self.roll_damping_interference_factor = roll_damping_interference_factor
        self.roll_forcing_interference_factor = roll_forcing_interference_factor

    def info(self):
        self.prints.geometry()
        self.prints.lift()
        return None

    def all_info(self):
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
    Tail.top_radius : int, float
        Radius of the top of the tail. The top radius is defined as the radius
        of the transversal section that is closest to the rocket's nose.
    Tail.bottom_radius : int, float
        Radius of the bottom of the tail.
    Tail.length : int, float
        Length of the tail. The length is defined as the distance between the
        top and bottom of the tail. The length is measured along the rocket's
        longitudinal axis. Has the unit of meters.
    Tail.rocket_radius: int, float
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
    Tail.slant_length : float
        Slant length of the tail. The slant length is defined as the distance
        between the top and bottom of the tail. The slant length is measured
        along the tail's slant axis. Has the unit of meters.
    Tail.surface_area : float
        Surface area of the tail. Has the unit of meters squared.

    """

    def __init__(self, top_radius, bottom_radius, length, rocket_radius, name="Tail"):
        """Initializes the tail object by computing and storing the most
        important values.

        Parameters
        ----------
        top_radius : int, float
            Radius of the top of the tail. The top radius is defined as the radius
            of the transversal section that is closest to the rocket's nose.
        bottom_radius : int, float
            Radius of the bottom of the tail.
        length : int, float
            Length of the tail.
        rocket_radius : int, float
            The reference rocket radius used for lift coefficient normalization.
        name : str
            Name of the tail. Default is 'Tail'.

        Returns
        -------
        None
        """
        super().__init__(name)

        # Store arguments as attributes
        self._top_radius = top_radius
        self._bottom_radius = bottom_radius
        self._length = length
        self._rocket_radius = rocket_radius

        # Calculate geometrical parameters
        self.evaluate_geometrical_parameters()
        self.evaluate_lift_coefficient()
        self.evaluate_center_of_pressure()

        self.plots = _TailPlots(self)
        self.prints = _TailPrints(self)

        return None

    @property
    def top_radius(self):
        return self._top_radius

    @top_radius.setter
    def top_radius(self, value):
        self._top_radius = value
        self.evaluate_geometrical_parameters()
        self.evaluate_lift_coefficient()
        self.evaluate_center_of_pressure()

    @property
    def bottom_radius(self):
        return self._bottom_radius

    @bottom_radius.setter
    def bottom_radius(self, value):
        self._bottom_radius = value
        self.evaluate_geometrical_parameters()
        self.evaluate_lift_coefficient()
        self.evaluate_center_of_pressure()

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        self._length = value
        self.evaluate_geometrical_parameters()
        self.evaluate_center_of_pressure()

    @property
    def rocket_radius(self):
        return self._rocket_radius

    @rocket_radius.setter
    def rocket_radius(self, value):
        self._rocket_radius = value
        self.evaluate_lift_coefficient()

    def evaluate_geometrical_parameters(self):
        """Calculates and saves tail's slant length and surface area.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Calculate tail slant length
        self.slant_length = np.sqrt(
            (self.length) ** 2 + (self.top_radius - self.bottom_radius) ** 2
        )
        # Calculate the surface area of the tail
        self.surface_area = (
            np.pi * self.slant_length * (self.top_radius + self.bottom_radius)
        )

    def evaluate_lift_coefficient(self):
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
                (self.bottom_radius / self.rocket_radius) ** 2
                - (self.top_radius / self.rocket_radius) ** 2
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

    def evaluate_center_of_pressure(self):
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
        r = self.top_radius / self.bottom_radius
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

    def all_info(self):
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

        self.evaluate_lift_coefficient()
        self.evaluate_center_of_pressure()

        self.prints = _RailButtonsPrints(self)
        return None

    def evaluate_center_of_pressure(self):
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

    def evaluate_lift_coefficient(self):
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

    def evaluate_geometrical_parameters(self):
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

    def all_info(self):
        """Returns all info of the Rail Buttons.

        Returns
        -------
        None
        """
        self.prints.all()
        return None
