import numpy as np

from rocketpy.plots.aero_surface_plots import _TrapezoidalFinsPlots
from rocketpy.prints.aero_surface_prints import _TrapezoidalFinsPrints

from .fins import Fins


class TrapezoidalFins(Fins):
    """Class that defines and holds information for a trapezoidal fin set.

    This class inherits from the Fins class.

    Note
    ----
    Local coordinate system:
        - Origin located at the top of the root chord.
        - Z axis along the longitudinal axis of symmetry, positive downwards (top -> bottom).
        - Y axis perpendicular to the Z axis, in the span direction, positive upwards.
        - X axis completes the right-handed coordinate system.

    See Also
    --------
    Fins

    Attributes
    ----------
    TrapezoidalFins.n : int
        Number of fins in fin set.
    TrapezoidalFins.rocket_radius : float
        The reference rocket radius used for lift coefficient normalization, in
        meters.
    TrapezoidalFins.airfoil : tuple
        Tuple of two items. First is the airfoil lift curve.
        Second is the unit of the curve (radians or degrees).
    TrapezoidalFins.cant_angle : float
        Fins cant angle with respect to the rocket centerline, in degrees.
    TrapezoidalFins.changing_attribute_dict : dict
        Dictionary that stores the name and the values of the attributes that
        may be changed during a simulation. Useful for control systems.
    TrapezoidalFins.cant_angle_rad : float
        Fins cant angle with respect to the rocket centerline, in radians.
    TrapezoidalFins.root_chord : float
        Fin root chord in meters.
    TrapezoidalFins.tip_chord : float
        Fin tip chord in meters.
    TrapezoidalFins.span : float
        Fin span in meters.
    TrapezoidalFins.name : string
        Name of fin set.
    TrapezoidalFins.sweep_length : float
        Fins sweep length in meters. By sweep length, understand the axial
        distance between the fin root leading edge and the fin tip leading edge
        measured parallel to the rocket centerline.
    TrapezoidalFins.sweep_angle : float
        Fins sweep angle with respect to the rocket centerline. Must
        be given in degrees.
    TrapezoidalFins.d : float
        Reference diameter of the rocket, in meters.
    TrapezoidalFins.ref_area : float
        Reference area of the rocket, in m².
    TrapezoidalFins.Af : float
        Area of the longitudinal section of each fin in the set.
    TrapezoidalFins.AR : float
        Aspect ratio of each fin in the set
    TrapezoidalFins.gamma_c : float
        Fin mid-chord sweep angle.
    TrapezoidalFins.Yma : float
        Span wise position of the mean aerodynamic chord.
    TrapezoidalFins.roll_geometrical_constant : float
        Geometrical constant used in roll calculations.
    TrapezoidalFins.tau : float
        Geometrical relation used to simplify lift and roll calculations.
    TrapezoidalFins.lift_interference_factor : float
        Factor of Fin-Body interference in the lift coefficient.
    TrapezoidalFins.cp : tuple
        Tuple with the x, y and z local coordinates of the fin set center of
        pressure. Has units of length and is given in meters.
    TrapezoidalFins.cpx : float
        Fin set local center of pressure x coordinate. Has units of length and
        is given in meters.
    TrapezoidalFins.cpy : float
        Fin set local center of pressure y coordinate. Has units of length and
        is given in meters.
    TrapezoidalFins.cpz : float
        Fin set local center of pressure z coordinate. Has units of length and
        is given in meters.
    TrapezoidalFins.cl : Function
        Function which defines the lift coefficient as a function of the angle
        of attack and the Mach number. Takes as input the angle of attack in
        radians and the Mach number. Returns the lift coefficient.
    TrapezoidalFins.clalpha : float
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
            Number of fins, must be larger than 2.
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
            Fins sweep length in meters. By sweep length, understand the axial
            distance between the fin root leading edge and the fin tip leading
            edge measured parallel to the rocket centerline. If not given, the
            sweep length is assumed to be equal the root chord minus the tip
            chord, in which case the fin is a right trapezoid with its base
            perpendicular to the rocket's axis. Cannot be used in conjunction
            with sweep_angle.
        sweep_angle : int, float, optional
            Fins sweep angle with respect to the rocket centerline. Must
            be given in degrees. If not given, the sweep angle is automatically
            calculated, in which case the fin is assumed to be a right trapezoid
            with its base perpendicular to the rocket's axis.
            Cannot be used in conjunction with sweep_length.
        airfoil : tuple, optional
            Default is null, in which case fins will be treated as flat plates.
            Otherwise, if tuple, fins will be considered as airfoils. The
            tuple's first item specifies the airfoil's lift coefficient
            by angle of attack and must be either a .csv, .txt, ndarray
            or callable. The .csv and .txt files can contain a single line
            header and the first column must specify the angle of attack, while
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
        coordinates. The center of pressure position is saved and stored as a
        tuple.

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

    def evaluate_geometrical_parameters(self):  # pylint: disable=too-many-statements
        """Calculates and saves fin set's geometrical parameters such as the
        fins' area, aspect ratio and parameters for roll movement.

        Returns
        -------
        None
        """
        # pylint: disable=invalid-name
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
        lambda_ = self.tip_chord / self.root_chord

        # Parameters for Roll Moment.
        # Documented at: https://docs.rocketpy.org/en/latest/technical/
        roll_geometrical_constant = (
            (self.root_chord + 3 * self.tip_chord) * self.span**3
            + 4
            * (self.root_chord + 2 * self.tip_chord)
            * self.rocket_radius
            * self.span**2
            + 6 * (self.root_chord + self.tip_chord) * self.span * self.rocket_radius**2
        ) / 12
        roll_damping_interference_factor = 1 + (
            ((tau - lambda_) / (tau)) - ((1 - lambda_) / (tau - 1)) * np.log(tau)
        ) / (
            ((tau + 1) * (tau - lambda_)) / (2)
            - ((1 - lambda_) * (tau**3 - 1)) / (3 * (tau - 1))
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
        self.λ = lambda_  # pylint: disable=non-ascii-name
        self.roll_damping_interference_factor = roll_damping_interference_factor
        self.roll_forcing_interference_factor = roll_forcing_interference_factor

        self.evaluate_shape()

    def evaluate_shape(self):
        if self.sweep_length:
            points = [
                (0, 0),
                (self.sweep_length, self.span),
                (self.sweep_length + self.tip_chord, self.span),
                (self.root_chord, 0),
            ]
        else:
            points = [
                (0, 0),
                (self.root_chord - self.tip_chord, self.span),
                (self.root_chord, self.span),
                (self.root_chord, 0),
            ]

        x_array, y_array = zip(*points)
        self.shape_vec = [np.array(x_array), np.array(y_array)]

    def info(self):
        self.prints.geometry()
        self.prints.lift()

    def all_info(self):
        self.prints.all()
        self.plots.all()

    def to_dict(self, **kwargs):
        data = super().to_dict(**kwargs)
        data["tip_chord"] = self.tip_chord
        data["sweep_length"] = self.sweep_length
        data["sweep_angle"] = self.sweep_angle

        if kwargs.get("include_outputs", False):
            data.update(
                {
                    "shape_vec": self.shape_vec,
                    "Af": self.Af,
                    "AR": self.AR,
                    "gamma_c": self.gamma_c,
                    "Yma": self.Yma,
                    "roll_geometrical_constant": self.roll_geometrical_constant,
                    "tau": self.tau,
                    "lift_interference_factor": self.lift_interference_factor,
                    "roll_damping_interference_factor": self.roll_damping_interference_factor,
                    "roll_forcing_interference_factor": self.roll_forcing_interference_factor,
                }
            )
        return data

    @classmethod
    def from_dict(cls, data):
        return cls(
            n=data["n"],
            root_chord=data["root_chord"],
            tip_chord=data["tip_chord"],
            span=data["span"],
            rocket_radius=data["rocket_radius"],
            cant_angle=data["cant_angle"],
            airfoil=data["airfoil"],
            name=data["name"],
            sweep_length=data.get("sweep_length"),
        )
