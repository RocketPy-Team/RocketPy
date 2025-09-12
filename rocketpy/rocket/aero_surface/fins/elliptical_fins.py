import numpy as np

from rocketpy.plots.aero_surface_plots import _EllipticalFinsPlots
from rocketpy.prints.aero_surface_prints import _EllipticalFinsPrints

from .fins import Fins


class EllipticalFins(Fins):
    """Class that defines and holds information for an elliptical fin set.

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
    EllipticalFins.n : int
        Number of fins in fin set.
    EllipticalFins.rocket_radius : float
        The reference rocket radius used for lift coefficient normalization, in
        meters.
    EllipticalFins.airfoil : tuple
        Tuple of two items. First is the airfoil lift curve.
        Second is the unit of the curve (radians or degrees)
    EllipticalFins.cant_angle : float
        Fins cant angle with respect to the rocket centerline, in degrees.
    EllipticalFins.changing_attribute_dict : dict
        Dictionary that stores the name and the values of the attributes that
        may be changed during a simulation. Useful for control systems.
    EllipticalFins.cant_angle_rad : float
        Fins cant angle with respect to the rocket centerline, in radians.
    EllipticalFins.root_chord : float
        Fin root chord in meters.
    EllipticalFins.span : float
        Fin span in meters.
    EllipticalFins.name : string
        Name of fin set.
    EllipticalFins.sweep_length : float
        Fins sweep length in meters. By sweep length, understand the axial
        distance between the fin root leading edge and the fin tip leading edge
        measured parallel to the rocket centerline.
    EllipticalFins.sweep_angle : float
        Fins sweep angle with respect to the rocket centerline. Must
        be given in degrees.
    EllipticalFins.d : float
        Reference diameter of the rocket, in meters.
    EllipticalFins.ref_area : float
        Reference area of the rocket.
    EllipticalFins.Af : float
        Area of the longitudinal section of each fin in the set.
    EllipticalFins.AR : float
        Aspect ratio of each fin in the set.
    EllipticalFins.gamma_c : float
        Fin mid-chord sweep angle.
    EllipticalFins.Yma : float
        Span wise position of the mean aerodynamic chord.
    EllipticalFins.roll_geometrical_constant : float
        Geometrical constant used in roll calculations.
    EllipticalFins.tau : float
        Geometrical relation used to simplify lift and roll calculations.
    EllipticalFins.lift_interference_factor : float
        Factor of Fin-Body interference in the lift coefficient.
    EllipticalFins.cp : tuple
        Tuple with the x, y and z local coordinates of the fin set center of
        pressure. Has units of length and is given in meters.
    EllipticalFins.cpx : float
        Fin set local center of pressure x coordinate. Has units of length and
        is given in meters.
    EllipticalFins.cpy : float
        Fin set local center of pressure y coordinate. Has units of length and
        is given in meters.
    EllipticalFins.cpz : float
        Fin set local center of pressure z coordinate. Has units of length and
        is given in meters.
    EllipticalFins.cl : Function
        Function which defines the lift coefficient as a function of the angle
        of attack and the Mach number. Takes as input the angle of attack in
        radians and the Mach number. Returns the lift coefficient.
    EllipticalFins.clalpha : float
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
            Number of fins, must be larger than 2.
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

        self.evaluate_geometrical_parameters()
        self.evaluate_center_of_pressure()
        self.evaluate_lift_coefficient()
        self.evaluate_roll_parameters()

        self.prints = _EllipticalFinsPrints(self)
        self.plots = _EllipticalFinsPlots(self)

    def evaluate_center_of_pressure(self):
        """Calculates and returns the center of pressure of the fin set in local
        coordinates. The center of pressure position is saved and stored as a
        tuple.

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

    def evaluate_geometrical_parameters(self):  # pylint: disable=too-many-statements
        """Calculates and saves fin set's geometrical parameters such as the
        fins' area, aspect ratio and parameters for roll movement.

        Returns
        -------
        None
        """

        # Compute auxiliary geometrical parameters
        # pylint: disable=invalid-name
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
        if self.span > self.rocket_radius:
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
        elif self.span < self.rocket_radius:
            roll_damping_interference_factor = 1 - (
                self.rocket_radius**2
                * (
                    2 * self.span**3
                    - np.pi * self.span**2 * self.rocket_radius
                    - 2 * self.span * self.rocket_radius**2
                    + np.pi * self.rocket_radius**3
                    + 2
                    * self.rocket_radius**2
                    * np.sqrt(-(self.span**2) + self.rocket_radius**2)
                    * np.arctan(
                        (self.span) / (np.sqrt(-(self.span**2) + self.rocket_radius**2))
                    )
                    - np.pi
                    * self.rocket_radius**2
                    * np.sqrt(-(self.span**2) + self.rocket_radius**2)
                )
            ) / (
                2
                * self.span
                * (-(self.span**2) + self.rocket_radius**2)
                * (self.span**2 / 3 + np.pi * self.span * self.rocket_radius / 4)
            )
        else:
            roll_damping_interference_factor = (28 - 3 * np.pi) / (4 + 3 * np.pi)

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
        # pylint: disable=invalid-name
        self.Af = Af  # Fin area
        self.AR = AR  # Fin aspect ratio
        self.gamma_c = gamma_c  # Mid chord angle
        self.Yma = Yma  # Span wise coord of mean aero chord
        self.roll_geometrical_constant = roll_geometrical_constant
        self.tau = tau
        self.lift_interference_factor = lift_interference_factor
        self.roll_damping_interference_factor = roll_damping_interference_factor
        self.roll_forcing_interference_factor = roll_forcing_interference_factor

        self.evaluate_shape()

    def evaluate_shape(self):
        angles = np.arange(0, 180, 5)
        x_array = self.root_chord / 2 + self.root_chord / 2 * np.cos(np.radians(angles))
        y_array = self.span * np.sin(np.radians(angles))
        self.shape_vec = [x_array, y_array]

    def info(self):
        self.prints.geometry()
        self.prints.lift()

    def all_info(self):
        self.prints.all()
        self.plots.all()

    def to_dict(self, **kwargs):
        data = super().to_dict(**kwargs)
        if kwargs.get("include_outputs", False):
            data.update(
                {
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
            span=data["span"],
            rocket_radius=data["rocket_radius"],
            cant_angle=data["cant_angle"],
            airfoil=data["airfoil"],
            name=data["name"],
        )
