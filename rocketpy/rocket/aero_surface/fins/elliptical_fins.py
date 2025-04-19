import numpy as np

from rocketpy.plots.aero_surface_plots import _EllipticalFinsPlots
from rocketpy.prints.aero_surface_prints import _EllipticalFinsPrints
from rocketpy.rocket.aero_surface.fins._elliptical_mixin import _EllipticalMixin

from .fins import Fins


class EllipticalFins(_EllipticalMixin, Fins):
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

    def to_dict(self, include_outputs=False):
        data = super().to_dict(include_outputs)
        if include_outputs:
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
