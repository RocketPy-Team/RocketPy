import warnings

import numpy as np

from rocketpy.plots.aero_surface_plots import _FreeFormFinPlots
from rocketpy.prints.aero_surface_prints import _FreeFormFinPrints
from rocketpy.rocket.aero_surface.fins._free_form_mixin import _FreeFormMixin

from .fins import Fins


class FreeFormFin(_FreeFormMixin, Fins):
    """Class that defines and holds information for a free form fin set.

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
    FreeFormFin.n : int
        Number of fins in fin set.
    FreeFormFin.rocket_radius : float
        The reference rocket radius used for lift coefficient normalization, in
        meters.
    FreeFormFin.airfoil : tuple
        Tuple of two items. First is the airfoil lift curve.
        Second is the unit of the curve (radians or degrees).
    FreeFormFin.cant_angle : float
        Fins cant angle with respect to the rocket centerline, in degrees.
    FreeFormFin.cant_angle_rad : float
        Fins cant angle with respect to the rocket centerline, in radians.
    FreeFormFin.root_chord : float
        Fin root chord in meters.
    FreeFormFin.span : float
        Fin span in meters.
    FreeFormFin.name : string
        Name of fin set.
    FreeFormFin.d : float
        Reference diameter of the rocket, in meters.
    FreeFormFin.ref_area : float
        Reference area of the rocket, in m².
    FreeFormFin.Af : float
        Area of the longitudinal section of each fin in the set.
    FreeFormFin.AR : float
        Aspect ratio of each fin in the set
    FreeFormFin.gamma_c : float
        Fin mid-chord sweep angle.
    FreeFormFin.Yma : float
        Span wise position of the mean aerodynamic chord.
    FreeFormFin.roll_geometrical_constant : float
        Geometrical constant used in roll calculations.
    FreeFormFin.tau : float
        Geometrical relation used to simplify lift and roll calculations.
    FreeFormFin.lift_interference_factor : float
        Factor of Fin-Body interference in the lift coefficient.
    FreeFormFin.cp : tuple
        Tuple with the x, y and z local coordinates of the fin set center of
        pressure. Has units of length and is given in meters.
    FreeFormFin.cpx : float
        Fin set local center of pressure x coordinate. Has units of length and
        is given in meters.
    FreeFormFin.cpy : float
        Fin set local center of pressure y coordinate. Has units of length and
        is given in meters.
    FreeFormFin.cpz : float
        Fin set local center of pressure z coordinate. Has units of length and
        is given in meters.
    FreeFormFin.cl : Function
        Function which defines the lift coefficient as a function of the angle
        of attack and the Mach number. Takes as input the angle of attack in
        radians and the Mach number. Returns the lift coefficient.
    FreeFormFin.clalpha : float
        Lift coefficient slope. Has units of 1/rad.
    FreeFormFin.mac_length : float
        Mean aerodynamic chord length of the fin set.
    FreeFormFin.mac_lead : float
        Mean aerodynamic chord leading edge x coordinate.
    """

    def __init__(
        self,
        n,
        shape_points,
        rocket_radius,
        cant_angle=0,
        airfoil=None,
        name="Fins",
    ):
        """Initialize FreeFormFin class.

        Parameters
        ----------
        n : int
            Number of fins, must be larger than 2.
        shape_points : list
            List of tuples (x, y) containing the coordinates of the fin's
            geometry defining points. The point (0, 0) is the root leading edge.
            Positive x is rearwards, positive y is upwards (span direction).
            The shape will be interpolated between the points, in the order
            they are given. The last point connects to the first point, and
            represents the trailing edge.
        rocket_radius : int, float
            Reference radius to calculate lift coefficient, in meters.
        cant_angle : int, float, optional
            Fins cant angle with respect to the rocket centerline. Must
            be given in degrees.
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
        root_chord, span = self._initialize(shape_points)

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

        self.prints = _FreeFormFinPrints(self)
        self.plots = _FreeFormFinPlots(self)

    def evaluate_center_of_pressure(self):
        """Calculates and returns the center of pressure of the fin set in local
        coordinates. The center of pressure position is saved and stored as a
        tuple.

        Returns
        -------
        None
        """
        # Center of pressure position in local coordinates
        cpz = self.mac_lead + 0.25 * self.mac_length
        self.cpx = 0
        self.cpy = self.Yma
        self.cpz = cpz
        self.cp = (self.cpx, self.cpy, self.cpz)
