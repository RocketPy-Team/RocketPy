import math

import numpy as np

from rocketpy.plots.aero_surface_plots import _TrapezoidalFinPlots
from rocketpy.prints.aero_surface_prints import _TrapezoidalFinPrints
from rocketpy.rocket.aero_surface.fins._trapezoidal_mixin import _TrapezoidalMixin

from .fin import Fin


class TrapezoidalFin(_TrapezoidalMixin, Fin):
    """A class used to represent a single trapezoidal fin. TODO: review this

    This class inherits from the Fin class.

    Note
    ----
    Local coordinate system:
        - Origin located at the top of the root chord.
        - Z axis along the longitudinal axis of symmetry, positive downwards (top -> bottom).
        - Y axis perpendicular to the Z axis, in the span direction, positive upwards.
        - X axis completes the right-handed coordinate system.

    See Also
    --------
    Fin : Parent class

    Attributes
    ----------
    TrapezoidalFin.angular_position : float
        Angular position of the fin set with respect to the rocket centerline, in
        degrees.
    TrapezoidalFin.rocket_radius : float
        The reference rocket radius used for lift coefficient normalization, in
        meters.
    TrapezoidalFin.airfoil : tuple
        Tuple of two items. First is the airfoil lift curve.
        Second is the unit of the curve (radians or degrees).
    TrapezoidalFin.cant_angle : float
        Fins cant angle with respect to the rocket centerline, in degrees.
    TrapezoidalFin.changing_attribute_dict : dict
        Dictionary that stores the name and the values of the attributes that
        may be changed during a simulation. Useful for control systems.
    TrapezoidalFin.cant_angle_rad : float
        Fins cant angle with respect to the rocket centerline, in radians.
    TrapezoidalFin.root_chord : float
        Fin root chord in meters.
    TrapezoidalFin.tip_chord : float
        Fin tip chord in meters.
    TrapezoidalFin.span : float
        Fin span in meters.
    TrapezoidalFin.name : string
        Name of fin set.
    TrapezoidalFin.sweep_length : float
        Fins sweep length in meters. By sweep length, understand the axial
        distance between the fin root leading edge and the fin tip leading edge
        measured parallel to the rocket centerline.
    TrapezoidalFin.sweep_angle : float
        Fins sweep angle with respect to the rocket centerline. Must
        be given in degrees.
    TrapezoidalFin.d : float
        Reference diameter of the rocket, in meters.
    TrapezoidalFins.fin_area : float
        Area of the longitudinal section of each fin in the set.
    TrapezoidalFins.f_ar : float
        Aspect ratio of each fin in the set
    TrapezoidalFin.gamma_c : float
        Fin mid-chord sweep angle.
    TrapezoidalFin.yma : float
        Span wise position of the mean aerodynamic chord.
    TrapezoidalFin.roll_geometrical_constant : float
        Geometrical constant used in roll calculations.
    TrapezoidalFin.tau : float
        Geometrical relation used to simplify lift and roll calculations.
    TrapezoidalFin.lift_interference_factor : float
        Factor of Fin-Body interference in the lift coefficient.
    TrapezoidalFin.cp : tuple
        Tuple with the x, y and z local coordinates of the fin set center of
        pressure. Has units of length and is given in meters.
    TrapezoidalFin.cpx : float
        Fin set local center of pressure x coordinate. Has units of length and
        is given in meters.
    TrapezoidalFin.cpy : float
        Fin set local center of pressure y coordinate. Has units of length and
        is given in meters.
    TrapezoidalFin.cpz : float
        Fin set local center of pressure z coordinate. Has units of length and
        is given in meters.
    """

    def __init__(
        self,
        angular_position,
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
        super().__init__(
            angular_position,
            root_chord,
            span,
            rocket_radius,
            cant_angle,
            airfoil,
            name,
        )

        self._initialize(sweep_length, sweep_angle, root_chord, tip_chord, span)
        self.evaluate_rotation_matrix()

        self.prints = _TrapezoidalFinPrints(self)
        self.plots = _TrapezoidalFinPlots(self)

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
        self.cpy = self.Yma
        self.cpz = cpz
        self.cp = (self.cpx, self.cpy, self.cpz)
