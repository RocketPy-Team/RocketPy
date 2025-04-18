import math

import numpy as np

from rocketpy.plots.aero_surface_plots import _TrapezoidalFinPlots
from rocketpy.prints.aero_surface_prints import _TrapezoidalFinPrints

from .fin import Fin


class TrapezoidalFin(Fin):
    """A class used to represent a single trapezoidal fin.

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

        # Check if sweep angle or sweep length is given
        if sweep_length is not None and sweep_angle is not None:
            raise ValueError("Cannot use sweep_length and sweep_angle together")
        elif sweep_angle is not None:
            sweep_length = math.tan(sweep_angle * math.pi / 180) * span
        elif sweep_length is None:
            sweep_length = root_chord - tip_chord
        else:  # Sweep length is given
            pass

        self._tip_chord = tip_chord
        self._sweep_length = sweep_length
        self._sweep_angle = sweep_angle

        self.evaluate_geometrical_parameters()
        self.evaluate_center_of_pressure()
        self.evaluate_lift_coefficient()
        self.evaluate_roll_parameters()
        self.evaluate_rotation_matrix()

        self.prints = _TrapezoidalFinPrints(self)
        self.plots = _TrapezoidalFinPlots(self)

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
        self._sweep_length = math.tan(value * math.pi / 180) * self.span
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
        self.cpy = self.Yma
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
