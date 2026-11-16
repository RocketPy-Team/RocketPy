import numbers

import numpy as np

from rocketpy.mathutils.function import Function
from rocketpy.plots.aero_surface_plots import _TubeFinsPlots
from rocketpy.prints.aero_surface_prints import _TubeFinsPrints

from .aero_surface import AeroSurface


class TubeFins(AeroSurface):
    """Defines a symmetric set of tube fins.

    Each tube is treated as a ring wing on its own. Its normal force follows
    Ribner's result that a ring wing lifts twice as much as an elliptic flat
    wing spanning its diameter, and its center of pressure follows the wind
    tunnel measurements of NACA TN 4117. Both were obtained at low speed and
    are used unchanged at every Mach number. A roll rate is damped by the
    normal force it produces on each tube.

    The model is limited to uncanted tubes that touch both the rocket body and
    their two neighboring tubes. The effect the tubes have on each other, and
    the effect of the rocket body on them, are not accounted for.

    Parameters
    ----------
    n : int
        Number of tubes. Must be at least 3.
    length : int, float
        Tube length along the rocket axis, in meters.
    inner_radius : int, float
        Inner radius of each tube, in meters.
    outer_radius : int, float
        Outer radius of each tube, in meters. For the supported touching
        geometry, this must equal
        ``rocket_radius * sin(pi / n) / (1 - sin(pi / n))``.
    rocket_radius : int, float
        Radius of the rocket body where the tube fins are mounted, in meters.
    name : str, optional
        Name of the tube-fin set. Default is ``"Tube Fins"``.

    Notes
    -----
    Tube-fin friction and pressure drag are not calculated and must be
    included in the rocket's power-on and power-off drag curves. The normal
    force grows in proportion to the angle of attack, with no stall. Cant,
    separated tubes and overlapping tubes are not supported.

    See Also
    --------
    :doc:`Tube Fin Aerodynamics </technical/aerodynamics/tube_fins>`
    """

    # Center of pressure of a ring wing measured in NACA TN 4117 (figure 8):
    # distance from the leading edge, in tube lengths, against the ratio of
    # inner diameter to length
    _CP_ASPECT_RATIOS = (1 / 3, 2 / 3, 1.0, 1.5, 3.0)
    _CP_POSITIONS = (-0.11, 0.14, 0.20, 0.25, 0.35)

    def __init__(
        self,
        n,
        length,
        inner_radius,
        outer_radius,
        rocket_radius,
        name="Tube Fins",
    ):
        self._n = n
        self._length = length
        self._inner_radius = inner_radius
        self._outer_radius = outer_radius
        self._rocket_radius = rocket_radius

        self._validate_geometry()
        super().__init__(
            name=name,
            reference_area=np.pi * rocket_radius**2,
            reference_length=2 * rocket_radius,
        )

        self._evaluate_all()

        self.prints = _TubeFinsPrints(self)
        self.plots = _TubeFinsPlots(self)

    @staticmethod
    def _touching_outer_radius(n, rocket_radius):
        sin_half_angle = np.sin(np.pi / n)
        return rocket_radius * sin_half_angle / (1 - sin_half_angle)

    def _validate_geometry(self):
        if isinstance(self.n, bool) or not isinstance(self.n, numbers.Integral):
            raise ValueError("'n' must be an integer greater than or equal to 3.")
        if self.n < 3:
            raise ValueError("'n' must be greater than or equal to 3.")

        dimensions = {
            "length": self.length,
            "inner_radius": self.inner_radius,
            "outer_radius": self.outer_radius,
            "rocket_radius": self.rocket_radius,
        }
        for parameter, value in dimensions.items():
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise ValueError(f"'{parameter}' must be a positive real number.")
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"'{parameter}' must be finite and greater than zero.")

        if self.inner_radius >= self.outer_radius:
            raise ValueError("'inner_radius' must be smaller than 'outer_radius'.")

        touching_radius = self._touching_outer_radius(self.n, self.rocket_radius)
        tolerance = max(1e-12, touching_radius * 1e-6)
        if not np.isclose(
            self.outer_radius, touching_radius, rtol=1e-6, atol=tolerance
        ):
            geometry = (
                "separated" if self.outer_radius < touching_radius else "overlapping"
            )
            raise ValueError(
                f"The specified geometry produces {geometry} tube fins. "
                "Only mutually tangent tubes are supported; for "
                f"n={self.n} and rocket_radius={self.rocket_radius:g} m, "
                f"outer_radius must be {touching_radius:g} m."
            )

    def _evaluate_all(self):
        self.reference_area = np.pi * self.rocket_radius**2
        self.reference_length = 2 * self.rocket_radius
        self.evaluate_geometrical_parameters()
        self.evaluate_center_of_pressure()
        self.evaluate_lift_coefficient()
        self.evaluate_shape()

    def _set_geometry_attribute(self, attribute, value):
        old_value = getattr(self, attribute)
        setattr(self, attribute, value)
        try:
            self._validate_geometry()
        except (TypeError, ValueError):
            setattr(self, attribute, old_value)
            raise
        self._evaluate_all()

    @property
    def n(self):
        """Number of tubes in the set."""
        return self._n

    @n.setter
    def n(self, value):
        self._set_geometry_attribute("_n", value)

    @property
    def length(self):
        """Tube length along the rocket axis, in meters."""
        return self._length

    @length.setter
    def length(self, value):
        self._set_geometry_attribute("_length", value)

    @property
    def inner_radius(self):
        """Inner tube radius, in meters."""
        return self._inner_radius

    @inner_radius.setter
    def inner_radius(self, value):
        self._set_geometry_attribute("_inner_radius", value)

    @property
    def outer_radius(self):
        """Outer tube radius, in meters."""
        return self._outer_radius

    @outer_radius.setter
    def outer_radius(self, value):
        self._set_geometry_attribute("_outer_radius", value)

    @property
    def rocket_radius(self):
        """Reference rocket-body radius, in meters."""
        return self._rocket_radius

    @rocket_radius.setter
    def rocket_radius(self, value):
        self._set_geometry_attribute("_rocket_radius", value)

    @property
    def rocket_diameter(self):
        """Reference rocket-body diameter, in meters."""
        return 2 * self.rocket_radius

    def evaluate_geometrical_parameters(self):
        """Evaluate the ring wing aspect ratio, the tube spacing and the
        distance from the rocket axis to the center of each tube."""
        self.aspect_ratio = 2 * self.inner_radius / self.length
        self.touching_outer_radius = self._touching_outer_radius(
            self.n, self.rocket_radius
        )
        self.tube_separation = 2 * (self.touching_outer_radius - self.outer_radius)
        self.tube_center_radius = self.rocket_radius + self.outer_radius

    def evaluate_center_of_pressure(self):
        """Evaluate the center of pressure from the measurements of NACA
        TN 4117, interpolated on the ratio of inner diameter to length. Outside
        the measured range (1/3 to 3) the nearest measured value is used."""
        self.cpx = 0
        self.cpy = 0
        self.cpz = self.length * float(
            np.interp(self.aspect_ratio, self._CP_ASPECT_RATIOS, self._CP_POSITIONS)
        )
        self.cp = (self.cpx, self.cpy, self.cpz)

    def evaluate_lift_coefficient(self):
        """Evaluate the normal force coefficient derivative of the tube set.

        A ring wing lifts twice as much as an elliptic flat wing that spans its
        diameter and has the tube length as root chord. The lift slope of that
        wing comes from Helmbold's equation.
        """
        elliptic_aspect_ratio = 4 * self.aspect_ratio / np.pi
        elliptic_area = np.pi * self.inner_radius * self.length / 2
        elliptic_lift_slope = (
            2
            * np.pi
            * elliptic_aspect_ratio
            / (2 + np.sqrt(elliptic_aspect_ratio**2 + 4))
        )
        single_tube_constant = 2 * elliptic_lift_slope * elliptic_area
        clalpha_value = self.n * single_tube_constant / self.reference_area

        self.clalpha = Function(
            lambda mach: clalpha_value,
            "Mach",
            f"Lift coefficient derivative for {self.name}",
        )
        self.cl = Function(
            lambda alpha, mach: clalpha_value * alpha,
            ["Alpha (rad)", "Mach"],
            "Lift coefficient",
        )
        return self.cl

    def compute_forces_and_moments(
        self,
        stream_velocity,
        stream_speed,
        stream_mach,
        rho,
        cp,
        omega,
        *args,
    ):  # pylint: disable=arguments-differ
        """Computes the forces and moments acting on the tube fins.

        Parameters
        ----------
        stream_velocity : tuple of float
            The velocity of the airflow relative to the surface.
        stream_speed : float
            The magnitude of the airflow speed.
        stream_mach : float
            The Mach number of the airflow.
        rho : float
            Air density.
        cp : Vector
            Center of pressure coordinates in the body frame.
        omega : tuple[float, float, float]
            Tuple containing angular velocities around the x, y, z axes.

        Returns
        -------
        tuple of float
            The aerodynamic forces (lift, side_force, drag) and moments
            (pitch, yaw, roll) in the body frame.
        """
        R1, R2, R3, M1, M2, _ = super().compute_forces_and_moments(
            stream_velocity,
            stream_speed,
            stream_mach,
            rho,
            cp,
        )
        # A roll rate moves each tube sideways at omega[2] * tube_center_radius.
        # The tube sees this as an angle of attack, and the normal force it
        # produces acts against the roll.
        M3 = -(
            (1 / 2 * rho * stream_speed)
            * self.reference_area
            * self.clalpha.get_value_opt(stream_mach)
            * self.tube_center_radius**2
            * omega[2]
        )
        return R1, R2, R3, M1, M2, M3

    def evaluate_shape(self):
        """Store a side-view outline for plotting the tube-fin envelope."""
        lower = self.rocket_radius
        upper = self.rocket_radius + 2 * self.outer_radius
        self.shape_vec = [
            np.array([0, self.length, self.length, 0, 0]),
            np.array([lower, lower, upper, upper, lower]),
        ]

    def info(self):
        """Print tube-fin geometry and lift information."""
        self.prints.geometry()
        self.prints.lift()

    def all_info(self):
        """Print and plot all available tube-fin information."""
        self.prints.all()
        self.plots.all()

    def draw(self, *, filename=None):
        """Draw a side-view envelope of the tube-fin set."""
        return self.plots.draw(filename=filename)

    def to_dict(self, **kwargs):
        data = {
            "n": self.n,
            "length": self.length,
            "inner_radius": self.inner_radius,
            "outer_radius": self.outer_radius,
            "rocket_radius": self.rocket_radius,
            "name": self.name,
        }

        if kwargs.get("include_outputs", False):
            clalpha = self.clalpha
            cl = self.cl
            if kwargs.get("discretize", False):
                clalpha = clalpha.set_discrete(0, 4, 50, mutate_self=False)
                cl = cl.set_discrete(
                    (-np.pi / 6, 0), (np.pi / 6, 2), (10, 10), mutate_self=False
                )
            data.update(
                {
                    "aspect_ratio": self.aspect_ratio,
                    "cp": self.cp,
                    "clalpha": clalpha,
                    "cl": cl,
                    "reference_area": self.reference_area,
                    "reference_length": self.reference_length,
                }
            )

        return data

    @classmethod
    def from_dict(cls, data):
        return cls(
            n=data["n"],
            length=data["length"],
            inner_radius=data["inner_radius"],
            outer_radius=data["outer_radius"],
            rocket_radius=data["rocket_radius"],
            name=data["name"],
        )
