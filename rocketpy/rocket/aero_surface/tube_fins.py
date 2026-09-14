import numbers

import numpy as np

from rocketpy.mathutils.function import Function
from rocketpy.plots.aero_surface_plots import _TubeFinsPlots
from rocketpy.prints.aero_surface_prints import _TubeFinsPrints

from .aero_surface import AeroSurface


class TubeFins(AeroSurface):
    """Defines a symmetric set of tube fins for subsonic flight.

    The aerodynamic model follows the Ribner ring-airfoil normal-force
    derivative used by OpenRocket. It is limited to uncanted tube fins that
    touch both the rocket body and their two neighboring tubes. The center of
    pressure is fixed at the quarter chord, so the model is intended for
    Mach numbers up to 0.5.

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
    This model calculates normal force only. Tube-fin friction and pressure
    drag must be included in the rocket's power-on and power-off drag curves.
    Cant, roll, side-force, yaw, separated tubes, and overlapping tubes are not
    supported.
    """

    stall_angle = np.radians(20)

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
        """Evaluate the ring-airfoil aspect ratio and tube spacing."""
        self.aspect_ratio = 2 * self.inner_radius / self.length
        self.touching_outer_radius = self._touching_outer_radius(
            self.n, self.rocket_radius
        )
        self.tube_separation = 2 * (self.touching_outer_radius - self.outer_radius)

    def evaluate_center_of_pressure(self):
        """Set the subsonic center of pressure at the quarter chord."""
        self.cpx = 0
        self.cpy = 0
        self.cpz = self.length / 4
        self.cp = (self.cpx, self.cpy, self.cpz)

    def evaluate_lift_coefficient(self):
        """Evaluate the Ribner normal-force derivative for the tube set."""
        modified_aspect_ratio = 2 * self.aspect_ratio / np.pi
        single_tube_constant = (
            2
            * (modified_aspect_ratio / (1 + modified_aspect_ratio))
            * np.pi**2
            * self.inner_radius
            * self.length
        )
        clalpha_value = self.n * single_tube_constant / self.reference_area

        self.clalpha = Function(
            lambda mach: clalpha_value,
            "Mach",
            f"Lift coefficient derivative for {self.name}",
        )
        self.cl = Function(
            lambda alpha, mach: (
                self.clalpha(mach) * np.clip(alpha, -self.stall_angle, self.stall_angle)
            ),
            ["Alpha (rad)", "Mach"],
            "Lift coefficient",
        )
        return self.cl

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
                clalpha = clalpha.set_discrete(0, 0.5, 10, mutate_self=False)
                cl = cl.set_discrete(
                    (-self.stall_angle, 0),
                    (self.stall_angle, 0.5),
                    (10, 10),
                    mutate_self=False,
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
