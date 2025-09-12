import numpy as np

from rocketpy.mathutils.function import Function
from rocketpy.plots.aero_surface_plots import _TailPlots
from rocketpy.prints.aero_surface_prints import _TailPrints

from .aero_surface import AeroSurface


class Tail(AeroSurface):
    """Class that defines a tail. Currently only accepts conical tails.

    Note
    ----
    Local coordinate system:
        - Z axis along the longitudinal axis of symmetry, positive downwards (top -> bottom).
        - Origin located at top of the tail (generally the portion closest to the rocket's nose).

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
        The reference rocket radius used for lift coefficient normalization in
        meters.
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
            Radius of the top of the tail. The top radius is defined as the
            radius of the transversal section that is closest to the rocket's
            nose.
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
        super().__init__(name, np.pi * rocket_radius**2, 2 * rocket_radius)

        self._top_radius = top_radius
        self._bottom_radius = bottom_radius
        self._length = length
        self._rocket_radius = rocket_radius

        self.evaluate_geometrical_parameters()
        self.evaluate_lift_coefficient()
        self.evaluate_center_of_pressure()

        self.plots = _TailPlots(self)
        self.prints = _TailPrints(self)

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

        Returns
        -------
        None
        """
        self.slant_length = np.sqrt(
            (self.length) ** 2 + (self.top_radius - self.bottom_radius) ** 2
        )
        self.surface_area = (
            np.pi * self.slant_length * (self.top_radius + self.bottom_radius)
        )
        self.evaluate_shape()

    def evaluate_shape(self):
        # Assuming the tail is a cone, calculate the shape vector
        self.shape_vec = [
            np.array([0, self.length]),
            np.array([self.top_radius, self.bottom_radius]),
        ]

    def evaluate_lift_coefficient(self):
        """Calculates and returns tail's lift coefficient.
        The lift coefficient is saved and returned. This function
        also calculates and saves its lift coefficient derivative.

        Returns
        -------
        None
        """
        # Calculate clalpha
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

    def evaluate_center_of_pressure(self):
        """Calculates and returns the center of pressure of the tail in local
        coordinates. The center of pressure position is saved and stored as a
        tuple.

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

    def info(self):
        self.prints.geometry()
        self.prints.lift()

    def all_info(self):
        self.prints.all()
        self.plots.all()

    def to_dict(self, **kwargs):
        data = {
            "top_radius": self._top_radius,
            "bottom_radius": self._bottom_radius,
            "length": self._length,
            "rocket_radius": self._rocket_radius,
            "name": self.name,
        }

        if kwargs.get("include_outputs", False):
            clalpha = self.clalpha
            cl = self.cl
            if kwargs.get("discretize", False):
                clalpha = clalpha.set_discrete(0, 4, 50)
                cl = cl.set_discrete(
                    (-np.pi / 6, 0), (np.pi / 6, 2), (10, 10), mutate_self=False
                )

            data.update(
                {
                    "cp": self.cp,
                    "clalpha": clalpha,
                    "cl": cl,
                    "slant_length": self.slant_length,
                    "surface_area": self.surface_area,
                }
            )

        return data

    @classmethod
    def from_dict(cls, data):
        return cls(
            top_radius=data["top_radius"],
            bottom_radius=data["bottom_radius"],
            length=data["length"],
            rocket_radius=data["rocket_radius"],
            name=data["name"],
        )
