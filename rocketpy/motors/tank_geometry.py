from functools import cached_property

import numpy as np

from ..mathutils.function import Function, funcify_method
from ..mathutils.piecewise_function import PiecewiseFunction
from ..plots.tank_geometry_plots import _TankGeometryPlots
from ..prints.tank_geometry_prints import _TankGeometryPrints

try:
    from functools import cache
except ImportError:
    from functools import lru_cache

    cache = lru_cache(maxsize=None)


class TankGeometry:
    """Class to define the geometry of a tank. It is used to calculate the
    geometrical properties such as volume, area and radius. The tank is
    axi-symmetric, and its geometry is defined by a set of Functions that
    are used to calculate the radius as a function of height.

    Attributes
    ----------
    TankGeometry.geometry : dict
        Dictionary containing the geometry of the tank. The dictionary
        keys are disjoint domains of the corresponding coordinates in
        meters on the TankGeometry symmetry axis. The dictionary values
        are rocketpy.Function objects that map the Tank height to its
        corresponding radius.
        As an example, `{ (-1,1): Function(lambda h: (1 - h**2) ** (1/2)) }`
        defines an spherical tank of radius 1.
    TankGeometry.radius : Function
        Piecewise defined radius in meters as a rocketpy.Function based
        on the TankGeometry.geometry dict.
    TankGeometry.average_radius : float
        The average radius in meters of the Tank radius. It is calculated
        as the average of the radius Function over the tank height.
    TankGeometry.bottom : float
        The bottom of the tank. It is the lowest coordinate that belongs to
        the domain of the geometry.
    TankGeometry.top : float
        The top of the tank. It is the highest coordinate that belongs to
        the domain of the geometry.
    TankGeometry.total_height : float
        The total height of the tank, in meters. It is calculated as the
        difference between the top and bottom coordinates.
    TankGeometry.area : Function
        Tank cross sectional area in meters squared as a function of height,
        defined as the area of a circle with radius TankGeometry.radius.
    TankGeometry.volume : Function
        Tank volume in in meters cubed as a function of height, defined as
        the Tank volume from the bottom to the given height.
    TankGeometry.total_volume : float
        Total volume of the tank, in meters cubed. It is calculated as the
        volume from the bottom to the top of the tank.
    TankGeometry.inverse_volume : Function
        Tank height as a function of volume, defined as the inverse of the
        TankGeometry.volume Function.
    """

    def __init__(self, geometry_dict=None):
        """Initialize TankGeometry class.

        Parameters
        ----------
        geometry_dict : Union[dict, None], optional
            Dictionary containing the geometry of the tank. The geometry is
            calculated by a PiecewiseFunction. Hence, the dict keys are disjoint
            tuples containing the lower and upper bounds of the domain of the
            corresponding Function, while the values correspond to the radius
            function from an axis of symmetry.
        """
        self.geometry = geometry_dict or {}

        # Initialize plots and prints object
        self.prints = _TankGeometryPrints(self)
        self.plots = _TankGeometryPlots(self)

    @property
    def geometry(self):
        """
        The dictionary containing the geometry of the tank.

        Returns
        -------
        dict
            Dictionary containing the geometry of the tank.
        """
        return self._geometry

    @geometry.setter
    def geometry(self, geometry_dict):
        """
        Sets the geometry of the tank.

        Parameters
        ----------
        geometry_dict : dict
            Dictionary containing the geometry of the tank.
        """
        self._geometry = {}
        for domain, function in geometry_dict.items():
            self.add_geometry(domain, function)

    @funcify_method("Height (m)", "radius (m)", extrapolation="zero")
    def radius(self):
        """
        The radius of the tank as a function of height.

        Returns
        -------
        Function
            Piecewise defined Function of tank radius.
        """
        return self.radius

    @cached_property
    def average_radius(self):
        """
        The average radius of the tank.

        Returns
        -------
        float
            Average radius of the tank.
        """
        return self.radius.average(self.bottom, self.top)

    @property
    def bottom(self):
        """
        The bottom of the tank. It is the lowest coordinate that belongs to
        the domain of the geometry.

        Returns
        -------
        float
            Bottom coordinate of the tank.
        """
        return min(self._geometry.keys())[0]

    @property
    def top(self):
        """
        The top of the tank. It is the highest coordinate that belongs to
        the domain of the geometry.

        Returns
        -------
        float
            Top coordinate of the tank.
        """
        return max(self._geometry.keys())[1]

    @property
    def total_height(self):
        """
        The total height of the tank, in meters.

        Returns
        -------
        float
            Total height of the tank.
        """
        return self.top - self.bottom

    @funcify_method("Height (m)", "Area (m²)", extrapolation="zero")
    def area(self):
        """
        The area of the tank cross section as a function of height.

        Returns
        -------
        Function
            Tank cross sectional area as a function of height.
        """
        return np.pi * self.radius**2

    @funcify_method("Height (m)", "Volume (m³)", extrapolation="zero")
    def volume(self):
        """
        The volume of the tank as a function of height.

        Returns
        -------
        Function
            Tank volume as a function of height.
        """
        return self.area.integral_function(self.bottom)

    @cached_property
    def total_volume(self):
        """
        The total volume of the tank.

        Returns
        -------
        float
            Total volume of the tank.
        """
        return self.volume(self.top)

    @funcify_method("Volume (m³)", "Height (m)", extrapolation="natural")
    def inverse_volume(self):
        """
        The height of the tank as a function of volume.

        Returns
        -------
        Function
            Tank height as a function of volume.
        """
        return self.volume.inverse_function(
            lambda v: v / (np.pi * self.average_radius**2),
        )

    @cache
    def volume_moment(self, lower, upper):
        """
        Calculates the first volume moment in m^4 of the tank as a function of
        height. The first volume moment is used in the evaluation of the tank
        centroid, and can be understood as the weighted sum of the tank's
        infinitesimal slices volume by their height.

        The height referential is the zero level of the defined tank geometry,
        not to be confused with the tank bottom.

        Returns
        -------
        Function
            Tank's first volume moment as a function of height.

        See Also
        --------
        `<https://en.wikipedia.org/wiki/Moment_(physics)#Examples/>`_
        """
        height = self.area.identity_function()

        # Tolerance of 1e-8 is used to avoid numerical errors
        upper = upper + 1e-12 if upper - lower < 1e-8 else upper

        volume_moment = (height * self.area).integral_function(lower, upper)

        # Correct naming
        volume_moment.set_inputs("Height (m)")
        volume_moment.set_outputs("Volume Moment (m⁴)")

        return volume_moment

    @cache
    def Ix_volume(self, lower, upper):
        """The volume of inertia of the tank in m^5 with respect to
        the x-axis as a function of height. The x direction is
        assumed to be perpendicular to the motor body axis.

        The inertia reference point is the zero level of the defined
        tank geometry, not to be confused with the tank bottom.

        Parameters
        ----------
        lower : float
            Lower bound of the domain where the volume of inertia is valid.
        upper : float
            Upper bound of the domain where the volume of inertia is valid.

        Returns
        -------
        Function
            Tank volume of inertia as a function of height.

        See Also
        --------
        https://en.wikipedia.org/wiki/List_of_moments_of_inertia
        """
        height2 = self.radius.identity_function() ** 2

        # Tolerance of 1e-8 is used to avoid numerical errors
        upper = upper + 1e-12 if upper - lower < 1e-8 else upper

        inertia = (self.area * (height2 + self.radius**2 / 4)).integral_function(
            lower, upper
        )

        # Correct naming
        inertia.set_inputs("Height (m)")
        inertia.set_outputs("Volume of inertia (m⁵)")

        return inertia

    @cache
    def Iy_volume(self, lower, upper):
        """
        The volume of inertia of the tank with respect to
        the y-axis as a function of height. The y direction is
        assumed to be perpendicular to the motor body axis.

        The inertia reference point is the zero level of the defined
        tank geometry, not to be confused with the tank bottom.

        Due to symmetry, this is the same as the Ix_volume.

        Returns
        -------
        Function
            Tank volume of inertia as a function of height.
        """
        return self.Ix_volume(lower, upper)

    @cache
    def Iz_volume(self, lower, upper):
        """
        The volume of inertia of the tank with respect to
        the z-axis as a function of height. The z direction is
        assumed to be parallel to the motor body axis.

        The inertia reference point is the zero level of the defined
        tank geometry, not to be confused with the tank bottom.

        Returns
        -------
        Function
            Tank volume of inertia as a function of height.
        """
        # Tolerance of 1e-8 is used to avoid numerical errors
        upper = upper + 1e-12 if upper - lower < 1e-8 else upper

        inertia = (self.area * self.radius**2).integral_function(lower, upper) / 2

        return inertia

    def add_geometry(self, domain, radius_function):
        """
        Adds a new geometry to the tank. The geometry is defined by a Function
        source, and a domain where it is valid.

        Parameters
        ----------
        domain : tuple
            Tuple containing the lower and upper bounds of the domain where the
            radius is valid.
        radius_function : Function, callable
            Function that defines the radius of the tank as a function of height.
        """
        self._geometry[domain] = Function(radius_function)
        self.radius = PiecewiseFunction(self._geometry, "Height (m)", "radius (m)")

    def to_dict(self, **kwargs):
        data = {
            "geometry": {
                str(domain): function.set_discrete(*domain, 50, mutate_self=False)
                if kwargs.get("discretize", False)
                else function
                for domain, function in self._geometry.items()
            }
        }

        if kwargs.get("include_outputs", False):
            data["outputs"] = {
                "average_radius": self.average_radius,
                "bottom": self.bottom,
                "top": self.top,
                "total_height": self.total_height,
                "total_volume": self.total_volume,
            }

        return data

    @classmethod
    def from_dict(cls, data):
        geometry_dict = {}
        # Reconstruct tuple keys
        for domain, radius_function in data["geometry"].items():
            domain = tuple(map(float, domain.strip("()").split(", ")))
            geometry_dict[domain] = radius_function
        return cls(geometry_dict)


class CylindricalTank(TankGeometry):
    """Class to define the geometry of a cylindrical tank. The cylinder has
    its zero reference point at its center (i.e. half of its height). This
    class inherits from the TankGeometry class. See the TankGeometry class
    for more information on its attributes and methods.
    """

    def __init__(self, radius, height, spherical_caps=False, geometry_dict=None):
        """Initialize CylindricalTank class. The zero reference point of the
        cylinder is its center (i.e. half of its height). Therefore the its
        height coordinate span is (-height/2, height/2).

        Parameters
        ----------
        radius : float
            Radius of the cylindrical tank, in meters.
        height : float
            Height of the cylindrical tank, in meters.
        spherical_caps : bool, optional
            If True, the tank will have spherical caps at the top and bottom
            with the same radius as the cylindrical part. If False, the tank
            will have flat caps at the top and bottom. Defaults to False.
        geometry_dict : Union[dict, None], optional
            Dictionary containing the geometry of the tank. See TankGeometry.
        """
        geometry_dict = geometry_dict or {}
        super().__init__(geometry_dict)
        self.__input_radius = radius
        self.height = height
        self.has_caps = False
        if spherical_caps:
            self.add_geometry((-height / 2 + radius, height / 2 - radius), radius)
            self.add_spherical_caps()
        else:
            self.add_geometry((-height / 2, height / 2), radius)

    def add_spherical_caps(self):
        """
        Adds spherical caps to the tank. The caps are added at the bottom
        and at the top of the tank with the same radius as the cylindrical
        part. The height is not modified, meaning that the total volume of
        the tank will decrease.
        """
        print(
            "Warning: Adding spherical caps to the tank will not modify the "
            + f"total height of the tank {self.height} m. "
            + "Its cylindrical portion height will be reduced to "
            + f"{self.height - 2 * self.__input_radius} m."
        )

        if not self.has_caps:
            radius = self.__input_radius
            height = self.height
            bottom_cap_range = (-height / 2, -height / 2 + radius)
            upper_cap_range = (height / 2 - radius, height / 2)

            def bottom_cap_radius(h):
                return abs(radius**2 - (h + (height / 2 - radius)) ** 2) ** 0.5

            def upper_cap_radius(h):
                return abs(radius**2 - (h - (height / 2 - radius)) ** 2) ** 0.5

            self.add_geometry(bottom_cap_range, bottom_cap_radius)
            self.add_geometry(upper_cap_range, upper_cap_radius)
            self.has_caps = True
        else:
            raise ValueError("Tank already has caps.")

    def to_dict(self, **kwargs):
        data = {
            "radius": self.__input_radius,
            "height": self.height,
            "spherical_caps": self.has_caps,
        }

        if kwargs.get("include_outputs", False):
            data.update(super().to_dict(**kwargs))

        return data

    @classmethod
    def from_dict(cls, data):
        return cls(data["radius"], data["height"], data["spherical_caps"])


class SphericalTank(TankGeometry):
    """Class to define the geometry of a spherical tank. The sphere zero
    reference point is its center (i.e. half of its height). This class
    inherits from the TankGeometry class. See the TankGeometry class for
    more information on its attributes and methods."""

    def __init__(self, radius, geometry_dict=None):
        """Initialize SphericalTank class. The zero reference point of the
        sphere is its center (i.e. half of its height). Therefore, its height
        coordinate ranges between (-radius, radius).

        Parameters
        ----------
        radius : float
            Radius of the spherical tank.
        geometry_dict : Union[dict, None], optional
            Dictionary containing the geometry of the tank. See TankGeometry.
        """
        geometry_dict = geometry_dict or {}
        super().__init__(geometry_dict)
        self.__input_radius = radius
        self.add_geometry((-radius, radius), lambda h: (radius**2 - h**2) ** 0.5)

    def to_dict(self, **kwargs):
        data = {"radius": self.__input_radius}

        if kwargs.get("include_outputs", False):
            data.update(super().to_dict(**kwargs))

        return data

    @classmethod
    def from_dict(cls, data):
        return cls(data["radius"])
