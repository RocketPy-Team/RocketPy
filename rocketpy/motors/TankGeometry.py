__author__ = "Pedro Henrique Marinho Bressan, Mateus Stano Junqueira"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"

import numpy as np

from rocketpy.Function import Function, PiecewiseFunction, funcify_method

try:
    from functools import cache
except ImportError:
    from functools import lru_cache

    cache = lru_cache(maxsize=None)

try:
    from functools import cached_property
except ImportError:
    from rocketpy.tools import cached_property


class TankGeometry:
    """Class to define the geometry of a tank. It is used to calculate the
    geometrical properties such as volume, area and radius. The tank is
    axi-symmetric, and its geometry is defined by a set of Functions that are
    used to calculate the radius as a function of height.
    """

    def __init__(self, geometry_dict=dict()):
        """Initialize TankGeometry class.

        Parameters
        ----------
        geometry_dict : dict, optional
            Dictionary containing the geometry of the tank. The geometry is
            calculated by a PiecewiseFunction. Hence, the dict keys are disjoint
            tuples containing the lower and upper bounds of the domain of the
            corresponding Function, while the values correspond to the radius
            function from a axis of symmetry.
        """
        self.geometry = geometry_dict
        return None

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
        self._geometry = dict()
        for domain, function in geometry_dict.items():
            self.add_geometry(domain, function)

    @funcify_method("Height (m)", "radius (m)", extrapolation="zero")
    def radius(self):
        """
        The radius of the tank as a function of height.

        Returns
        -------
        rocketpy.Function
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

    @cached_property
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
        return self.area.integralFunction(self.bottom)

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
        rocketpy.Function
            Tank height as a function of volume.
        """
        return self.volume.inverseFunction(
            lambda v: v / (np.pi * self.average_radius**2),
        )

    @cache
    def volume_moment(self, lower, upper):
        """
        Calculates the first volume moment of the tank as a function of height.
        The first volume moment is used in the evaluation of the tank centroid,
        and can be understood as the weighted sum of the tank's infinitesimal
        slices volume by their height.

        The height referential is the zero level of the defined tank geometry,
        not to be confused with the tank bottom.

        See also:
        https://en.wikipedia.org/wiki/Moment_(physics)

        Returns
        -------
        rocketpy.Function
            Tank's first volume moment as a function of height.
        """
        height = self.area.identityFunction()

        # Tolerance of 1e-8 is used to avoid numerical errors
        upper = upper + 1e-12 if upper - lower < 1e-8 else upper

        volume_moment = (height * self.area).integralFunction(lower, upper)

        # Correct naming
        volume_moment.setInputs("Height (m)")
        volume_moment.setOutputs("Volume Moment (m⁴)")

        return volume_moment

    @cache
    def Ix_volume(self, lower, upper):
        """
        The volume of inertia of the tank with respect to
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
        rocketpy.Function
            Tank volume of inertia as a function of height.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/List_of_moments_of_inertia
        """
        height2 = self.radius.identityFunction() ** 2

        # Tolerance of 1e-8 is used to avoid numerical errors
        upper = upper + 1e-12 if upper - lower < 1e-8 else upper

        inertia = (self.area * (height2 + self.radius**2 / 4)).integralFunction(
            lower, upper
        )

        # Correct naming
        inertia.setInputs("Height (m)")
        inertia.setOutputs("Volume of inertia (m⁵)")

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
        rocketpy.Function
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
        rocketpy.Function
            Tank volume of inertia as a function of height.
        """
        # Tolerance of 1e-8 is used to avoid numerical errors
        upper = upper + 1e-12 if upper - lower < 1e-8 else upper

        inertia = (self.area * self.radius**2).integralFunction(lower, upper) / 2

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
        radius_function : rocketpy.Function, callable
            Function that defines the radius of the tank as a function of height.
        """
        self._geometry[domain] = Function(radius_function)
        self.radius = PiecewiseFunction(self._geometry, "Height (m)", "radius (m)")


class CylindricalTank(TankGeometry):
    """Class to define the geometry of a cylindrical tank."""

    def __init__(self, radius, height, spherical_caps=False, geometry_dict=dict()):
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
            If True, the tank will have spherical caps. The default is False.
        geometry_dict : dict, optional
            Dictionary containing the geometry of the tank. See TankGeometry.
        """
        super().__init__(geometry_dict)
        self.has_caps = False
        self.add_geometry((-height / 2, height / 2), radius)
        self.add_spherical_caps() if spherical_caps else None
        return None

    def add_spherical_caps(self):
        """
        Adds spherical caps to the tank. The caps are added at the bottom
        and at the top of the tank. If the tank already has caps, it raises a
        ValueError.
        """
        if not self.has_caps:
            radius = self.radius(0)
            height = self.total_height
            bottom_cap_range = (-height / 2 - radius, -height / 2)
            upper_cap_range = (height / 2, height / 2 + radius)
            bottom_cap_radius = lambda h: (radius**2 - (h + height / 2) ** 2) ** 0.5
            upper_cap_radius = lambda h: (radius**2 - (h - height / 2) ** 2) ** 0.5
            self.add_geometry(bottom_cap_range, bottom_cap_radius)
            self.add_geometry(upper_cap_range, upper_cap_radius)
            self.has_caps = True
        else:
            raise ValueError("Tank already has caps.")


class SphericalTank(TankGeometry):
    """Class to define the geometry of a spherical tank. This class inherits
    from TankGeometry."""

    def __init__(self, radius, geometry_dict=dict()):
        """Initialize SphericalTank class. The zero reference point of the
        sphere is its center (i.e. half of its height). Therefore, its height
        coordinate ranges between (-radius, radius).

        Parameters
        ----------
        radius : float
            Radius of the spherical tank.
        geometry_dict : dict, optional
            Dictionary containing the geometry of the tank. See TankGeometry.
        """
        super().__init__(geometry_dict)
        self.add_geometry((-radius, radius), lambda h: (radius**2 - h**2) ** 0.5)
        return None
