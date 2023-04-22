import numpy as np

from rocketpy import Function
from rocketpy.Function import PiecewiseFunction, funcify_method
from functools import cached_property


class TankGeometry:
    """Class to define the geometry of a tank. It is used to calculate the
    its geometrical properties, such as volume, area and radius. The tank is
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

    @funcify_method("height (m)", "radius (m)", extrapolation="zero")
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

    @cached_property
    def total_height(self):
        """
        The total height of the tank.

        Returns
        -------
        float
            Total height of the tank.
        """
        return self.top - self.bottom

    @funcify_method("height (m)", "area (m²)", extrapolation="zero")
    def area(self):
        """
        The area of the tank cross section as a function of height.

        Returns
        -------
        Function
            Tank cross sectional area as a function of height.
        """
        return np.pi * self.radius**2

    @funcify_method("height (m)", "volume (m³)", extrapolation="zero")
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

    @funcify_method("volume (m³)", "height (m)", extrapolation="natural")
    def inverse_volume(self):
        """
        The height of the tank as a function of volume.

        Returns
        -------
        Function
            Tank height as a function of volume.
        """
        return self.volume.inverseFunction(
            lambda v: v / (np.pi * self.average_radius**2),
        )

    @funcify_method("height (m)", "balance (m⁴)")
    def balance(self):
        """
        The volume balance of the tank as a function of height.

        Returns
        -------
        Function
            Tank centroid as a function of height.
        """
        height = self.area.identityFunction()
        return (height * self.area).integralFunction()

    @funcify_method("height (m)", "volume of inertia (m⁵)")
    def Ix_volume(self):
        """
        The volume of inertia of the tank with respect to
        the x-axis as a function of height. The x direction is
        assumed to be perpendicular to the motor body axis.

        Returns
        -------
        Function
            Tank volume of inertia as a function of height.
        """
        height2 = self.radius.identityFunction() ** 2
        return (self.area * (height2 + self.radius**2 / 4)).integralFunction()

    @funcify_method("height (m)", "volume of inertia (m⁵)")
    def Iy_volume(self):
        """
        The volume of inertia of the tank with respect to
        the y-axis as a function of height. The y direction is
        assumed to be perpendicular to the motor body axis.

        Due to symmetry, this is the same as the Ix_volume.

        Returns
        -------
        Function
            Tank volume of inertia as a function of height.
        """
        return self.Ix_volume

    @funcify_method("height (m)", "volume of inertia (m⁵)")
    def Iz_volume(self):
        """
        The volume of inertia of the tank with respect to
        the z-axis as a function of height. The z direction is
        assumed to be parallel to the motor body axis.

        Returns
        -------
        Function
            Tank volume of inertia as a function of height.
        """
        return (self.area * self.radius**2).integralFunction() / 2

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
        self.radius = PiecewiseFunction(self._geometry, "height (m)", "radius (m)")


class CylindricalTank(TankGeometry):
    """Class to define the geometry of a cylindrical tank."""

    def __init__(self, radius, height, spherical_caps=False, geometry_dict=dict()):
        """Initialize CylindricalTank class. The zero reference point of the
        cylinder is its center (i.e. half of its height). Therefore the its
        height coordinate span is (-height/2, height/2).

        Parameters
        ----------
        radius : float
            Radius of the cylindrical tank.
        height : float
            Height of the cylindrical tank.
        spherical_caps : bool, optional
            If True, the tank will have spherical caps. The default is False.
        geometry_dict : dict, optional
            Dictionary containing the geometry of the tank. See TankGeometry.
        """
        super().__init__(geometry_dict)
        self.has_caps = False
        self.add_geometry((-height / 2, height / 2), radius)
        self.add_spherical_caps() if spherical_caps else None

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
    """Class to define the geometry of a spherical tank."""

    def __init__(self, radius, geometry_dict=dict()):
        """Initialize SphericalTank class. The zero reference point of the
        sphere is its center (i.e. half of its height). Therefore the its
        height coordinate span is (-radius, radius).

        Parameters
        ----------
        radius : float
            Radius of the spherical tank.
        geometry_dict : dict, optional
            Dictionary containing the geometry of the tank. See TankGeometry.
        """
        super().__init__(geometry_dict)
        self.add_geometry((-radius, radius), lambda h: (radius**2 - h**2) ** 0.5)
