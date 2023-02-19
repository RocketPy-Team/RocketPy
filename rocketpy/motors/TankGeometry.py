import numpy as np

from rocketpy import Function
from rocketpy.Function import PiecewiseFunction, funcify_method
from functools import cached_property


class TankGeometry:
    def __init__(self, geometry_dict=dict()):
        self.geometry = geometry_dict

    @property
    def geometry(self):
        return self._geometry

    @geometry.setter
    def geometry(self, geometry_dict):
        self._geometry = dict()
        for domain, function in geometry_dict.items():
            self.add_geometry(domain, function)

    @funcify_method("height (m)", "radius (m)", extrapolation="zero")
    def radius(self):
        return self.radius

    @cached_property
    def average_radius(self):
        return self.radius.average(self.bottom, self.top)

    @property
    def bottom(self):
        return min(self._geometry.keys())[0]

    @property
    def top(self):
        return max(self._geometry.keys())[1]

    @cached_property
    def total_height(self):
        return self.top - self.bottom

    @funcify_method("height (m)", "area (m²)", extrapolation="zero")
    def area(self):
        return np.pi * self.radius**2

    @funcify_method("height (m)", "volume (m³)", extrapolation="zero")
    def volume(self):
        return self.area.integralFunction(self.bottom)

    @cached_property
    def total_volume(self):
        return self.volume(self.top)

    @funcify_method("volume (m³)", "height (m)", extrapolation="constant")
    def inverse_volume(self):
        return self.volume.inverseFunction(
            lambda v: v / (np.pi * self.average_radius**2),
        )

    def add_geometry(self, domain, function):
        self._geometry[domain] = Function(function)
        self.radius = PiecewiseFunction(self._geometry, "height (m)", "radius (m)")


class CylindricalTank(TankGeometry):
    def __init__(self, radius, height, spherical_caps=False, geometry_dict=dict()):
        super().__init__(geometry_dict)
        self.has_caps = False
        self.add_geometry((-height / 2, height / 2), radius)
        self.add_spherical_caps() if spherical_caps else None

    def add_spherical_caps(self):
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
    def __init__(self, radius, geometry_dict=dict()):
        super().__init__(geometry_dict)
        self.add_geometry((-radius, radius), lambda h: (radius**2 - h**2) ** 0.5)
