import numpy as np

from rocketpy.Function import PiecewiseFunction, funcify_method


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

    @property
    def radius(self):
        return self._radius

    @property
    def bottom(self):
        return min(self._geometry.keys())[0]

    @property
    def top(self):
        return max(self._geometry.keys())[1]

    @property
    def total_height(self):
        return self.top - self.bottom

    @property
    def area(self):
        return np.pi * self.radius**2

    @property
    def volume(self):
        return self.area.integralFunction(self.bottom)

    @property
    def total_volume(self):
        return self.volume(self.top)

    @property
    def inverse_volume(self):
        return self.volume.inverseFunction()

    def add_geometry(self, domain, function):
        self._geometry[domain] = function
        self._radius = PiecewiseFunction(self._geometry)


class CylindricalTank(TankGeometry):
    def __init__(self, radius, height, spherical_caps=False):
        super().__init__()
        self.add_geometry((-height / 2, height / 2), radius)
        self.add_spherical_caps() if spherical_caps else None

    def add_spherical_caps(self):
        radius = self.radius(0)
        bottom_cap_range = (-self.height / 2 - radius, -self.height / 2)
        upper_cap_range = (self.height / 2, self.height / 2 + radius)
        bottom_cap_radius = lambda h: (radius**2 - (h + self.height / 2) ** 2) ** 0.5
        upper_cap_radius = lambda h: (radius**2 - (h - self.height / 2) ** 2) ** 0.5
        self.add_geometry(bottom_cap_range, bottom_cap_radius)
        self.add_geometry(upper_cap_range, upper_cap_radius)


class SphericalTank(TankGeometry):
    def __init__(self, radius):
        super().__init__()
        self.add_geometry((-radius, radius), lambda h: (radius**2 - h**2) ** 0.5)
