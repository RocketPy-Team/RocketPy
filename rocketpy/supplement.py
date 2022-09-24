import numpy as np
from abc import ABC, abstractmethod
from scipy import fsolve


class Geometry(ABC):
    def __init__(self):
        pass

    @property
    @abstractmethod
    def volume(self):
        pass

    @property
    @abstractmethod
    def centroid(self):
        pass

    @abstractmethod
    def volume_to_height(self):
        pass


class Geometry2D(Geometry):
    def __init__(self):
        pass

    @property
    @abstractmethod
    def area(self):
        pass

    @Geometry.volume.getter
    def volume(self):
        return 0

    @Geometry.centroid.getter
    def centroid(self):
        return 0

    def volume_to_height(self):
        return 0


class Geometry3D(Geometry):
    def __init__(self, filled_volume=None):
        self.filled_volume = filled_volume

    @property
    @abstractmethod
    def filled_height(self):
        pass

    @property
    @abstractmethod
    def filled_volume(self):
        pass

    @property
    @abstractmethod
    def filled_centroid(self):
        pass

    @Geometry.filled_volume.setter
    def filled_volume(self, volume):
        if volume:
            self._filled_volume = volume
            self._filled_height = self.volume_to_height(volume)
        else:
            self._filled_volume = 0
            self._filled_height = 0


class Disk(Geometry2D):
    def __init__(self, radius):
        self.radius = radius

    @Geometry2D.area.getter
    def area(self):
        return np.pi * self.radius**2


class Cylinder(Geometry3D):
    def __init__(self, radius, height, filled_volume=None):
        self.radius = radius
        self.height = height
        self.sectional_area = Disk(radius).area
        super().__init__(filled_volume)

    @Geometry.volume.getter
    def volume(self):
        return self.sectional_area * self.height

    @Geometry3D.filled_volume.getter
    def filled_volume(self):
        return self.sectional_area * self.filled_height

    @Geometry.centroid.getter
    def centroid(self):
        return self.height / 2

    @Geometry3D.filled_centroid.getter
    def filled_centroid(self):
        return self.filled_height / 2

    def volume_to_height(self, volume):
        return volume / self.sectional_area


class Hemisphere(Geometry3D):
    def __init__(self, radius, filled_volume=None):
        self.radius = radius
        super().__init__(filled_volume)

    @Geometry.volume.getter
    def volume(self):
        return 2 / 3 * np.pi * self.radius**3

    @Geometry3D.filled_volume.getter
    def filled_volume(self):
        return (
            np.pi * self.filled_height**2 * (3 * self.radius - self.filled_height) / 3
        )

    @Geometry.centroid.getter
    def centroid(self):
        return 0

    @Geometry3D.filled_centroid.getter
    def filled_centroid(self):
        # height from cap top pole
        # distance from center of sphere
        centroid = (
            3
            / 4
            * (2 * self.radius - self.filled_height) ** 2
            / (3 * self.radius - self.filled_height)
        )
        # distance from cap base
        return centroid - (self.radius - self.filled_height)

    def volume_to_height(self, volume):
        height = lambda height: volume - Geometry.spherical_cap_volume(
            self.radius, height
        )
        return fsolve(height, np.array([self.radius / 2]))[0]
