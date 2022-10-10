import numpy as np
from abc import ABC, abstractmethod
from copy import copy
from scipy.optimize import fsolve


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

    @property
    @abstractmethod
    def filled_volume(self):
        pass

    @property
    @abstractmethod
    def empty_volume(self):
        pass

    @property
    @abstractmethod
    def empty_centroid(self):
        pass

    @property
    @abstractmethod
    def filled_centroid(self):
        pass

    @property
    @abstractmethod
    def filled_height(self):
        pass

    @abstractmethod
    def volume_to_height(self):
        pass


class Geometry2D(Geometry):
    def __init__(self, **kwargs):
        self.fill_direction = None
        pass

    @property
    @abstractmethod
    def area(self):
        pass

    @property
    def volume(self):
        return 0

    @property
    def centroid(self):
        return 0

    @property
    def filled_volume(self):
        return 0

    @filled_volume.setter
    def filled_volume(self, value):
        pass

    @property
    def empty_volume(self):
        return 0

    @property
    def empty_centroid(self):
        return 0

    @property
    def filled_centroid(self):
        return 0

    @property
    def filled_height(self):
        return 0

    def volume_to_height(self):
        return 0


class Geometry3D(Geometry):
    def __init__(self, filled_volume=None, fill_direction="upwards"):
        self.filled_volume = filled_volume
        self.fill_direction = fill_direction

    @property
    def filled_height(self):
        return self._filled_height

    @property
    def filled_volume(self):
        return self._filled_volume

    @filled_volume.setter
    def filled_volume(self, volume):
        if volume:
            self._filled_volume = volume
            self._filled_height = self.volume_to_height(volume)
        else:
            self._filled_volume = 0
            self._filled_height = 0

    @property
    def empty_volume(self):
        return self.volume - self._filled_volume

    @property
    def empty_centroid(self):
        if self.empty_volume == 0:
            return 0
        else:
            empty_region = copy(self)
            empty_region.reverse_fill()
            empty_region.filled_volume = self.empty_volume
            return self.height - empty_region.filled_centroid

    def reverse_fill(self):
        self.fill_direction = (
            "upwards" if self.fill_direction == "downwards" else "downwards"
        )


class Disk(Geometry2D):
    def __init__(self, radius, **kwargs):
        self.radius = radius
        self.height = 0
        super().__init__(**kwargs)

    @property
    def area(self):
        return np.pi * self.radius**2


class Cylinder(Geometry3D):
    def __init__(self, radius, height, filled_volume=None, fill_direction="upwards"):
        self.radius = radius
        self.height = height
        self.sectional_area = Disk(radius).area
        super().__init__(filled_volume, fill_direction)

    @property
    def volume(self):
        return self.sectional_area * self.height

    @property
    def centroid(self):
        return self.height / 2

    @property
    def filled_centroid(self):
        return self.filled_height / 2

    def volume_to_height(self, volume):
        return volume / self.sectional_area


class Hemisphere(Geometry3D):
    def __init__(self, radius, filled_volume=None, fill_direction="upwards"):
        self.radius = radius
        self.height = radius
        self.fill_direction = fill_direction
        super().__init__(filled_volume, fill_direction)

    @property
    def volume(self):
        return 2 / 3 * np.pi * self.radius**3

    @property
    def centroid(self):
        return 0

    @property
    def filled_centroid(self):
        if self.fill_direction == "upwards":
            centroid = self.radius - (
                0.75
                * (2 * self.radius - self.filled_height) ** 2
                / (3 * self.radius - self.filled_height)
            )
        elif self.fill_direction == "downwards":
            centroid = (
                0.75
                * (self.filled_height**3 - 2 * self.filled_height * self.radius**2)
                / (self.filled_height**2 - 3 * self.radius**2)
            )
        else:
            raise AttributeError("Input a valid fill_direction")

        return centroid

    def volume_to_height(self, volume):
        if self.fill_direction == "upwards":
            height = (
                lambda height: volume
                - np.pi * height**2 * (3 * self.radius - height) / 3
            )
        elif self.fill_direction == "downwards":
            height = (
                lambda height: volume
                - np.pi * height * (3 * self.radius**2 - height**2) / 3
            )
        else:
            raise AttributeError("Input a valid fill_direction")

        return fsolve(height, np.array([self.radius / 2]))[0]
