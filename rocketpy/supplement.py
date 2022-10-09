import numpy as np
from abc import ABC, abstractmethod
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


class Disk(Geometry2D):
    def __init__(self, radius, **kwargs):
        self.radius = radius
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
