import numpy as np
from abc import ABC
from copy import copy
from scipy.optimize import fsolve


class TankGeometry(ABC):
    def __init__(
        self, radius, height=None, filled_volume=None, fill_direction="upwards"
    ):
        self.radius = radius
        self.height = height
        self.filled_volume = filled_volume
        self.fill_direction = fill_direction

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, radius_value):
        if radius_value > 0:
            self._radius = radius_value
        else:
            raise ValueError("Radius cannot be zero.")

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, height_value):
        if height_value is None:
            self._height = self._radius
        elif height_value >= 0:
            self._height = height_value
        else:
            raise ValueError("Tank characteristic height must be positive.")

    @property
    def volume(self):
        return 0

    @volume.setter
    def volume(self, volume_value):
        if volume_value > 0:
            self._volume = volume_value
        else:
            raise ValueError("Tank volume must be non-negative.")

    @property
    def centroid(self):
        return 0

    @property
    def filled_volume(self):
        return self._filled_volume

    @filled_volume.setter
    def filled_volume(self, volume):
        if not volume:
            self._filled_volume = 0
            self._filled_height = 0
        elif 0 < volume <= self.volume:
            self._filled_volume = volume
            self._filled_height = self.volume_to_height(volume)
        else:
            raise ValueError(
                "Filled volume cannot be negative nor greater than geometry's volume."
            )

    @property
    def filled_centroid(self):
        return 0

    @property
    def filled_height(self):
        return self._filled_height

    @property
    def empty_volume(self):
        return self.volume - self.filled_volume

    @property
    def empty_centroid(self):
        if self.empty_volume == 0:
            return self.height
        else:
            empty_region = copy(self)
            empty_region.reverse_fill()
            empty_region.filled_volume = self.empty_volume
            return self.height - empty_region.filled_centroid

    def reverse_fill(self):
        self.fill_direction = (
            "upwards" if self.fill_direction == "downwards" else "downwards"
        )

    def volume_to_height(self):
        return 0


class Disk(TankGeometry):
    def __init__(
        self, radius, height=None, filled_volume=None, fill_direction="upwards"
    ):
        self.height = height = 0
        super().__init__(radius, height, filled_volume, fill_direction)

    @property
    def area(self):
        return np.pi * self.radius**2


class Cylinder(TankGeometry):
    def __init__(
        self, radius, height=None, filled_volume=None, fill_direction="upwards"
    ):
        self.sectional_area = np.pi * radius**2
        super().__init__(radius, height, filled_volume, fill_direction)

    @TankGeometry.volume.getter
    def volume(self):
        return self.sectional_area * self.height

    @TankGeometry.centroid.getter
    def centroid(self):
        return self.height / 2

    @TankGeometry.filled_centroid.getter
    def filled_centroid(self):
        return self.filled_height / 2

    def volume_to_height(self, volume):
        return volume / self.sectional_area


class Hemisphere(TankGeometry):
    def __init__(
        self, radius, height=None, filled_volume=None, fill_direction="upwards"
    ):
        super().__init__(radius, height, filled_volume, fill_direction)

    @TankGeometry.volume.getter
    def volume(self):
        return 2 / 3 * np.pi * self.radius**3

    @TankGeometry.centroid.getter
    def centroid(self):
        return 0

    @TankGeometry.filled_centroid.getter
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
