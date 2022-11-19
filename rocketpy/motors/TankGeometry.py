import numpy as np
from copy import copy
from scipy.optimize import fsolve


class TankGeometry:
    """Class to evaluate mathematical and geometrical properties of a tank
    geometry, such as volume, mass, center of mass, inertia tensor, etc. The
    specific geometry is defined by its subclasses and may be used as the main
    body of the tank or its end caps.
    """

    def __init__(
        self, radius, height=None, filled_volume=None, fill_direction="upwards"
    ):
        """Initializes a tank geometry object.

        Parameters
        ----------
        radius : float
            Inner radius of the geometry, in meters.
        height : float, optional
            Height of the geometry, in meters. If not provided, is set as zero
            for two-dimenstional geometries or equal to the radius for tri-dimensional.
        filled_volume : float, optional
            Volume of the geometry that is filled with propellant, in meters cubed.
            If not provided, is set as zero.
        fill_direction : str, optional
            Direction of propellant filling. Relevant for tri-dimensional geometries
            which are not vertically simetrical. Can be either "upwards" or "downwards",
            as such:
                - "upwards": propellant is filled with the convex site facing down, as in
                a cup of water. Most used for bottom caps.
                - "downwards": propellant is filled with the concave site facing down, as
                in an inverted cup. Most used for upper caps.
        """
        self.radius = radius
        self.height = height
        self.filled_volume = filled_volume
        self.fill_direction = fill_direction

    @property
    def radius(self):
        """Returns the inner radius of the tank or its caps.

        Returns
        -------
        float
            Tanks radius.
        """
        return self._radius

    @radius.setter
    def radius(self, radius_value):
        """Sets the inner radius of the tank or its caps.
        Must be a positive value, raises an error otherwise.

        Parameters
        ----------
        radius_value : float
            New radius value to be set.

        Returns
        -------
        None
        """
        if radius_value > 0:
            self._radius = radius_value
        else:
            raise ValueError("Radius cannot be zero.")

    @property
    def height(self):
        """Returns the characteristic vertical height of the tank or caps. Is
        zero if the geometry is two dimensional.
        e.g. the height of a cylindrical tank or the radius of a spherical cap.

        Returns
        -------
        float
            Geometry characteristic height.
        """
        return self._height

    @height.setter
    def height(self, height_value):
        """Sets the characteristic vertical height of the tank or caps. Must be
        non-negative, raises an error otherwise.

        Parameters
        ----------
        height_value : float
            New height value to be set.

        Returns
        -------
        None
        """
        if height_value is None:
            self._height = self._radius
        elif height_value >= 0:
            self._height = height_value
        else:
            raise ValueError("Tank characteristic height must be positive.")

    @property
    def volume(self):
        """Returns the total volume of the geometry.

        Returns
        -------
        float
            Geometry's volume.
        """
        return 0

    @property
    def centroid(self):
        """Returns the centroid height of the geometry body, not considering
        filling.

        Returns
        -------
        float
            Geometry's centroid.
        """
        return 0

    @property
    def filled_volume(self):
        """Returns the volume of the partial geometry that is filled with
        propellant.

        Returns
        -------
        float
            Filled volume.
        """
        return self._filled_volume

    @filled_volume.setter
    def filled_volume(self, volume):
        """Sets the volume of the partial geometry that is filled with
        propellant. Must non-negative and smaller than the geometry's volume,
        raises an error otherwise. Also sets the filled height.

        Parameters
        ----------
        volume : float
            New filled volume value to be set.

        Returns
        -------
        None
        """
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
        """Returns the centroid height of filled portion of the geometry. The
        zero level reference is the bottom of the geometry (level that the filling
        begins).

        Returns
        -------
        float
            Filled centroid height.
        """
        return 0

    @property
    def filled_height(self):
        """Returns the height of the filled portion of the geometry, i.e., the
        height of the boundary between the filled portion and the gaseous portion.
        Generally considered the level of propellant. The zero level reference
        is the bottom of the geometry (level that the filling begins).

        Returns
        -------
        float
            Filled height.
        """
        return self._filled_height

    @property
    def empty_volume(self):
        """Returns the volume of the empty (or gaseous phase) portion of the
        geometry, i.e., ullage volume. Is zero if the tank is fully filled.

        Returns
        -------
        float
            Empty volume.
        """
        return self.volume - self._filled_volume

    @property
    def empty_height(self):
        """Returns the height of the empty (or gaseous phase) portion of the
        geometry. The zero level reference is the bottom of the geometry (level
        that the filling begins).

        Returns
        -------
        float
            Empty height.
        """
        return self.height - self._filled_height

    @property
    def empty_centroid(self):
        """Returns the centroid height of the empty (or gaseous phase) portion.
        The zero level reference is the bottom of the geometry (level that the
        filling begins).

        Returns
        -------
        None
        """
        if self.empty_volume == 0:
            return self.height
        else:
            empty_region = copy(self)
            empty_region.reverse_fill()
            empty_region.filled_volume = self.empty_volume
            return self.height - empty_region.filled_centroid

    @property
    def filled_inertia(self):
        """Returns the principal volumes of inertia of the filled portion with
        respect to the centroid the geometry. The z-axis is the direction of
        filling and it is perpendicular to the liquid level.
        Note: the volumes of inertia must be multiplied by the density of the
        fluid to get the actual moments of inertia.

        Returns
        -------
        tuple
            Principal volumes of inertia: Ixx, Iyy, Izz.
        """
        return 0, 0, 0

    @property
    def empty_inertia(self):
        """Returns the principal volumes of inertia of the empty portion with
        respect to the centroid the geometry. The z-axis is the direction of
        filling and it is perpendicular to the liquid level.
        Note: the volumes of inertia must be multiplied by the density of the
        fluid to get the actual moments of inertia.

        Returns
        -------
        tuple
            Principal volumes of inertia: Ixx, Iyy, Izz.
        """
        return 0, 0, 0

    @property
    def fill_direction(self):
        """Returns the direction of filling of the geometry. Can be either
        "upwards" or "downwards".

        Returns
        -------
        str
            Filling direction.
        """
        return self._fill_direction

    @fill_direction.setter
    def fill_direction(self, direction):
        """Sets the direction of filling of the geometry. Must be either
        "upwards" or "downwards", raises an error otherwise.

        Parameters
        ----------
        direction : str
            New filling direction value to be set.
        
        Returns
        -------
        None
        """
        if direction in ("upwards", "downwards"):
            self._fill_direction = direction
        else:
            raise AttributeError(
                f"""Filling direction '{direction}' is not recognized, must be either 
                'upwards' or 'downwards'."""
            )

    def reverse_fill(self):
        """Reverses the filling direction of the geometry. Useful for calculating
        the empty portion of the geometry.

        Returns
        -------
        None
        """
        self.fill_direction = (
            "upwards" if self.fill_direction == "downwards" else "downwards"
        )

    def volume_to_height(self):
        """Returns the filled height of the geometry for a given filled volume.
        The zero level reference is the bottom of the geometry (level that the
        filling begins).

        Returns
        -------
        float
            The height for the given volume.
        """
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

    @TankGeometry.filled_inertia.getter
    def filled_inertia(self):
        inertia_x = self.filled_volume * (
            self.radius**2 / 4 + self.filled_height**2 / 12
        )
        inertia_y = inertia_x
        inertia_z = self.filled_volume * self.radius**2 / 2
        return inertia_x, inertia_y, inertia_z

    @TankGeometry.empty_inertia.getter
    def empty_inertia(self):
        inertia_x = self.empty_volume * (
            self.radius**2 / 4 + self.empty_height**2 / 12
        )
        inertia_y = inertia_x
        inertia_z = self.empty_volume * self.radius**2 / 2
        return inertia_x, inertia_y, inertia_z

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
        if self.fill_direction == "downwards":
            centroid = 3 * self.radius / 8
        else:
            centroid = 5 * self.radius / 8

        return centroid

    @TankGeometry.filled_centroid.getter
    def filled_centroid(self):
        if self.fill_direction == "downwards":
            centroid = (
                0.75
                * (self.filled_height**3 - 2 * self.filled_height * self.radius**2)
                / (self.filled_height**2 - 3 * self.radius**2)
            )
        else:
            centroid = self.radius - (
                0.75
                * (2 * self.radius - self.filled_height) ** 2
                / (3 * self.radius - self.filled_height)
            )

        return centroid

    def __upwards_inertia(self, height):
        inertia_x = (
            np.pi
            * height**2
            * (
                -3 * height**3
                + 15 * height**2 * self.radius
                - 25 * height * self.radius**2
                + 15 * self.radius**3
            )
            / 15
        )
        inertia_y = inertia_x
        inertia_z = (
            np.pi
            * height**3
            * (3 * height**2 - 15 * height * self.radius + 20 * self.radius**2)
            / 30
        )
        return inertia_x, inertia_y, inertia_z

    def __downwards_inertia(self, height):
        inertia_x = np.pi * height**3 * (self.radius**2 / 3 - height**2 / 5)
        inertia_y = inertia_x
        inertia_z = (
            np.pi
            * height
            * (
                3 * height**4
                - 10 * height**2 * self.radius**2
                + 15 * self.radius**4
            )
            / 30
        )
        return inertia_x, inertia_y, inertia_z

    @TankGeometry.filled_inertia.getter
    def filled_inertia(self):
        if self.fill_direction == "downwards":
            inertia_x, inertia_y, inertia_z = self.__downwards_inertia(
                self.filled_height
            )
            # Steiner theorem to move inertia to the filled centroid
            inertia_x -= self.filled_volume * self.filled_centroid**2
            inertia_y = inertia_x
            return inertia_x, inertia_y, inertia_z
        else:
            inertia_x, inertia_y, inertia_z = self.__upwards_inertia(self.filled_height)
            # Steiner theorem to move inertia to the filled centroid
            inertia_x -= self.filled_volume * (self.filled_centroid - self.height) ** 2
            inertia_y = inertia_x
            return inertia_x, inertia_y, inertia_z

    @TankGeometry.empty_inertia.getter
    def empty_inertia(self):
        if self.fill_direction == "downwards":
            inertia_x, inertia_y, inertia_z = self.__upwards_inertia(self.empty_height)
            # Steiner theorem to move inertia to the empty centroid
            inertia_x -= self.empty_volume * self.empty_centroid**2
            inertia_y = inertia_x
            return inertia_x, inertia_y, inertia_z
        else:
            inertia_x, inertia_y, inertia_z = self.__downwards_inertia(
                self.empty_height
            )
            # Steiner theorem to move inertia to the empty centroid
            inertia_x -= self.empty_volume * (self.empty_centroid - self.height) ** 2
            inertia_y = inertia_x
            return inertia_x, inertia_y, inertia_z

    def volume_to_height(self, volume):
        if self.fill_direction == "downwards":
            height = (
                lambda height: volume
                - np.pi * height * (3 * self.radius**2 - height**2) / 3
            )
        else:
            height = (
                lambda height: volume
                - np.pi * height**2 * (3 * self.radius - height) / 3
            )

        return fsolve(height, np.array([self.radius / 2]))[0]
