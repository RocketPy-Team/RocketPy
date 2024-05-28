import json
import math

import numpy as np

from ..mathutils.vector_matrix import Matrix, Vector
from ..prints.sensors_prints import _BarometerPrints, _GNSSPrints
from .sensors import ScalarSensors


class GNSS(ScalarSensors):
    units = "°"

    def __init__(
        self,
        sampling_rate,
        position_accuracy=0,
        altitude_accuracy=0,
        name="GNSS",
    ):
        """Initialize the GNSS sensor.

        Parameters
        ----------
        sampling_rate : float
            Sample rate of the sensor in Hz.
        position_accuracy : float
            The position accuracy of the sensor in meters. Default is 0.
        altitude_accuracy : float
            The altitude accuracy of the sensor in meters. Default is 0.
        name : str
            The name of the sensor. Default is "GNSS".
        """
        super().__init__(sampling_rate=sampling_rate, name=name)
        self.position_accuracy = position_accuracy
        self.altitude_accuracy = altitude_accuracy

        self.prints = _GNSSPrints(self)

    def measure(self, time, **kwargs):
        """Measure the position of the rocket in latitude, longitude and
        altitude.

        Parameters
        ----------
        time : float
            Current time in seconds.
        kwargs : dict
            Keyword arguments dictionary containing the following keys:
            - u : np.array
                State vector of the rocket.
            - u_dot : np.array
                Derivative of the state vector of the rocket.
            - relative_position : np.array
                Position of the sensor relative to the rocket center of mass.
            - environment : Environment
                Environment object containing the atmospheric conditions.
        """
        u = kwargs["u"]
        relative_position = kwargs["relative_position"]
        lat, lon = kwargs["environment"].latitude, kwargs["environment"].longitude
        earth_radius = kwargs["environment"].earth_radius

        # Get from state u and add relative position
        x, y, z = (Matrix.transformation(u[6:10]) @ relative_position) + Vector(u[0:3])
        altitude = z

        # Convert x and y to latitude and longitude
        lat1 = math.radians(lat)  # Launch lat point converted to radians
        lon1 = math.radians(lon)  # Launch lon point converted to radians
        drift = (x**2 + y**2) ** 0.5
        bearing = (2 * math.pi - math.atan2(-x, y)) * (180 / math.pi)

        # Applies the haversine equation to find final lat/lon coordinates
        latitude = math.degrees(
            math.asin(
                math.sin(lat1) * math.cos(drift / earth_radius)
                + math.cos(lat1)
                * math.sin(drift / earth_radius)
                * math.cos(math.radians(bearing))
            )
        )

        # Applies the haversine equation to find final lat/lon coordinates
        longitude = math.degrees(
            lon1
            + math.atan2(
                math.sin(math.radians(bearing))
                * math.sin(drift / earth_radius)
                * math.cos(lat1),
                math.cos(drift / earth_radius)
                - math.sin(lat1) * math.sin(math.radians(latitude)),
            )
        )

        latitude = np.random.normal(latitude, self.position_accuracy)
        longitude = np.random.normal(longitude, self.position_accuracy)
        altitude = np.random.normal(altitude, self.altitude_accuracy)

        self.measurement = (latitude, longitude, altitude)
        self._save_data((time, *self.measurement))

    def export_measured_data(self, filename, format):
        """Export the measured values to a file

        Parameters
        ----------
        filename : str
            Name of the file to export the values to
        format : str
            Format of the file to export the values to. Options are "csv" and
            "json". Default is "csv".

        Returns
        -------
        None
        """
        super().export_measured_data(
            filename=filename,
            format=format,
            data_labels=("t", "latitude", "longitude", "altitude"),
        )
