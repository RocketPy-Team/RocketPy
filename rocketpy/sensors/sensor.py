import json
import warnings
from abc import ABC, abstractmethod

import numpy as np

from rocketpy.mathutils.vector_matrix import Matrix, Vector


# pylint: disable=too-many-statements
class Sensor(ABC):
    """Abstract class for sensors

    Attributes
    ----------
    sampling_rate : float
        Sample rate of the sensor in Hz.
    measurement_range : float, tuple
        The measurement range of the sensor in the sensor units.
    resolution : float
        The resolution of the sensor in sensor units/LSB.
    noise_density : float, list
        The noise density of the sensor in sensor units/√Hz.
    noise_variance : float, list
        The variance of the noise of the sensor in sensor units^2.
    random_walk_density : float, list
        The random walk density of the sensor in sensor units/√Hz.
    random_walk_variance : float, list
        The variance of the random walk of the sensor in sensor units^2.
    constant_bias : float, list
        The constant bias of the sensor in sensor units.
    operating_temperature : float
        The operating temperature of the sensor in Kelvin.
    temperature_bias : float, list
        The temperature bias of the sensor in sensor units/K.
    temperature_scale_factor : float, list
        The temperature scale factor of the sensor in %/K.
    name : str
        The name of the sensor.
    measurement : float
        The measurement of the sensor after quantization, noise and temperature
        drift.
    measured_data : list
        The stored measured data of the sensor after quantization, noise and
        temperature drift.
    """

    def __init__(
        self,
        sampling_rate,
        measurement_range=np.inf,
        resolution=0,
        noise_density=0,
        noise_variance=1,
        random_walk_density=0,
        random_walk_variance=1,
        constant_bias=0,
        operating_temperature=25,
        temperature_bias=0,
        temperature_scale_factor=0,
        name="Sensor",
    ):
        """
        Initialize the accelerometer sensor

        Parameters
        ----------
        sampling_rate : float
            Sample rate of the sensor
        measurement_range : float, tuple, optional
            The measurement range of the sensor in the sensor units. If a float,
            the same range is applied both for positive and negative values. If
            a tuple, the first value is the positive range and the second value
            is the negative range. Default is np.inf.
        resolution : float, optional
            The resolution of the sensor in sensor units/LSB. Default is 0,
            meaning no quantization is applied.
        noise_density : float, list, optional
            The noise density of the sensor for a Gaussian white noise in sensor
            units/√Hz. Sometimes called "white noise drift",
            "angular random walk" for gyroscopes, "velocity random walk" for
            accelerometers or "(rate) noise density". Default is 0, meaning no
            noise is applied.
        noise_variance : float, list, optional
            The noise variance of the sensor for a Gaussian white noise in
            sensor units^2. Default is 1, meaning the noise is normally
            distributed with a standard deviation of 1 unit.
        random_walk_density : float, list, optional
            The random walk density of the sensor for a Gaussian random walk in
            sensor units/√Hz. Sometimes called "bias (in)stability" or
            "bias drift". Default is 0, meaning no random walk is applied.
        random_walk_variance : float, list, optional
            The random walk variance of the sensor for a Gaussian random walk in
            sensor units^2. Default is 1, meaning the noise is normally
            distributed with a standard deviation of 1 unit.
        constant_bias : float, list, optional
            The constant bias of the sensor in sensor units. Default is 0,
            meaning no constant bias is applied.
        operating_temperature : float, optional
            The operating temperature of the sensor in Kelvin.
            At 298.15 K (25 °C), the sensor is assumed to operate ideally, no
            temperature related noise is applied. Default is 298.15.
        temperature_bias : float, list, optional
            The temperature bias of the sensor in sensor units/K. Default is 0,
            meaning no temperature bias is applied.
        temperature_scale_factor : float, list, optional
            The temperature scale factor of the sensor in %/K. Default is 0,
            meaning no temperature scale factor is applied.
        name : str, optional
            The name of the sensor. Default is "Sensor".

        Returns
        -------
        None

        See Also
        --------
        TODO link to documentation on noise model
        """
        warnings.warn(
            "The Sensor class (and all its subclasses) is still under "
            "experimental development. Some features may be changed in future "
            "versions, although we will try to keep the changes to a minimum.",
            UserWarning,
        )

        self.sampling_rate = sampling_rate
        self.resolution = resolution
        self.operating_temperature = operating_temperature
        self.noise_density = noise_density
        self.noise_variance = noise_variance
        self.random_walk_density = random_walk_density
        self.random_walk_variance = random_walk_variance
        self.constant_bias = constant_bias
        self.temperature_bias = temperature_bias
        self.temperature_scale_factor = temperature_scale_factor
        self.name = name
        self.measurement = None
        self.measured_data = []
        self._counter = 0
        self._save_data = self._save_data_single
        self._random_walk_drift = 0
        self.normal_vector = Vector([0, 0, 0])

        # handle measurement range
        if isinstance(measurement_range, (tuple, list)):
            if len(measurement_range) != 2:
                raise ValueError("Invalid measurement range format")
            self.measurement_range = measurement_range
        elif isinstance(measurement_range, (int, float)):
            self.measurement_range = (-measurement_range, measurement_range)
        else:
            raise ValueError("Invalid measurement range format")

        # map which rocket(s) the sensor is attached to and how many times
        self._attached_rockets = {}

    def __repr__(self):
        return f"{self.name}"

    def __call__(self, *args, **kwargs):
        return self.measure(*args, **kwargs)

    def _reset(self, simulated_rocket):
        """Reset the sensor data for a new simulation."""
        self._random_walk_drift = (
            Vector([0, 0, 0]) if isinstance(self._random_walk_drift, Vector) else 0
        )
        self.measured_data = []
        if self._attached_rockets[simulated_rocket] > 1:
            self.measured_data = [
                [] for _ in range(self._attached_rockets[simulated_rocket])
            ]
            self._save_data = self._save_data_multiple
        else:
            self._save_data = self._save_data_single

    def _save_data_single(self, data):
        """Save the measured data to the sensor data list for a sensor that is
        added only once to the simulated rocket."""
        self.measured_data.append(data)

    def _save_data_multiple(self, data):
        """Save the measured data to the sensor data list for a sensor that is
        added multiple times to the simulated rocket."""
        self.measured_data[self._counter].append(data)
        # counter for cases where the sensor is added multiple times in a rocket
        self._counter += 1
        if self._counter == len(self.measured_data):
            self._counter = 0

    @abstractmethod
    def measure(self, time, **kwargs):
        """Measure the sensor data at a given time"""

    @abstractmethod
    def quantize(self, value):
        """Quantize the sensor measurement"""

    @abstractmethod
    def apply_noise(self, value):
        """Add noise to the sensor measurement"""

    @abstractmethod
    def apply_temperature_drift(self, value):
        """Apply temperature drift to the sensor measurement"""

    @abstractmethod
    def export_measured_data(self, filename, file_format="csv"):
        """Export the measured values to a file"""

    def _generic_export_measured_data(self, filename, file_format, data_labels):
        """Export the measured values to a file given the data labels of each
        sensor.

        Parameters
        ----------
        sensor : Sensor
            Sensor object to export the measured values from.
        filename : str
            Name of the file to export the values to
        file_format : str
            file_format of the file to export the values to. Options are "csv"
            and "json". Default is "csv".
        data_labels : tuple
            Tuple of strings representing the labels for the data columns

        Returns
        -------
        None
        """
        if file_format.lower() not in ["json", "csv"]:
            raise ValueError("Invalid file_format")

        if file_format.lower() == "csv":
            # if sensor has been added multiple times to the simulated rocket
            if isinstance(self.measured_data[0], list):
                print("Data saved to", end=" ")
                for i, data in enumerate(self.measured_data):
                    with open(filename + f"_{i + 1}", "w") as f:
                        f.write(",".join(data_labels) + "\n")
                        for entry in data:
                            f.write(",".join(map(str, entry)) + "\n")
                    print(filename + f"_{i + 1},", end=" ")
            else:
                with open(filename, "w") as f:
                    f.write(",".join(data_labels) + "\n")
                    for entry in self.measured_data:
                        f.write(",".join(map(str, entry)) + "\n")
                print(f"Data saved to {filename}")
            return

        if file_format.lower() == "json":
            if isinstance(self.measured_data[0], list):
                print("Data saved to", end=" ")
                for i, data in enumerate(self.measured_data):
                    data_dict = {label: [] for label in data_labels}
                    for entry in data:
                        for label, value in zip(data_labels, entry):
                            data_dict[label].append(value)
                    with open(filename + f"_{i + 1}", "w") as f:
                        json.dump(data_dict, f)
                    print(filename + f"_{i + 1},", end=" ")
            else:
                data_dict = {label: [] for label in data_labels}
                for entry in self.measured_data:
                    for label, value in zip(data_labels, entry):
                        data_dict[label].append(value)
                with open(filename, "w") as f:
                    json.dump(data_dict, f)
                print(f"Data saved to {filename}")
            return

    # pylint: disable=unused-argument
    def to_dict(self, **kwargs):
        return {
            "sampling_rate": self.sampling_rate,
            "measurement_range": self.measurement_range,
            "resolution": self.resolution,
            "operating_temperature": self.operating_temperature,
            "noise_density": self.noise_density,
            "noise_variance": self.noise_variance,
            "random_walk_density": self.random_walk_density,
            "random_walk_variance": self.random_walk_variance,
            "constant_bias": self.constant_bias,
            "temperature_bias": self.temperature_bias,
            "temperature_scale_factor": self.temperature_scale_factor,
            "name": self.name,
        }


class InertialSensor(Sensor):
    """Model of an inertial sensor (accelerometer, gyroscope, magnetometer).
    Inertial sensors measurements are handled as vectors. The measurements are
    affected by the sensor's orientation in the rocket.

    Attributes
    ----------
    sampling_rate : float
        Sample rate of the sensor in Hz.
    orientation : tuple, list
        Orientation of the sensor in the rocket.
    measurement_range : float, tuple
        The measurement range of the sensor in the sensor units.
    resolution : float
        The resolution of the sensor in sensor units/LSB.
    noise_density : float, list
        The noise density of the sensor in sensor units/√Hz.
    noise_variance : float, list
        The variance of the noise of the sensor in sensor units^2.
    random_walk_density : float, list
        The random walk density of the sensor in sensor units/√Hz.
    random_walk_variance : float, list
        The variance of the random walk of the sensor in sensor units^2.
    constant_bias : float, list
        The constant bias of the sensor in sensor units.
    operating_temperature : float
        The operating temperature of the sensor in Kelvin.
    temperature_bias : float, list
        The temperature bias of the sensor in sensor units/K.
    temperature_scale_factor : float, list
        The temperature scale factor of the sensor in %/K.
    cross_axis_sensitivity : float
        The cross axis sensitivity of the sensor in percentage.
    name : str
        The name of the sensor.
    rotation_matrix : Matrix
        The rotation matrix of the sensor from the rocket frame to the sensor
        frame of reference.
    normal_vector : Vector
        The normal vector of the sensor in the rocket frame of reference.
    measurement : float
        The measurement of the sensor after quantization, noise and temperature
        drift.
    measured_data : list
        The stored measured data of the sensor after quantization, noise and
        temperature drift.
    """

    def __init__(
        self,
        sampling_rate,
        orientation=(0, 0, 0),
        measurement_range=np.inf,
        resolution=0,
        noise_density=0,
        noise_variance=1,
        random_walk_density=0,
        random_walk_variance=1,
        constant_bias=0,
        operating_temperature=298.15,
        temperature_bias=0,
        temperature_scale_factor=0,
        cross_axis_sensitivity=0,
        name="Sensor",
    ):
        """
        Initialize the accelerometer sensor

        Parameters
        ----------
        sampling_rate : float
            Sample rate of the sensor
        orientation : tuple, list, optional
            Orientation of the sensor in relation to the rocket frame of
            reference (Body Axes Coordinate System). See :ref:`rocket_axes` for
            more information.
            If orientation is not given, the sensor axes will be aligned with
            the rocket axis.
            The orientation can be given as either:

            - A list or tuple of length 3, where the elements are the intrinsic
              rotation angles in radians. The rotation sequence z-x-z (3-1-3) is
              used, meaning the sensor is first around the z axis (roll), then
              around the new x axis (pitch) and finally around the new z axis
              (roll).
            - A list of lists (matrix) of shape 3x3, representing the rotation
              matrix from the sensor frame to the rocket frame. The sensor frame
              of reference is defined as being initially aligned with the rocket
              frame of reference.
        measurement_range : float, tuple, optional
            The measurement range of the sensor in the sensor units. If a float,
            the same range is applied both for positive and negative values. If
            a tuple, the first value is the positive range and the second value
            is the negative range. Default is np.inf.
        resolution : float, optional
            The resolution of the sensor in sensor units/LSB. Default is 0,
            meaning no quantization is applied.
        noise_density : float, list, optional
            The noise density of the sensor for a Gaussian white noise in sensor
            units/√Hz. Sometimes called "white noise drift",
            "angular random walk" for gyroscopes, "velocity random walk" for
            accelerometers or "(rate) noise density". Default is 0, meaning no
            noise is applied. If a float or int is given, the same noise density
            is applied to all axes. The values of each axis can be set
            individually by passing a list of length 3.
        noise_variance : float, list, optional
            The noise variance of the sensor for a Gaussian white noise in
            sensor units^2. Default is 1, meaning the noise is normally
            distributed with a standard deviation of 1 unit. If a float or int
            is given, the same noise variance is applied to all axes. The values
            of each axis can be set individually by passing a list of length 3.
        random_walk_density : float, list, optional
            The random walk density of the sensor for a Gaussian random walk in
            sensor units/√Hz. Sometimes called "bias (in)stability" or
            "bias drift". Default is 0, meaning no random walk is applied.
            If a float or int is given, the same random walk is applied to all
            axes. The values of each axis can be set individually by passing a
            list of length 3.
        random_walk_variance : float, list, optional
            The random walk variance of the sensor for a Gaussian random walk in
            sensor units^2. Default is 1, meaning the noise is normally
            distributed with a standard deviation of 1 unit. If a float or int
            is given, the same random walk variance is applied to all axes. The
            values of each axis can be set individually by passing a list of
            length 3.
        constant_bias : float, list, optional
            The constant bias of the sensor in sensor units. Default is 0,
            meaning no constant bias is applied. If a float or int is given, the
            same constant bias is applied to all axes. The values of each axis
            can be set individually by passing a list of length 3.
        operating_temperature : float, optional
            The operating temperature of the sensor in Kelvin.
            At 298.15 K (25 °C), the sensor is assumed to operate ideally, no
            temperature related noise is applied. Default is 298.15.
        temperature_bias : float, list, optional
            The temperature bias of the sensor in sensor units/K. Default is 0,
            meaning no temperature bias is applied. If a float or int is given,
            the same temperature bias is applied to all axes. The values of each
            axis can be set individually by passing a list of length 3.
        temperature_scale_factor : float, list, optional
            The temperature scale factor of the sensor in %/K. Default is 0,
            meaning no temperature scale factor is applied. If a float or int is
            given, the same temperature scale factor is applied to all axes. The
            values of each axis can be set individually by passing a list of
            length 3.
        cross_axis_sensitivity : float, optional
            Skewness of the sensor's axes in percentage. Default is 0, meaning
            no cross-axis sensitivity is applied.
        name : str, optional
            The name of the sensor. Default is "Sensor".

        Returns
        -------
        None

        See Also
        --------
        TODO link to documentation on noise model
        """
        super().__init__(
            sampling_rate=sampling_rate,
            measurement_range=measurement_range,
            resolution=resolution,
            noise_density=self._vectorize_input(noise_density, "noise_density"),
            noise_variance=self._vectorize_input(noise_variance, "noise_variance"),
            random_walk_density=self._vectorize_input(
                random_walk_density, "random_walk_density"
            ),
            random_walk_variance=self._vectorize_input(
                random_walk_variance, "random_walk_variance"
            ),
            constant_bias=self._vectorize_input(constant_bias, "constant_bias"),
            operating_temperature=operating_temperature,
            temperature_bias=self._vectorize_input(
                temperature_bias, "temperature_bias"
            ),
            temperature_scale_factor=self._vectorize_input(
                temperature_scale_factor, "temperature_scale_factor"
            ),
            name=name,
        )

        self.orientation = orientation
        self.cross_axis_sensitivity = cross_axis_sensitivity
        self._random_walk_drift = Vector([0, 0, 0])

        # rotation matrix and normal vector
        if any(isinstance(row, (tuple, list)) for row in orientation):  # matrix
            self.rotation_sensor_to_body = Matrix(orientation)
        elif len(orientation) == 3:  # euler angles
            self.rotation_sensor_to_body = Matrix.transformation_euler_angles(
                *np.deg2rad(orientation)
            ).round(12)
        else:
            raise ValueError("Invalid orientation format")
        self.normal_vector = Vector(
            [
                self.rotation_sensor_to_body[0][2],
                self.rotation_sensor_to_body[1][2],
                self.rotation_sensor_to_body[2][2],
            ]
        ).unit_vector

        # cross axis sensitivity matrix
        cross_axis_matrix = 0.01 * Matrix(
            [
                [100, self.cross_axis_sensitivity, self.cross_axis_sensitivity],
                [self.cross_axis_sensitivity, 100, self.cross_axis_sensitivity],
                [self.cross_axis_sensitivity, self.cross_axis_sensitivity, 100],
            ]
        )

        # compute total rotation matrix given cross axis sensitivity
        self._total_rotation_sensor_to_body = (
            self.rotation_sensor_to_body @ cross_axis_matrix
        )

    def _vectorize_input(self, value, name):
        if isinstance(value, (int, float)):
            return Vector([value, value, value])
        elif isinstance(value, (tuple, list)):
            return Vector(value)
        else:
            raise ValueError(f"Invalid {name} format")

    def quantize(self, value):
        """
        Quantize the sensor measurement

        Parameters
        ----------
        value : float
            The value to quantize

        Returns
        -------
        float
            The quantized value
        """
        x = min(max(value.x, self.measurement_range[0]), self.measurement_range[1])
        y = min(max(value.y, self.measurement_range[0]), self.measurement_range[1])
        z = min(max(value.z, self.measurement_range[0]), self.measurement_range[1])
        if self.resolution != 0:
            x = round(x / self.resolution) * self.resolution
            y = round(y / self.resolution) * self.resolution
            z = round(z / self.resolution) * self.resolution
        return Vector([x, y, z])

    def apply_noise(self, value):
        """
        Add noise to the sensor measurement

        Parameters
        ----------
        value : float
            The value to add noise to

        Returns
        -------
        float
            The value with added noise
        """
        # white noise
        white_noise = Vector(
            [np.random.normal(0, self.noise_variance[i] ** 0.5) for i in range(3)]
        ) & (self.noise_density * self.sampling_rate**0.5)

        # random walk
        self._random_walk_drift = self._random_walk_drift + Vector(
            [np.random.normal(0, self.random_walk_variance[i] ** 0.5) for i in range(3)]
        ) & (self.random_walk_density / self.sampling_rate**0.5)

        # add noise
        value += white_noise + self._random_walk_drift + self.constant_bias

        return value

    def apply_temperature_drift(self, value):
        """
        Apply temperature drift to the sensor measurement

        Parameters
        ----------
        value : float
            The value to apply temperature drift to

        Returns
        -------
        float
            The value with applied temperature drift
        """
        # temperature drift
        value += (self.operating_temperature - 298.15) * self.temperature_bias
        # temperature scale factor
        scale_factor = (
            Vector([1, 1, 1])
            + (self.operating_temperature - 298.15)
            / 100
            * self.temperature_scale_factor
        )
        return value & scale_factor

    def to_dict(self, **kwargs):
        data = super().to_dict(**kwargs)
        data.update(
            {
                "orientation": self.orientation,
                "cross_axis_sensitivity": self.cross_axis_sensitivity,
            }
        )
        return data


class ScalarSensor(Sensor):
    """Model of a scalar sensor (e.g. Barometer). Scalar sensors are used
    to measure a single scalar value. The measurements are not affected by the
    sensor's orientation in the rocket.

    Attributes
    ----------
    sampling_rate : float
        Sample rate of the sensor in Hz.
    measurement_range : float, tuple
        The measurement range of the sensor in the sensor units.
    resolution : float
        The resolution of the sensor in sensor units/LSB.
    noise_density : float
        The noise density of the sensor in sensor units/√Hz.
    noise_variance : float
        The variance of the noise of the sensor in sensor units^2.
    random_walk_density : float
        The random walk density of the sensor in sensor units/√Hz.
    random_walk_variance : float
        The variance of the random walk of the sensor in sensor units^2.
    constant_bias : float
        The constant bias of the sensor in sensor units.
    operating_temperature : float
        The operating temperature of the sensor in Kelvin.
    temperature_bias : float
        The temperature bias of the sensor in sensor units/K.
    temperature_scale_factor : float
        The temperature scale factor of the sensor in %/K.
    name : str
        The name of the sensor.
    measurement : float
        The measurement of the sensor after quantization, noise and temperature
        drift.
    measured_data : list
        The stored measured data of the sensor after quantization, noise and
        temperature drift.
    """

    def __init__(
        self,
        sampling_rate,
        measurement_range=np.inf,
        resolution=0,
        noise_density=0,
        noise_variance=1,
        random_walk_density=0,
        random_walk_variance=1,
        constant_bias=0,
        operating_temperature=25,
        temperature_bias=0,
        temperature_scale_factor=0,
        name="Sensor",
    ):
        """
        Initialize the accelerometer sensor

        Parameters
        ----------
        sampling_rate : float
            Sample rate of the sensor
        measurement_range : float, tuple, optional
            The measurement range of the sensor in the sensor units. If a float,
            the same range is applied both for positive and negative values. If
            a tuple, the first value is the positive range and the second value
            is the negative range. Default is np.inf.
        resolution : float, optional
            The resolution of the sensor in sensor units/LSB. Default is 0,
            meaning no quantization is applied.
        noise_density : float, list, optional
            The noise density of the sensor for a Gaussian white noise in sensor
            units/√Hz. Sometimes called "white noise drift",
            "angular random walk" for gyroscopes, "velocity random walk" for
            accelerometers or "(rate) noise density". Default is 0, meaning no
            noise is applied.
        noise_variance : float, list, optional
            The noise variance of the sensor for a Gaussian white noise in
            sensor units^2. Default is 1, meaning the noise is normally
            distributed with a standard deviation of 1 unit.
        random_walk_density : float, list, optional
            The random walk density of the sensor for a Gaussian random walk in
            sensor units/√Hz. Sometimes called "bias (in)stability" or
            "bias drift". Default is 0, meaning no random walk is applied.
        random_walk_variance : float, list, optional
            The random walk variance of the sensor for a Gaussian random walk in
            sensor units^2. Default is 1, meaning the noise is normally
            distributed with a standard deviation of 1 unit.
        constant_bias : float, list, optional
            The constant bias of the sensor in sensor units. Default is 0,
            meaning no constant bias is applied.
        operating_temperature : float, optional
            The operating temperature of the sensor in Kelvin.
            At 298.15 K (25 °C), the sensor is assumed to operate ideally, no
            temperature related noise is applied. Default is 298.15.
        temperature_bias : float, list, optional
            The temperature bias of the sensor in sensor units/K. Default is 0,
            meaning no temperature bias is applied.
        temperature_scale_factor : float, list, optional
            The temperature scale factor of the sensor in %/K. Default is 0,
            meaning no temperature scale factor is applied.
        name : str, optional
            The name of the sensor. Default is "Sensor".

        Returns
        -------
        None

        See Also
        --------
        TODO link to documentation on noise model
        """
        super().__init__(
            sampling_rate=sampling_rate,
            measurement_range=measurement_range,
            resolution=resolution,
            noise_density=noise_density,
            noise_variance=noise_variance,
            random_walk_density=random_walk_density,
            random_walk_variance=random_walk_variance,
            constant_bias=constant_bias,
            operating_temperature=operating_temperature,
            temperature_bias=temperature_bias,
            temperature_scale_factor=temperature_scale_factor,
            name=name,
        )

    def quantize(self, value):
        """
        Quantize the sensor measurement

        Parameters
        ----------
        value : float
            The value to quantize

        Returns
        -------
        float
            The quantized value
        """
        value = min(max(value, self.measurement_range[0]), self.measurement_range[1])
        if self.resolution != 0:
            value = round(value / self.resolution) * self.resolution
        return value

    def apply_noise(self, value):
        """
        Add noise to the sensor measurement

        Parameters
        ----------
        value : float
            The value to add noise to

        Returns
        -------
        float
            The value with added noise
        """
        # white noise
        white_noise = (
            np.random.normal(0, self.noise_variance**0.5)
            * self.noise_density
            * self.sampling_rate**0.5
        )

        # random walk
        self._random_walk_drift = (
            self._random_walk_drift
            + np.random.normal(0, self.random_walk_variance**0.5)
            * self.random_walk_density
            / self.sampling_rate**0.5
        )

        # add noise
        value += white_noise + self._random_walk_drift + self.constant_bias

        return value

    def apply_temperature_drift(self, value):
        """
        Apply temperature drift to the sensor measurement

        Parameters
        ----------
        value : float
            The value to apply temperature drift to

        Returns
        -------
        float
            The value with applied temperature drift
        """
        # temperature drift
        value += (self.operating_temperature - 298.15) * self.temperature_bias
        # temperature scale factor
        scale_factor = (
            1
            + (self.operating_temperature - 298.15)
            / 100
            * self.temperature_scale_factor
        )
        value = value * scale_factor

        return value
