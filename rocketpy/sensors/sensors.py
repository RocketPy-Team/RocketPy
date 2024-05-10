from abc import ABC, abstractmethod

import numpy as np

from rocketpy.mathutils.vector_matrix import Matrix, Vector


class Sensors(ABC):
    """Abstract class for sensors

    Attributes
    ----------
    type : str
        Type of the sensor (e.g. Accelerometer, Gyroscope).
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
        The operating temperature of the sensor in degrees Celsius.
    temperature_bias : float, list
        The temperature bias of the sensor in sensor units/°C.
    temperature_scale_factor : float, list
        The temperature scale factor of the sensor in %/°C.
    cross_axis_sensitivity : float
        The cross axis sensitivity of the sensor in percentage.
    name : str
        The name of the sensor.
    rotation_matrix : Matrix
        The rotation matrix of the sensor from the sensor frame to the rocket
        frame of reference.
    normal_vector : Vector
        The normal vector of the sensor in the rocket frame of reference.
    _random_walk_drift : Vector
        The random walk drift of the sensor in sensor units.
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
        operating_temperature=25,
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
            Orientation of the sensor in the rocket. The orientation can be
            given as:
            - A list of length 3, where the elements are the Euler angles for
              the rotation roll (φ), pitch (θ) and yaw (ψ) in radians. The
              standard rotation sequence is z-y-x (3-2-1) is used, meaning the
              sensor is first rotated by ψ around the z axis, then by θ around
              the new y axis and finally by φ around the new x axis.
              TODO: x and y are not defined in the rocket class. User has no
              way to know which axis is which.
            - A list of lists (matrix) of shape 3x3, representing the rotation
              matrix from the sensor frame to the rocket frame. The sensor frame
              of reference is defined as to have z axis along the sensor's normal
              vector pointing upwards, x and y axes perpendicular to the z axis
              and each other.
            The rocket frame of reference is defined as to have z axis
            along the rocket's axis of symmetry pointing upwards, x and y axes
            perpendicular to the z axis and each other. Default is (0, 0, 0),
            meaning the sensor is aligned with the rocket's axis of symmetry.
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
            The operating temperature of the sensor in degrees Celsius. At 25°C,
            the temperature bias and scale factor are 0. Default is 25.
        temperature_bias : float, list, optional
            The temperature bias of the sensor in sensor units/°C. Default is 0,
            meaning no temperature bias is applied. If a float or int is given,
            the same temperature bias is applied to all axes. The values of each
            axis can be set individually by passing a list of length 3.
        temperature_scale_factor : float, list, optional
            The temperature scale factor of the sensor in %/°C. Default is 0,
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
        self.sampling_rate = sampling_rate
        self.orientation = orientation
        self.resolution = resolution
        self.operating_temperature = operating_temperature
        self.noise_density = self._vectorize_input(noise_density, "noise_density")
        self.noise_variance = self._vectorize_input(noise_variance, "noise_variance")
        self.random_walk_density = self._vectorize_input(
            random_walk_density, "random_walk_density"
        )
        self.random_walk_variance = self._vectorize_input(
            random_walk_variance, "random_walk_variance"
        )
        self.constant_bias = self._vectorize_input(constant_bias, "constant_bias")
        self.temperature_bias = self._vectorize_input(
            temperature_bias, "temperature_bias"
        )
        self.temperature_scale_factor = self._vectorize_input(
            temperature_scale_factor, "temperature_scale_factor"
        )
        self.cross_axis_sensitivity = cross_axis_sensitivity
        self.name = name
        self._random_walk_drift = Vector([0, 0, 0])
        self.measurement = None
        self.measured_data = []
        self._counter = 0
        self._save_data = self._save_data_single

        # handle measurement range
        if isinstance(measurement_range, (tuple, list)):
            if len(measurement_range) != 2:
                raise ValueError("Invalid measurement range format")
            self.measurement_range = measurement_range
        elif isinstance(measurement_range, (int, float)):
            self.measurement_range = (-measurement_range, measurement_range)
        else:
            raise ValueError("Invalid measurement range format")

        # rotation matrix and normal vector
        if any(isinstance(row, (tuple, list)) for row in orientation):  # matrix
            self.rotation_matrix = Matrix(orientation)
        elif len(orientation) == 3:  # euler angles
            self.rotation_matrix = Matrix.transformation_euler_angles(
                *orientation
            ).round(12)
        else:
            raise ValueError("Invalid orientation format")
        self.normal_vector = Vector(
            [
                self.rotation_matrix[0][2],
                self.rotation_matrix[1][2],
                self.rotation_matrix[2][2],
            ]
        ).unit_vector

        # cross axis sensitivity matrix
        _cross_axis_matrix = 0.01 * Matrix(
            [
                [100, self.cross_axis_sensitivity, self.cross_axis_sensitivity],
                [self.cross_axis_sensitivity, 100, self.cross_axis_sensitivity],
                [self.cross_axis_sensitivity, self.cross_axis_sensitivity, 100],
            ]
        )

        # compute total rotation matrix given cross axis sensitivity
        self._total_rotation_matrix = self.rotation_matrix @ _cross_axis_matrix

        # map which rocket(s) the sensor is attached to and how many times
        self._attached_rockets = {}

    def __repr__(self):
        return f"{self.name}"

    def __call__(self, *args, **kwargs):
        return self.measure(*args, **kwargs)

    def _vectorize_input(self, value, name):
        if isinstance(value, (int, float)):
            return Vector([value, value, value])
        elif isinstance(value, (tuple, list)):
            return Vector(value)
        else:
            raise ValueError(f"Invalid {name} format")

    def _reset(self, simulated_rocket):
        """Reset the sensor data for a new simulation."""
        self._random_walk_drift = Vector([0, 0, 0])
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
    def measure(self, *args, **kwargs):
        pass

    @abstractmethod
    def export_measured_data(self):
        pass

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
        white_noise = (
            Vector(
                [np.random.normal(0, self.noise_variance[i] ** 0.5) for i in range(3)]
            )
            & self.noise_density
        ) * self.sampling_rate**0.5

        # random walk
        self._random_walk_drift = (
            self._random_walk_drift
            + (
                Vector(
                    [
                        np.random.normal(0, self.random_walk_variance[i] ** 0.5)
                        for i in range(3)
                    ]
                )
                & self.random_walk_density
            )
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
        value += (self.operating_temperature - 25) * self.temperature_bias
        # temperature scale factor
        scale_factor = (
            Vector([1, 1, 1])
            + (self.operating_temperature - 25) / 100 * self.temperature_scale_factor
        )
        return value & scale_factor
