import numpy as np

from ..mathutils.vector_matrix import Matrix, Vector
from ..prints.sensors_prints import _InertialSensorPrints
from ..sensors.sensor import InertialSensor

# pylint: disable=too-many-arguments


class Accelerometer(InertialSensor):
    """Class for the accelerometer sensor

    Attributes
    ----------
    consider_gravity : bool
        Whether the sensor considers the effect of gravity on the acceleration.
    prints : _InertialSensorPrints
        Object that contains the print functions for the sensor.
    sampling_rate : float
        Sample rate of the sensor in Hz.
    orientation : tuple, list
        Orientation of the sensor in the rocket.
    measurement_range : float, tuple
        The measurement range of the sensor in m/s^2.
    resolution : float
        The resolution of the sensor in m/s^2/LSB.
    noise_density : float, list
        The noise density of the sensor in m/s^2/√Hz.
    noise_variance : float, list
        The variance of the noise of the sensor in (m/s^2)^2.
    random_walk_density : float, list
        The random walk density of the sensor in m/s^2/√Hz.
    random_walk_variance : float, list
        The variance of the random walk of the sensor in (m/s^2)^2.
    constant_bias : float, list
        The constant bias of the sensor in m/s^2.
    operating_temperature : float
        The operating temperature of the sensor in Kelvin.
    temperature_bias : float, list
        The temperature bias of the sensor in m/s^2/K.
    temperature_scale_factor : float, list
        The temperature scale factor of the sensor in %/K.
    cross_axis_sensitivity : float
        The cross axis sensitivity of the sensor in percentage.
    name : str
        The name of the sensor.
    rotation_sensor_to_body : Matrix
        The rotation matrix of the sensor from the sensor frame to the rocket
        frame of reference.
    normal_vector : Vector
        The normal vector of the sensor in the rocket frame of reference.
    _random_walk_drift : Vector
        The random walk drift of the sensor in m/s^2.
    measurement : float
        The measurement of the sensor after quantization, noise and temperature
        drift.
    measured_data : list
        The stored measured data of the sensor after quantization, noise and
        temperature drift.
    """

    units = "m/s^2"

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
        consider_gravity=False,
        name="Accelerometer",
    ):
        """
        Initialize the accelerometer sensor

        Parameters
        ----------
        sampling_rate : float
            Sample rate of the sensor in Hz.
        orientation : tuple, list, optional
            Orientation of the sensor in the rocket. The orientation can be
            given as either:

            - A list of length 3, where the elements are the Euler angles for
              the rotation yaw (ψ), pitch (θ) and roll (φ) in radians. The
              standard rotation sequence is z-y-x (3-2-1) is used, meaning the
              sensor is first rotated by ψ around the x axis, then by θ around
              the new y axis and finally by φ around the new z axis.
            - A list of lists (matrix) of shape 3x3, representing the rotation
              matrix from the sensor frame to the rocket frame. The sensor frame
              of reference is defined as to have z axis along the sensor's normal
              vector pointing upwards, x and y axes perpendicular to the z axis
              and each other.

            The rocket frame of reference is defined as to have z axis
            along the rocket's axis of symmetry pointing upwards, x and y axes
            perpendicular to the z axis and each other. A rotation around the x
            axis configures a pitch, around the y axis a yaw and around z axis a
            roll. Default is (0, 0, 0), meaning the sensor is aligned with all
            of the rocket's axis.
        measurement_range : float, tuple, optional
            The measurement range of the sensor in the m/s^2. If a float, the
            same range is applied both for positive and negative values. If a
            tuple, the first value is the positive range and the second value is
            the negative range. Default is np.inf.
        resolution : float, optional
            The resolution of the sensor in m/s^2/LSB. Default is 0, meaning no
            quantization is applied.
        noise_density : float, list, optional
            The noise density of the sensor for a Gaussian white noise in m/s^2/√Hz.
            Sometimes called "white noise drift", "angular random walk" for
            gyroscopes, "velocity random walk" for accelerometers or
            "(rate) noise density". Default is 0, meaning no noise is applied.
            If a float or int is given, the same noise density is applied to all
            axes. The values of each axis can be set individually by passing a
            list of length 3.
        noise_variance : float, list, optional
            The noise variance of the sensor for a Gaussian white noise in (m/s^2)^2.
            Default is 1, meaning the noise is normally distributed with a
            standard deviation of 1 m/s^2. If a float or int is given, the same
            variance is applied to all axes. The values of each axis can be set
            individually by passing a list of length 3.
        random_walk_density : float, list, optional
            The random walk of the sensor for a Gaussian random walk in m/s^2/√Hz.
            Sometimes called "bias (in)stability" or "bias drift"". Default is 0,
            meaning no random walk is applied. If a float or int is given, the
            same random walk is applied to all axes. The values of each axis can
            be set individually by passing a list of length 3.
        random_walk_variance : float, list, optional
            The random walk variance of the sensor for a Gaussian random walk in
            (m/s^2)^2. Default is 1, meaning the noise is normally distributed
            with a standard deviation of 1 m/s^2. If a float or int is given,
            the same variance is applied to all axes. The values of each axis
            can be set individually by passing a list of length 3.
        constant_bias : float, list, optional
            The constant bias of the sensor in m/s^2. Default is 0, meaning no
            constant bias is applied. If a float or int is given, the same bias
            is applied to all axes. The values of each axis can be set
            individually by passing a list of length 3.
        operating_temperature : float, optional
            The operating temperature of the sensor in Kelvin.
            At 298.15 K (25 °C), the sensor is assumed to operate ideally, no
            temperature related noise is applied. Default is 298.15.
        temperature_bias : float, list, optional
            The temperature bias of the sensor in m/s^2/K. Default is 0,
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
        consider_gravity : bool, optional
            If True, the sensor will consider the effect of gravity on the
            acceleration. Default is False.
        name : str, optional
            The name of the sensor. Default is "Accelerometer".

        Returns
        -------
        None

        See Also
        --------
        TODO link to documentation on noise model
        """
        super().__init__(
            sampling_rate,
            orientation,
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
            cross_axis_sensitivity=cross_axis_sensitivity,
            name=name,
        )
        self.consider_gravity = consider_gravity
        self.prints = _InertialSensorPrints(self)

    def measure(self, time, **kwargs):
        """Measure the acceleration of the rocket

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
        u_dot = kwargs["u_dot"]
        relative_position = kwargs["relative_position"]
        gravity = kwargs["environment"].gravity.get_value_opt(u[3])

        # Linear acceleration of rocket cdm in inertial frame
        gravity = (
            Vector([0, 0, -gravity]) if self.consider_gravity else Vector([0, 0, 0])
        )
        inertial_acceleration = Vector(u_dot[3:6]) + gravity

        # Vector from rocket cdm to sensor in rocket frame
        r = relative_position

        # Angular velocity and accel of rocket
        omega = Vector(u[10:13])
        omega_dot = Vector(u_dot[10:13])

        # Measured acceleration at sensor position in inertial frame
        A = (
            inertial_acceleration
            + Vector.cross(omega_dot, r)
            + Vector.cross(omega, Vector.cross(omega, r))
        )
        # Transform to sensor frame
        inertial_to_sensor = (
            self._total_rotation_sensor_to_body
            @ Matrix.transformation(u[6:10]).transpose
        )
        A = inertial_to_sensor @ A

        # Apply noise + bias and quantize
        A = self.apply_noise(A)
        A = self.apply_temperature_drift(A)
        A = self.quantize(A)

        self.measurement = tuple([*A])
        self._save_data((time, *A))

    def export_measured_data(self, filename, file_format="csv"):
        """Export the measured values to a file

        Parameters
        ----------
        filename : str
            Name of the file to export the values to
        file_format : str
            Format of the file to export the values to. Options are "csv" and
            "json". Default is "csv".

        Returns
        -------
        None
        """
        self._generic_export_measured_data(
            filename=filename,
            file_format=file_format,
            data_labels=("t", "ax", "ay", "az"),
        )

    def to_dict(self, **kwargs):
        data = super().to_dict(**kwargs)
        data.update({"consider_gravity": self.consider_gravity})
        return data

    @classmethod
    def from_dict(cls, data):
        return cls(
            sampling_rate=data["sampling_rate"],
            orientation=data["orientation"],
            measurement_range=data["measurement_range"],
            resolution=data["resolution"],
            noise_density=data["noise_density"],
            noise_variance=data["noise_variance"],
            random_walk_density=data["random_walk_density"],
            random_walk_variance=data["random_walk_variance"],
            constant_bias=data["constant_bias"],
            operating_temperature=data["operating_temperature"],
            temperature_bias=data["temperature_bias"],
            temperature_scale_factor=data["temperature_scale_factor"],
            cross_axis_sensitivity=data["cross_axis_sensitivity"],
            consider_gravity=data["consider_gravity"],
            name=data["name"],
        )
