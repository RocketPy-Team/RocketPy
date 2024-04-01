import numpy as np
from ..mathutils.vector_matrix import Matrix, Vector
from ..sensors.sensors import Sensors
from ..prints.sensors_prints import _AccelerometerPrints


class Accelerometer(Sensors):
    """
    Class for the accelerometer sensor
    """

    def __init__(
        self,
        sampling_rate,
        orientation=(0, 0, 0),
        measurement_range=np.inf,
        resolution=0,
        noise_density=0,
        random_walk=0,
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
            Sample rate of the sensor
        orientation : tuple, list, optional
            Orientation of the sensor in the rocket. The orientation can be
            given as:
            - A list of length 3, where the elements are the Euler angles for
              the rotation roll (φ), pitch (θ) and yaw (ψ) in radians. The
              standard rotation sequence is z-y-x (3-2-1) is used, meaning the
              sensor is first rotated by ψ around the z axis, then by θ around
              the new y axis and finally by φ around the new x axis.
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
        noise_density : float, optional
            The noise density of the sensor in m/s^2/√Hz. Sometimes called
            "white noise drift", "angular random walk" for gyroscopes, "velocity
            random walk" for the accelerometers or "(rate) noise density".
            Default is 0, meaning no noise is applied.
        random_walk : float, optional
            The random walk of the sensor in m/s^2/√Hz. Sometimes called "bias
            (in)stability" or "bias drift"". Default is 0, meaning no random
            walk is applied.
        constant_bias : float, optional
            The constant bias of the sensor in m/s^2. Default is 0, meaning no
            constant bias is applied.
        operating_temperature : float, optional
            The operating temperature of the sensor in degrees Celsius. At 25°C,
            the temperature bias and scale factor are 0. Default is 25.
        temperature_bias : float, optional
            The temperature bias of the sensor in m/s^2/°C. Default is 0,
            meaning no temperature bias is applied.
        temperature_scale_factor : float, optional
            The temperature scale factor of the sensor in %/°C. Default is 0,
            meaning no temperature scale factor is applied.
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
        """
        self.type = "Accelerometer"
        self.consider_gravity = consider_gravity
        self.prints = _AccelerometerPrints(self)
        super().__init__(
            sampling_rate,
            orientation,
            measurement_range=measurement_range,
            resolution=resolution,
            noise_density=noise_density,
            random_walk=random_walk,
            constant_bias=constant_bias,
            operating_temperature=operating_temperature,
            temperature_bias=temperature_bias,
            temperature_scale_factor=temperature_scale_factor,
            cross_axis_sensitivity=cross_axis_sensitivity,
            name=name,
        )

    def measure(self, t, u, u_dot, relative_position, gravity, *args):
        """
        Measure the acceleration of the rocket
        """
        # Linear acceleration of rocket cdm in inertial frame
        gravity = (
            Vector([0, 0, -gravity]) if self.consider_gravity else Vector([0, 0, 0])
        )
        a_I = Vector(u_dot[3:6]) + gravity

        # Vector from rocket cdm to sensor in rocket frame
        r = relative_position

        # Angular velocity and accel of rocket
        omega = Vector(u[10:13])
        omega_dot = Vector(u_dot[10:13])

        # Measured acceleration at sensor position in inertial frame
        A = (
            a_I
            + Vector.cross(omega_dot, r)
            + Vector.cross(omega, Vector.cross(omega, r))
        )
        # Transform to sensor frame
        inertial_to_sensor = self._total_rotation_matrix @ Matrix.transformation(
            u[6:10]
        )
        A = inertial_to_sensor @ A

        # Apply noise + bias and quatize
        A = self.apply_noise(A)
        A = self.apply_temperature_drift(A)
        A = self.quantize(A)

        self.measurement = tuple([*A])
        self.measured_values.append((t, *A))

    def export_measured_values(self, filename, format="csv"):
        """
        Export the measured values to a file

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
        if format == "csv":
            with open(filename, "w") as f:
                f.write("t,ax,ay,az\n")
                for t, ax, ay, az in self.measured_values:
                    f.write(f"{t},{ax},{ay},{az}\n")
        elif format == "json":
            import json

            data = {"t": [], "ax": [], "ay": [], "az": []}
            for t, ax, ay, az in self.measured_values:
                data["t"].append(t)
                data["ax"].append(ax)
                data["ay"].append(ay)
                data["az"].append(az)
            with open(filename, "w") as f:
                json.dump(data, f)
        else:
            raise ValueError("Invalid format")
