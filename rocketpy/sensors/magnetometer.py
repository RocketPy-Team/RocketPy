import math
from datetime import datetime

import numpy as np
from pywmm import WMMv2
from pywmm.calculator import calculate_geomagnetic
from pywmm.date_utils import decimal_year

from rocketpy.exceptions import InvalidParameterError
from rocketpy.mathutils.vector_matrix import Matrix, Vector
from rocketpy.prints.sensors_prints import _InertialSensorPrints
from rocketpy.rocket import Rocket
from rocketpy.sensors.sensor import InertialSensor
from rocketpy.tools import inverted_haversine


class Magnetometer(InertialSensor):
    """Class for simulating a magnetometer sensor during rocket flight.

    Inherits from the InertialSensor subclass. Models Earth's geomagnetic
    field using the World Magnetic Model (WMM) and simulates physical sensor
    distortions, including hard iron, soft iron (via plates), power wire
    interferences (via communication & ignition wires), thermal drifts, noise, and
    quantization.

    Attributes
    ----------
    sampling_rate : float
        Sampling rate of the sensor in Hz.
    orientation : tuple, list
        Orientation of the sensor in the rocket.
    magnetic_interference : list
        Magnetic interference on the magnetometer in Tesla (T), formed by the sum of
        hard iron distortion, power interference, and soft iron distortion difference.
    hard_iron_distortion : list
        Hard iron distortion in Tesla (T).
    _soft_iron_distortion_matrix : Matrix
        Soft iron distortion matrix applied to the magnetic reading (dimensionless).
    soft_iron_distortion_difference : list
        Difference between the vector after application of soft iron distortion
        and before in Tesla (T).
    power_interference : list
        Total magnetic distortion due to system interference in Tesla (T).
    communications_interference : list
        Magnetic distortion due to communication wires in Tesla (T).
    activation_signal_interference : list
        Magnetic distortion due to ignition/activation signals in Tesla (T).
    measurement_range : float or tuple
        The measurement range of the sensor in Tesla (T).
    resolution : float
        The resolution of the sensor in T/LSB.
    noise_density : float, list
        The noise density of the sensor in T/√Hz.
    noise_variance : float, list
        The variance of the noise of the sensor in T^2.
    random_walk_density : float, list
        The random walk density of the sensor in T/√Hz.
    random_walk_variance : float, list
        The variance of the random walk of the sensor in T^2.
    constant_bias : float, list
        The constant bias of the sensor in Tesla (T).
    operating_temperature : float
        The operating temperature of the sensor in Kelvin (K).
    temperature_bias : float, list
        The temperature bias of the sensor in T/K.
    temperature_scale_factor : float, list
        The temperature scale factor of the sensor in %/K.
    cross_axis_sensitivity : float
        The cross axis sensitivity of the sensor in percentage (%).
    name : str
        The name of the sensor.
    _random_walk_drift : Vector
        The random walk drift of the sensor in Tesla (T).
    measurement : tuple
        The measurement of the sensor after quantization, noise, temperature
        drift, and magnetic interference in Tesla (T).
    measured_data : list
        The stored measured data of the sensor after quantization, noise, and
        temperature drift.
    wmm : WMMv2
        WMMv2 object from the pywmm library. Details on its GitHub repository:
        https://github.com/dougc95/pywmm/tree/main
    year : float
        Current decimal year.
    rotation_sensor_to_body : Matrix
        The rotation matrix of the sensor from the sensor frame to the rocket
        frame of reference.
    normal_vector : Vector
        The normal vector of the sensor in the rocket frame of reference.
    """

    units = "T"

    def __init__(  # pylint: disable=too-many-arguments
        self,
        sampling_rate,
        orientation=(0, 0, 0),
        measurement_range=np.inf,
        resolution=0,
        hard_iron_distortion=0,
        soft_iron_distortion=Matrix.identity(),
        power_interference=0,
        activation_signal_interference=None,
        communications_interference=None,
        noise_density=0,
        noise_variance=1,
        random_walk_density=0,
        random_walk_variance=1,
        constant_bias=0,
        operating_temperature=298,
        temperature_bias=0,
        temperature_scale_factor=0,
        cross_axis_sensitivity=0,
        name="Magnetometer",
        seed=None,
    ) -> None:
        """Initializes the Magnetometer sensor:

        Parameters
        ----------
        sampling_rate : float
            Sampling rate of the sensor in Hz.
        orientation : tuple, list, optional
            Orientation of the sensor in the rocket. The orientation can be
            given as either:

            - A list of length 3, where the elements are the Euler angles for
              the rotation yaw (ψ), pitch (θ), and roll (φ) in radians. The
              standard rotation sequence z-y-x (3-2-1) is used, meaning the
              sensor is first rotated by ψ around the z axis, then by θ around
              the new y axis, and finally by φ around the new x axis.
            - A list of lists (matrix) of shape 3x3, representing the rotation
              matrix from the sensor frame to the rocket frame. The sensor frame
              of reference is defined as having the z axis along the sensor's normal
              vector pointing upwards, with x and y axes perpendicular to the z axis
              and to each other.

            The rocket frame of reference is defined as having the z axis
            along the rocket's axis of symmetry pointing upwards, with x and y axes
            perpendicular to the z axis and to each other. A rotation around the x
            axis configures a pitch, around the y axis a yaw, and around the z axis a
            roll. Default is (0, 0, 0), meaning the sensor is aligned with all
            rocket axes.
        measurement_range : float, tuple, optional
            The measurement range of the sensor in T. If a float, the
            same range is applied both for positive and negative values. If a
            tuple, the first value is the positive range and the second value is
            the negative range. Default is np.inf.
        resolution : float, optional
            The resolution of the sensor in T/LSB. Default is 0, meaning no
            quantization is applied.
        hard_iron_distortion : float, list, optional
            The hard iron distortion desired for the sensor in T.

            - If a float, the same value is applied to each axis.
            - If a list, the distortion is applied per axis.

            Default is 0.
        soft_iron_distortion : Matrix, str, optional
            Soft iron distortion caused by materials with high permeability on
            the rocket.

            - If a Matrix, a direct 3x3 transformation matrix applied
              to the magnetic field.
            - If the string "plates", computes distortion
              based on plates attached to the Rocket object.

            Default is Matrix.identity().
        power_interference : int, float, list, str, optional
            The power interference is the magnetic distortion due to current
            flowing through wires in the avionics bay in T:

            - If an int or float, the same magnetic field will be applied to each axis.
            - If a list, the given vector will be considered as power interference.
            - If "wires", models total interference using rocket wires.
            - If "personalized", requires passing activation_signal_interference and
              communications_interference explicitly.

            Default is 0.
        activation_signal_interference : int, float, list, str, optional
            Magnetic field in T generated by ignition wires:

            - If an int or float, applied uniformly to each axis.
            - If list, the given vector is used.
            - If "wires", computed from ignition wires attached to the rocket.

            Default is None.
        communications_interference : int, float, list, str, optional
            Magnetic field in T generated by communication wires:

            - If an int or float, applied uniformly to each axis.
            - If list, the given vector is used.
            - If "wires", computed from communication wires attached to the rocket.

            Default is None.
        noise_density : float, list, optional
            The noise density of the sensor for Gaussian white noise in T/√Hz.
            Default is 0, meaning no noise is applied.
        noise_variance : float, list, optional
            The noise variance of the sensor for Gaussian white noise in T^2.
            Default is 1.
        random_walk_density : float, list, optional
            The random walk density of the sensor in T/√Hz. Default is 0.
        random_walk_variance : float, list, optional
            The random walk variance of the sensor in T^2. Default is 1.
        constant_bias : float, list, optional
            The constant bias of the sensor in T. Default is 0.
        operating_temperature : float, optional
            The operating temperature of the sensor in Kelvin. Default is 298.
        temperature_bias : float, list, optional
            The temperature bias of the sensor in T/K. Default is 0.
        temperature_scale_factor : float, list, optional
            The temperature scale factor of the sensor in %/K. Default is 0.
        cross_axis_sensitivity : float, optional
            Skewness of the sensor's axes in percentage. Default is 0.
        name : str, optional
            Name of the sensor. Default is "Magnetometer".
        seed : int, optional
            Seed for the random number generator. Default is None.
        """
        self.magnetic_interference = [0, 0, 0]

        if isinstance(power_interference, str):
            if power_interference.lower() == "wires":
                self.initial_power_interference = "wires"
                self._validate_power_wires_parameters()
            elif power_interference.lower() == "personalized":
                self.initial_power_interference = "personalized"
                self._validate_power_personalized_parameters(
                    activation_signal_interference, communications_interference
                )
            else:
                raise InvalidParameterError(
                    "The accepted strings are 'wires' or 'personalized'"
                )
        else:
            self.initial_power_interference = "number"
            if isinstance(power_interference, (int, float)):
                power_interference = [power_interference] * 3
            self.power_interference = list(power_interference)
            self._power_interference = Vector(self.power_interference)

        self._validate_soft_iron(soft_iron_distortion)
        self._validate_hard_iron(hard_iron_distortion)
        self.prints = _InertialSensorPrints(self)
        self._sensor_from_bacs, self.sensor_from_bacs_t = [None, None]

        # Get current decimal year
        current_date = datetime.now().strftime("%Y-%m-%d")
        self.year = decimal_year(current_date)

        # initialize the magnetic model
        self.wmm = WMMv2()

        # initialize InertialSensor class
        super().__init__(
            sampling_rate=sampling_rate,
            orientation=orientation,
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
            seed=seed,
        )

    def _validate_soft_iron(self, soft_iron_distortion) -> None:
        """Checks and defines the soft_iron_distortion parameter."""
        if isinstance(soft_iron_distortion, Matrix):
            self._soft_iron_distortion_matrix = soft_iron_distortion
            self.soft_iron_distortion_difference = []
            self.initial_soft_iron_distortion_matrix = "matrix"
        elif isinstance(soft_iron_distortion, str):
            if soft_iron_distortion == "plates":
                self._soft_iron_distortion_matrix = Matrix.identity()
                self.initial_soft_iron_distortion_matrix = "plates"
                self.total_soft_iron_distortion_matrix_computed = False
                self.soft_iron_distortion_difference = []
            else:
                raise InvalidParameterError("The accepted string must be plates")

    def _validate_hard_iron(self, hard_iron_distortion) -> None:
        """Checks and defines the hard_iron_distortion parameter."""
        if isinstance(hard_iron_distortion, (float, int)):
            hard_iron_distortion = [hard_iron_distortion] * 3

        self.hard_iron_distortion = list(hard_iron_distortion)
        self._hard_iron_distortion = Vector(self.hard_iron_distortion)

    def _validate_power_personalized_parameters(
        self, activation_signal_interference, communications_interference
    ) -> None:
        """Checks and defines the parameters related to the power interference."""
        if (
            activation_signal_interference is None
            or communications_interference is None
        ):
            raise InvalidParameterError(
                "For 'personalized' interference, you must provide values for both 'activation_signal_interference' and 'communications_interference'."
            )

        self._validate_activation_signal_interference(activation_signal_interference)
        self._validate_communications_interference(communications_interference)

    def _validate_activation_signal_interference(
        self, activation_signal_interference
    ) -> None:
        """Checks activation signal interference and defines the related attributes."""

        if isinstance(activation_signal_interference, str):
            if activation_signal_interference.lower() == "wires":
                self.activation_signal_interference = [0, 0, 0]
                self.initial_activation_signal_interference = "wires"
            else:
                raise InvalidParameterError("The accepted string is wires")
        else:
            if isinstance(activation_signal_interference, (int, float)):
                activation_signal_interference = [activation_signal_interference] * 3
            self.initial_activation_signal_interference = "number"
            self.activation_signal_interference = list(activation_signal_interference)
        self._activation_signal_interference = Vector(
            self.activation_signal_interference
        )

    def _validate_communications_interference(
        self, communications_interference
    ) -> None:
        """Checks communications interference and defines the related attributes."""

        if isinstance(communications_interference, str):
            if communications_interference.lower() == "wires":
                self.communications_interference = [0, 0, 0]
                self.initial_communications_interference = "wires"
                self.communications_computed = False
            else:
                raise InvalidParameterError("The accepted string is wires")
        else:
            if isinstance(communications_interference, (int, float)):
                communications_interference = [communications_interference] * 3
            self.initial_communications_interference = "number"
            self.communications_interference = list(communications_interference)

        self._communications_interference = Vector(self.communications_interference)

    def _validate_power_wires_parameters(self):
        """Defines the power related attributes when the power_interference parmeter is "wires"."""
        self.power_interference = [0, 0, 0]
        self.communications_interference = [0, 0, 0]
        self.communications_computed = False
        self.activation_signal_interference = [0, 0, 0]

        self.initial_communications_interference = "wires"
        self.initial_activation_signal_interference = "wires"

    def measure(self, time: float, **kwargs) -> None:
        """Obtains the simulated reading of the magnetometer for a given time step.

        Parameters
        ----------
        time : float
            Current time in seconds.
        kwargs : dict
            Keyword arguments dictionary containing the following keys:

            - u : np.ndarray
                State vector of the rocket.
                u = [x, y, z, vx, vy, vz, e0, e1, e2, e3, wx, wy, wz]

            - rocket : Rocket, optional
                RocketPy Rocket class. It is necessary when the magnetic distortion is modeled
                using wires or plates. If there is no magnetic distortion, or if it is defined directly,
                it is not necessary.

            - u_dot : np.ndarray
                Derivative of the state vector of the rocket.

            - relative_position : Vector
                Position of the sensor relative to the rocket center of dry mass in m.

            - environment : Environment
                Environment object containing atmospheric conditions.

            - parachute_events : list, optional
                List storing parachute events triggered during flight (only required if
                ``ignition_wire_function`` is "parachute_deployment"). Each element is a list
                containing the trigger time as the first element and the parachute object
                as the second.
        """
        # initialization of parameters
        u = kwargs["u"]  # state vector
        parachute_events = kwargs.get("parachute_events")
        self._sensor_from_bacs = kwargs[
            "relative_position"
        ]  # sensor position from body axis coordinate system
        self.sensor_from_bacs_t = tuple(self._sensor_from_bacs)
        current_time = time
        lat0, lon0, launch_site_elevation = (
            kwargs["environment"].latitude,
            kwargs["environment"].longitude,
            kwargs["environment"].elevation,
        )
        earth_radius = kwargs["environment"].earth_radius
        rocket = kwargs.get(
            "rocket"
        )  # if there is no distortion the rocket is not necessary

        # u[6:10]: Quaternion represents the center of dry mass with respect to the inertial frame.
        rotation_bacs_to_inertial = Matrix.transformation(
            u[6:10]
        )  # rotation matrix from cdm to inertial frame

        # --- obtain the current longitude, latitude and elevation ---
        # obtain the sensor coordinates in the inertial frame, by adding the offset to the position vector
        x_inertial, y_inertial, z_inertial = (
            rotation_bacs_to_inertial @ self._sensor_from_bacs
            + Vector(
                u[0:3]
            )  # Vector(u[0:3]) is the coordinates center of dry mass in the inertial frame
        )
        b_north, b_east, b_down = self._obtain_magnetic_field(
            x_inertial,
            y_inertial,
            z_inertial,
            launch_site_elevation,
            earth_radius,
            lat0,
            lon0,
        )

        # --- Transform from NED to Rocketpy's inertial frame ---
        b_field_inertial = Vector([b_east, b_north, -b_down])  # T

        # --- from Rocketpy's inertial frame to bacs frame ---
        b_field_bacs = rotation_bacs_to_inertial.transpose @ b_field_inertial  # T

        # --- Apply magnetic interference ---
        b_field_bacs = self.apply_magnetic_interference(
            b_field_bacs, rocket, current_time, parachute_events
        )  # T
        # Transform body frame (bacs) to sensor frame, includes the cross-axis sensitivity adjustment
        b_field_sensor = (
            self._total_rotation_sensor_to_body.transpose @ b_field_bacs
        )  # Ts
        # --- apply noise and quantize ---
        b_field_sensor = self.apply_temperature_drift(b_field_sensor)  # T
        b_field_sensor = self.apply_noise(b_field_sensor)  # T
        b_field_sensor = self.quantize(b_field_sensor)  # T

        self.measurement = (b_field_sensor.x, b_field_sensor.y, b_field_sensor.z)  # T
        self._save_data((time, *b_field_sensor))

    def _obtain_magnetic_field(
        self,
        x_inertial: float,
        y_inertial: float,
        z_inertial: float,
        launch_site_elevation: float,
        earth_radius: float,
        lat0: float,
        lon0: float,
    ) -> tuple[float, float, float]:
        """Calculates Earth's magnetic field components in the North-East-Down (NED) frame.

        Parameters
        ----------
        x_inertial : float
            Inertial x-coordinate (East) in meters.
        y_inertial : float
            Inertial y-coordinate (North) in meters.
        z_inertial : float
            Inertial z-coordinate (Up) in meters.
        launch_site_elevation : float
            Launch site elevation above sea level in meters.
        earth_radius : float
            Radius of Earth at launch location in meters.
        lat0 : float
            Launch site latitude in degrees.
        lon0 : float
            Launch site longitude in degrees.

        Returns
        -------
        b_north : float
            Geomagnetic field North component in Tesla (T).
        b_east : float
            Geomagnetic field East component in Tesla (T).
        b_down : float
            Geomagnetic field Down component in Tesla (T).
        """
        altitude_wgs84_km = (z_inertial + launch_site_elevation) / 1000.0
        drift = math.hypot(x_inertial, y_inertial)
        bearing = math.atan2(x_inertial, y_inertial) * (180 / math.pi) % 360
        latitude, longitude = inverted_haversine(
            lat0, lon0, drift, bearing, earth_radius
        )

        calculate_geomagnetic(
            self.wmm, latitude, longitude, self.year, altitude_wgs84_km
        )

        b_north = self.wmm.bx / 1e9
        b_east = self.wmm.by / 1e9
        b_down = self.wmm.bz / 1e9

        return b_north, b_east, b_down

    def apply_magnetic_interference(
        self,
        b_field: Vector,
        rocket: Rocket,
        current_time: float | int,
        parachute_events: list | None = None,
    ) -> Vector:
        """Applies magnetic distortion due to power interference,
        hard iron, and soft iron.

        Parameters
        ----------
        b_field : Vector
            Magnetic field Vector.
        rocket : Rocket
            RocketPy Rocket class.
        current_time : float, optional
            Current time of the simulation (required if ``ignition_wire_function``
            is "motor_ignition").
        parachute_events : list, optional
            List storing parachute events triggered during flight (required if
            ``ignition_wire_function`` is "parachute_deployment"). Each element is a
            list containing the trigger time as the first element and the parachute
            object as the second.

        Returns
        -------
        b_field : Vector
            Magnetic field vector after adjustment for hard iron, soft iron,
            and power interference.
        """
        self.magnetic_interference = [0, 0, 0]

        b_field = self.apply_soft_iron(b_field, rocket)  # T
        b_field = self.apply_hard_iron(b_field)  # T
        b_field = self.apply_power_interference(
            b_field, rocket, current_time, parachute_events
        )  # T

        self.magnetic_interference = [
            self.power_interference[0]
            + self.hard_iron_distortion[0]
            + self.soft_iron_distortion_difference[0],
            self.power_interference[1]
            + self.hard_iron_distortion[1]
            + self.soft_iron_distortion_difference[1],
            self.power_interference[2]
            + self.hard_iron_distortion[2]
            + self.soft_iron_distortion_difference[2],
        ]
        return b_field

    def apply_soft_iron(self, b_field: Vector, rocket: Rocket) -> Vector:
        """Applies soft iron distortion, which is the distortion of the
        magnetic field caused by materials with high magnetic permeability relative
        to the permeability of free space. These materials have lower magnetic resistance,
        bending magnetic field lines through them.

        Parameters
        ----------
        b_field : Vector
            Vector reading of the Earth's magnetic field.
        rocket : Rocket
            RocketPy Rocket class.

        Returns
        -------
        b_field_distorted : Vector
            Magnetic field vector after soft iron distortion.
        """
        if self.initial_soft_iron_distortion_matrix == "plates":
            if not self.total_soft_iron_distortion_matrix_computed:
                for plate, _, _ in rocket.plates:
                    if (
                        self.sensor_from_bacs_t
                        not in plate._magnetic_distortion_matrices
                    ):
                        plate.calculate_soft_iron_distortion_matrix(
                            self._sensor_from_bacs, frame="bacs"
                        )
                        self._soft_iron_distortion_matrix = (
                            self._soft_iron_distortion_matrix
                            + plate._magnetic_distortion_matrices[
                                self.sensor_from_bacs_t
                            ]
                        )
                self.total_soft_iron_distortion_matrix_computed = True
            b_field_distorted = self._soft_iron_distortion_matrix @ b_field
        else:
            b_field_distorted = self._soft_iron_distortion_matrix @ b_field

        self.soft_iron_distortion_difference = list(b_field_distorted - b_field)
        return b_field_distorted

    def apply_hard_iron(self, b_field: Vector) -> Vector:
        """Applies hard iron distortion. This magnetic distortion is caused by
        permanent magnets or magnetized materials on the rocket that move along with
        the sensor (e.g., steel screws, battery casing, ferromagnetic components),
        producing a constant offset that shifts the center of the magnetic data.

        Parameters
        ----------
        b_field : Vector
            Magnetic field Vector.

        Returns
        -------
        b_field : Vector
            Magnetic field vector after applying hard iron distortion.
        """
        b_field = b_field + self._hard_iron_distortion
        return b_field

    def apply_power_interference(
        self,
        b_field: Vector,
        rocket: Rocket,
        current_time: float,
        parachute_events: list | None = None,
    ) -> Vector:
        """Applies electromagnetic interference to the magnetic field vector.
        Considers current flow when trigger conditions are fulfilled for ignition wires,
        or continuously for communication wires.

        Parameters
        ----------
        b_field : Vector
            Magnetic field Vector.
        rocket : Rocket
            RocketPy Rocket class.
        current_time : float, optional
            Current time of the simulation (required if ``ignition_wire_function``
            is "motor_ignition").
        parachute_events : list, optional
            List storing parachute events triggered during flight (required if
            ``ignition_wire_function`` is "parachute_deployment"). Each element is a
            list containing the trigger time as the first element and the parachute
            object as the second.

        Returns
        -------
        b_field : Vector
            Magnetic field vector after adjustment for both activation signal
            interference and communications interference.
        """
        if self.initial_power_interference in ("wires", "personalized"):
            self.power_interference = [0, 0, 0]
            b_field = self.apply_communications_interference(b_field, rocket)
            b_field = self.apply_activation_signal_interference(
                b_field, rocket, current_time, parachute_events
            )
            self.power_interference = [
                self.activation_signal_interference[0]
                + self.communications_interference[0],
                self.activation_signal_interference[1]
                + self.communications_interference[1],
                self.activation_signal_interference[2]
                + self.communications_interference[2],
            ]
            self._power_interference = Vector(self.power_interference)
        else:
            b_field = b_field + self._power_interference

        return b_field

    def apply_communications_interference(
        self, b_field: Vector, rocket: Rocket
    ) -> Vector:
        """Applies interference caused by current flowing through communication wires.

        Parameters
        ----------
        b_field : Vector
            Magnetic field Vector.
        rocket : Rocket
            RocketPy Rocket class.

        Returns
        -------
        b_field : Vector
            Magnetic field vector after adjustment for communications magnetic
            interference.
        """
        if self.initial_communications_interference == "wires":
            if rocket._communication_wires:
                if not self.communications_computed:
                    for communication_wire in rocket._communication_wires:
                        communication_wire.measure_magnetic_field(
                            self._sensor_from_bacs, frame="bacs"
                        )
                        self.communications_interference = [
                            self.communications_interference[0]
                            + communication_wire.magnetic_field[
                                self.sensor_from_bacs_t
                            ][0],
                            self.communications_interference[1]
                            + communication_wire.magnetic_field[
                                self.sensor_from_bacs_t
                            ][1],
                            self.communications_interference[2]
                            + communication_wire.magnetic_field[
                                self.sensor_from_bacs_t
                            ][2],
                        ]

                    self._communications_interference = Vector(
                        self.communications_interference
                    )
                    b_field = b_field + self._communications_interference
                    self.communications_computed = True
                else:
                    b_field = b_field + self._communications_interference
            else:
                raise InvalidParameterError(
                    "You must define first some communication wires, to be able to consider the magnetic disturbance created by them"
                )
        else:
            b_field = b_field + self._communications_interference

        return b_field

    def apply_activation_signal_interference(
        self,
        b_field: Vector,
        rocket: Rocket,
        current_time: float,
        parachute_events: list | None = None,
    ) -> Vector:
        """Applies magnetic interference caused by current flowing through
        ignition wires during an activation signal.

        Parameters
        ----------
        b_field : Vector
            Magnetic field Vector.
        rocket : Rocket
            RocketPy Rocket class.
        current_time : float, optional
            Current time of the simulation (required if ``ignition_wire_function``
            is "motor_ignition").
        parachute_events : list, optional
            List storing parachute events triggered during flight (required if
            ``ignition_wire_function`` is "parachute_deployment"). Each element is a
            list containing the trigger time as the first element and the parachute
            object as the second.

        Returns
        -------
        b_field : Vector
            Magnetic field vector after adjustment for activation signal interference.
        """

        if self.initial_activation_signal_interference == "wires":
            if rocket._ignition_wires:
                self.activation_signal_interference = [0, 0, 0]
                for ignition_wire in rocket._ignition_wires:
                    if ignition_wire.ignition_wire_function == "parachute_deployment":
                        b_field = (
                            self._calculate_activation_signal_interference_parachute(
                                b_field, current_time, parachute_events, ignition_wire
                            )
                        )

                    elif ignition_wire.ignition_wire_function == "motor_ignition":
                        b_field = self._calculate_activation_signal_interference_motor(
                            b_field, rocket, current_time, ignition_wire
                        )
                    else:
                        raise InvalidParameterError(
                            "The accepted strings for the ignition_wire_function are motor_ignition and parachute_deployment"
                        )
            else:
                raise InvalidParameterError(
                    "You must define some ignition wire to be able to consider its magnetic disturbance."
                )
        else:
            b_field = b_field + self._activation_signal_interference

        return b_field

    def _calculate_activation_signal_interference_parachute(
        self, b_field, current_time, parachute_events, ignition_wire
    ):
        """Applies activation signal interference generated during parachute deployment.

        Parameters
        ----------
        b_field : Vector
            Magnetic field Vector.
        current_time : float
            Current time of the simulation.
        parachute_events : list
            List storing parachute events triggered during flight.
        ignition_wire : Wire
            Wire instance with ``wire_type`` "ignition" and function "parachute_deployment".

        Returns
        -------
        b_field : Vector
            Magnetic field vector after calculating parachute ignition interference.
        """
        if parachute_events is not None:
            for parachute_event in parachute_events:
                ejection_time = parachute_event[0]
                parachute = parachute_event[1]
                if (
                    parachute.name == ignition_wire.parachute_name
                    and ejection_time != 0
                    and current_time
                    <= ejection_time + ignition_wire.extra_ignition_time
                ):
                    if self.sensor_from_bacs_t not in ignition_wire._magnetic_field:
                        ignition_wire.measure_magnetic_field(
                            self._sensor_from_bacs, frame="bacs"
                        )
                    self.activation_signal_interference = [
                        self.activation_signal_interference[0]
                        + ignition_wire.magnetic_field[self.sensor_from_bacs_t][0],
                        self.activation_signal_interference[1]
                        + ignition_wire.magnetic_field[self.sensor_from_bacs_t][1],
                        self.activation_signal_interference[2]
                        + ignition_wire.magnetic_field[self.sensor_from_bacs_t][2],
                    ]

                    b_field = (
                        b_field + ignition_wire._magnetic_field[self.sensor_from_bacs_t]
                    )
        else:
            raise InvalidParameterError(
                "The parachute events should be passed if a wire has ignition_wire_function == parachute_deployment."
            )

        return b_field

    def _calculate_activation_signal_interference_motor(
        self, b_field, rocket, current_time, ignition_wire
    ):
        """Applies activation signal interference generated during motor ignition.

        Parameters
        ----------
        b_field : Vector
            Magnetic field Vector.
        rocket : Rocket
            RocketPy Rocket class.
        current_time : float
            Current time of the simulation.
        ignition_wire : Wire
            Wire instance with ``wire_type`` "ignition" and function "motor_ignition".

        Returns
        -------
        b_field : Vector
            Magnetic field vector after calculating motor ignition interference.
        """
        if (
            current_time
            <= rocket.motor.burn_start_time + ignition_wire.extra_ignition_time
        ):
            if self.sensor_from_bacs_t not in ignition_wire._magnetic_field:
                ignition_wire.measure_magnetic_field(
                    self._sensor_from_bacs, frame="bacs"
                )
            self.activation_signal_interference = [
                self.activation_signal_interference[0]
                + ignition_wire.magnetic_field[self.sensor_from_bacs_t][0],
                self.activation_signal_interference[1]
                + ignition_wire.magnetic_field[self.sensor_from_bacs_t][1],
                self.activation_signal_interference[2]
                + ignition_wire.magnetic_field[self.sensor_from_bacs_t][2],
            ]
            b_field = b_field + ignition_wire._magnetic_field[self.sensor_from_bacs_t]
        return b_field

    def export_measured_data(self, filename, file_format="csv") -> None:
        """Exports the measured values to a file.

        Parameters
        ----------
        filename : str
            Name of the file to export the values to.
        file_format : str
            Format of the file to export the values to. Options are "csv" and
            "json". Default is "csv".
        """
        self._generic_export_measured_data(
            filename=filename,
            file_format=file_format,
            data_labels=("t", "Bx", "By", "Bz"),
        )

    @classmethod
    def from_dict(cls, data: dict) -> "Magnetometer":
        """Creates a Magnetometer instance from a dictionary containing the
        initialization parameters.

        Parameters
        ----------
        data : dict
            Dictionary containing magnetometer initialization parameters.

        Returns
        -------
        magnetometer : Magnetometer
            Magnetometer instance initialized with the specified parameters.
        """
        return cls(
            # Mandatory Parameter
            sampling_rate=data["sampling_rate"],
            # Optional Parameters
            orientation=data.get("orientation", (0, 0, 0)),
            measurement_range=data.get("measurement_range", np.inf),
            resolution=data.get("resolution", 0),
            hard_iron_distortion=data.get("hard_iron_distortion", 0.0),
            soft_iron_distortion=data.get("soft_iron_distortion", Matrix.identity()),
            power_interference=data.get("power_interference", 0),
            activation_signal_interference=data.get(
                "activation_signal_interference", None
            ),
            communications_interference=data.get("communications_interference", None),
            # Noise Profiles
            noise_density=data.get("noise_density", 0),
            noise_variance=data.get("noise_variance", 1),
            random_walk_density=data.get("random_walk_density", 0),
            random_walk_variance=data.get("random_walk_variance", 1),
            constant_bias=data.get("constant_bias", 0),
            # Environmental & Structural Shifts
            operating_temperature=data.get("operating_temperature", 298),
            temperature_bias=data.get("temperature_bias", 0),
            temperature_scale_factor=data.get("temperature_scale_factor", 0),
            cross_axis_sensitivity=data.get("cross_axis_sensitivity", 0),
            seed=data.get("seed", None),
        )
