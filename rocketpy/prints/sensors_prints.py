from abc import ABC, abstractmethod

UNITS = {
    "Gyroscope": "rad/s",
    "Accelerometer": "m/s^2",
    "Magnetometer": "T",
    "Barometer": "Pa",
    "TemperatureSensor": "K",
}


class _SensorsPrints(ABC):
    def __init__(self, sensor):
        self.sensor = sensor
        self.units = UNITS[sensor.type]

    def _print_aligned(self, label, value):
        """Prints a label and a value aligned vertically."""
        print(f"{label:<26} {value}")

    def identity(self):
        """Prints the identity of the sensor."""
        print("Identification of the Sensor:\n")
        self._print_aligned("Name:", self.sensor.name)
        self._print_aligned("Type:", self.sensor.type)

    def orientation(self):
        """Prints the orientation of the sensor."""
        print("\nOrientation of the Sensor:\n")
        self._print_aligned("Orientation:", self.sensor.orientation)
        self._print_aligned("Normal Vector:", self.sensor.normal_vector)
        print("Rotation Matrix:")
        for row in self.sensor.rotation_matrix:
            value = " ".join(f"{val:.2f}" for val in row)
            value = [float(val) for val in value.split()]
            self._print_aligned("", value)

    def quantization(self):
        """Prints the quantization of the sensor."""
        print("\nQuantization of the Sensor:\n")
        self._print_aligned(
            "Measurement Range:",
            f"{self.sensor.measurement_range[0]} to {self.sensor.measurement_range[1]} ({self.units})",
        )
        self._print_aligned("Resolution:", f"{self.sensor.resolution} {self.units}/LSB")

    def noise(self):
        """Prints the noise of the sensor."""
        self._general_noise()

    def _general_noise(self):
        """Prints the noise of the sensor."""
        print("\nNoise of the Sensor:\n")
        self._print_aligned(
            "Noise Density:", f"{self.sensor.noise_density} {self.units}/√Hz"
        )
        self._print_aligned(
            "Noise Variance:", f"{self.sensor.noise_variance} ({self.units})^2"
        )
        self._print_aligned(
            "Random Walk Density:",
            f"{self.sensor.random_walk_density} {self.units}/√Hz",
        )
        self._print_aligned(
            "Random Walk Variance:",
            f"{self.sensor.random_walk_variance} ({self.units})^2",
        )
        self._print_aligned(
            "Constant Bias:", f"{self.sensor.constant_bias} {self.units}"
        )
        self._print_aligned(
            "Operating Temperature:", f"{self.sensor.operating_temperature} °C"
        )
        self._print_aligned(
            "Temperature Bias:", f"{self.sensor.temperature_bias} {self.units}/°C"
        )
        self._print_aligned(
            "Temperature Scale Factor:", f"{self.sensor.temperature_scale_factor} %/°C"
        )
        self._print_aligned(
            "Cross Axis Sensitivity:", f"{self.sensor.cross_axis_sensitivity} %"
        )

    def all(self):
        """Prints all information of the sensor."""
        self.identity()
        self.quantization()
        self.noise()


class _InertialSensorsPrints(_SensorsPrints):
    def __init__(self, sensor):
        super().__init__(sensor)

    def orientation(self):
        """Prints the orientation of the sensor."""
        print("\nOrientation of the Sensor:\n")
        self._print_aligned("Orientation:", self.sensor.orientation)
        self._print_aligned("Normal Vector:", self.sensor.normal_vector)
        print("Rotation Matrix:")
        for row in self.sensor.rotation_matrix:
            value = " ".join(f"{val:.2f}" for val in row)
            value = [float(val) for val in value.split()]
            self._print_aligned("", value)

    def all(self):
        """Prints all information of the sensor."""
        self.identity()
        self.orientation()
        self.quantization()
        self.noise()


class _AccelerometerPrints(_InertialSensorsPrints):
    """Class that contains all accelerometer prints."""

    def __init__(self, accelerometer):
        """Initialize the class."""
        super().__init__(accelerometer)


class _GyroscopePrints(_InertialSensorsPrints):
    """Class that contains all gyroscope prints."""

    def __init__(self, gyroscope):
        """Initialize the class."""
        super().__init__(gyroscope)

    def noise(self):
        """Prints the noise of the sensor."""
        self._general_noise()
        self._print_aligned(
            "Acceleration Sensitivity:",
            f"{self.sensor.acceleration_sensitivity} rad/s/g",
        )


class _BarometerPrints(_SensorsPrints):
    """Class that contains all barometer prints."""

    def __init__(self, barometer):
        """Initialize the class."""
        super().__init__(barometer)
