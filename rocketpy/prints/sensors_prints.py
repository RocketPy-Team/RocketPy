from abc import ABC


class _SensorPrints(ABC):
    def __init__(self, sensor):
        self.sensor = sensor
        self.units = sensor.units

    def _print_aligned(self, label, value):
        """Prints a label and a value aligned vertically."""
        print(f"{label:<26} {value}")

    def identity(self):
        """Prints the identity of the sensor."""
        print("Identification:\n")
        self._print_aligned("Name:", self.sensor.name)
        self._print_aligned("Type:", self.sensor.__class__.__name__)

    def quantization(self):
        """Prints the quantization of the sensor."""
        print("\nQuantization:\n")
        self._print_aligned(
            "Measurement Range:",
            f"{self.sensor.measurement_range[0]} "
            + f"to {self.sensor.measurement_range[1]} ({self.units})",
        )
        self._print_aligned("Resolution:", f"{self.sensor.resolution} {self.units}/LSB")

    def noise(self):
        """Prints the noise of the sensor."""
        self._general_noise()

    def _general_noise(self):
        """Prints the noise of the sensor."""
        print("\nNoise:\n")
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
            "Operating Temperature:", f"{self.sensor.operating_temperature} K"
        )
        self._print_aligned(
            "Temperature Bias:", f"{self.sensor.temperature_bias} {self.units}/K"
        )
        self._print_aligned(
            "Temperature Scale Factor:", f"{self.sensor.temperature_scale_factor} %/K"
        )

    def all(self):
        """Prints all information of the sensor."""
        self.identity()
        self.quantization()
        self.noise()


class _InertialSensorPrints(_SensorPrints):
    def orientation(self):
        """Prints the orientation of the sensor."""
        print("\nOrientation of the Sensor:\n")
        self._print_aligned("Orientation:", self.sensor.orientation)
        self._print_aligned("Normal Vector:", self.sensor.normal_vector)
        print("Rotation Matrix:")
        for row in self.sensor.rotation_sensor_to_body:
            value = " ".join(f"{val:.2f}" for val in row)
            value = [float(val) for val in value.split()]
            self._print_aligned("", value)

    def _general_noise(self):
        super()._general_noise()
        self._print_aligned(
            "Cross Axis Sensitivity:", f"{self.sensor.cross_axis_sensitivity} %"
        )

    def all(self):
        """Prints all information of the sensor."""
        self.identity()
        self.orientation()
        self.quantization()
        self.noise()


class _GyroscopePrints(_InertialSensorPrints):
    """Class that contains all gyroscope prints."""

    def noise(self):
        """Prints the noise of the sensor."""
        self._general_noise()
        self._print_aligned(
            "Acceleration Sensitivity:",
            f"{self.sensor.acceleration_sensitivity} rad/s/g",
        )


class _GnssReceiverPrints(_SensorPrints):
    """Class that contains all GnssReceiver prints."""

    def accuracy(self):
        """Prints the accuracy of the sensor."""
        print("\nAccuracy:\n")
        self._print_aligned("Position Accuracy:", f"{self.sensor.position_accuracy} m")
        self._print_aligned("Altitude Accuracy:", f"{self.sensor.altitude_accuracy} m")

    def all(self):
        """Prints all information of the sensor."""
        self.identity()
        self.accuracy()
