from abc import ABC


class _SensorsPrints(ABC):
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

    def orientation(self):
        """Prints the orientation of the sensor."""
        print("\nOrientation:\n")
        self._print_aligned("Orientation:", self.sensor.orientation)
        self._print_aligned("Normal Vector:", self.sensor.normal_vector)
        print("Rotation Matrix:")
        for row in self.sensor.rotation_matrix:
            value = " ".join(f"{val:.2f}" for val in row)
            value = [float(val) for val in value.split()]
            self._print_aligned("", value)

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
            "Operating Temperature:", f"{self.sensor.operating_temperature} °C"
        )
        self._print_aligned(
            "Temperature Bias:", f"{self.sensor.temperature_bias} {self.units}/°C"
        )
        self._print_aligned(
            "Temperature Scale Factor:", f"{self.sensor.temperature_scale_factor} %/°C"
        )

    def all(self):
        """Prints all information of the sensor."""
        self.identity()
        self.quantization()
        self.noise()


class _InertialSensorsPrints(_SensorsPrints):

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


class _AccelerometerPrints(_InertialSensorsPrints):
    """Class that contains all accelerometer prints."""


class _GyroscopePrints(_InertialSensorsPrints):
    """Class that contains all gyroscope prints."""

    def noise(self):
        """Prints the noise of the sensor."""
        self._general_noise()
        self._print_aligned(
            "Acceleration Sensitivity:",
            f"{self.sensor.acceleration_sensitivity} rad/s/g",
        )


# TODO: simplify prints
class _BarometerPrints(_SensorsPrints):
    """Class that contains all barometer prints."""
