import numpy as np


class _HybridMotorPrints:
    """Class that holds prints methods for HybridMotor class.

    Attributes
    ----------
    _HybridMotorPrints.hybrid_motor : hybrid_motor
        HybridMotor object that will be used for the prints.

    """

    def __init__(
        self,
        hybrid_motor,
    ):
        """Initializes _HybridMotorPrints class

        Parameters
        ----------
        hybrid_motor: HybridMotor
            Instance of the HybridMotor class.

        Returns
        -------
        None
        """
        self.hybrid_motor = hybrid_motor

    def nozzle_details(self):
        """Prints out all data available about the Nozzle.

        Returns
        -------
        None
        """
        # Print nozzle details
        print("Nozzle Details")
        print(f"Outlet Radius: {self.hybrid_motor.nozzle_radius} m")
        print(f"Throat Radius: {self.hybrid_motor.solid.throat_radius} m")
        print(f"Outlet Area: {np.pi * self.hybrid_motor.nozzle_radius ** 2:.6f} m²")
        print(
            f"Throat Area: {np.pi * self.hybrid_motor.solid.throat_radius ** 2:.6f} m²"
        )
        print(f"Position: {self.hybrid_motor.nozzle_position} m\n")

    def grain_details(self):
        """Prints out all data available about the Grain.

        Returns
        -------
        None
        """
        print("Grain Details")
        print(f"Number of Grains: {self.hybrid_motor.solid.grain_number}")
        print(f"Grain Spacing: {self.hybrid_motor.solid.grain_separation} m")
        print(f"Grain Density: {self.hybrid_motor.solid.grain_density} kg/m3")
        print(f"Grain Outer Radius: {self.hybrid_motor.solid.grain_outer_radius} m")
        print(
            "Grain Inner Radius: "
            f"{self.hybrid_motor.solid.grain_initial_inner_radius} m"
        )
        print(f"Grain Height: {self.hybrid_motor.solid.grain_initial_height} m")
        print(f"Grain Volume: {self.hybrid_motor.solid.grain_initial_volume:.3f} m3")
        print(f"Grain Mass: {self.hybrid_motor.solid.grain_initial_mass:.3f} kg\n")

    def motor_details(self):
        """Prints out all data available about the HybridMotor.

        Returns
        -------
        None
        """
        print("Motor Details")
        print(f"Total Burning Time: {self.hybrid_motor.burn_duration} s")
        print(
            f"Total Propellant Mass: {self.hybrid_motor.propellant_initial_mass:.3f} kg"
        )
        print(f"Structural Mass Ratio: {self.hybrid_motor.structural_mass_ratio:.3f}")
        avg = self.hybrid_motor.exhaust_velocity.average(*self.hybrid_motor.burn_time)
        print(f"Average Propellant Exhaust Velocity: {avg:.3f} m/s")
        print(f"Average Thrust: {self.hybrid_motor.average_thrust:.3f} N")
        print(
            f"Maximum Thrust: {self.hybrid_motor.max_thrust} N at "
            f"{self.hybrid_motor.max_thrust_time} s after ignition."
        )
        print(f"Total Impulse: {self.hybrid_motor.total_impulse:.3f} Ns\n")

    def all(self):
        """Prints out all data available about the HybridMotor.

        Returns
        -------
        None
        """

        self.nozzle_details()
        self.grain_details()
        self.motor_details()
