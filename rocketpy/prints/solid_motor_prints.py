class _SolidMotorPrints:
    """Class that holds prints methods for SolidMotor class.

    Attributes
    ----------
    _SolidMotorPrints.solid_motor : solid_motor
        SolidMotor object that will be used for the prints.

    """

    def __init__(
        self,
        solid_motor,
    ):
        """Initializes _SolidMotorPrints class

        Parameters
        ----------
        solid_motor: SolidMotor
            Instance of the SolidMotor class.

        Returns
        -------
        None
        """
        self.solid_motor = solid_motor

    def nozzle_details(self):
        """Prints out all data available about the SolidMotor nozzle.

        Returns
        -------
        None
        """
        print("Nozzle Details")
        print(f"Nozzle Radius: {self.solid_motor.nozzle_radius} m")
        print(f"Nozzle Throat Radius: {self.solid_motor.throat_radius} m\n")

    def grain_details(self):
        """Prints out all data available about the SolidMotor grain.

        Returns
        -------
        None
        """
        print("Grain Details")
        print(f"Number of Grains: {self.solid_motor.grain_number}")
        print(f"Grain Spacing: {self.solid_motor.grain_separation} m")
        print(f"Grain Density: {self.solid_motor.grain_density} kg/m3")
        print(f"Grain Outer Radius: {self.solid_motor.grain_outer_radius} m")
        print(f"Grain Inner Radius: {self.solid_motor.grain_initial_inner_radius} m")
        print(f"Grain Height: {self.solid_motor.grain_initial_height} m")
        print(f"Grain Volume: {self.solid_motor.grain_initial_volume:.3f} m3")
        print(f"Grain Mass: {self.solid_motor.grain_initial_mass:.3f} kg\n")

    def motor_details(self):
        """Prints out all data available about the SolidMotor.

        Returns
        -------
        None
        """
        print("Motor Details")
        print("Total Burning Time: " + str(self.solid_motor.burn_duration) + " s")
        print(
            f"Total Propellant Mass: {self.solid_motor.propellant_initial_mass:.3f} kg"
        )
        print(f"Structural Mass Ratio: {self.solid_motor.structural_mass_ratio:.3f}")
        average = self.solid_motor.exhaust_velocity.average(*self.solid_motor.burn_time)
        print(f"Average Propellant Exhaust Velocity: {average:.3f} m/s")
        print(f"Average Thrust: {self.solid_motor.average_thrust:.3f} N")
        print(
            f"Maximum Thrust: {self.solid_motor.max_thrust} N "
            f"at {self.solid_motor.max_thrust_time} s after ignition."
        )
        print(f"Total Impulse: {self.solid_motor.total_impulse:.3f} Ns\n")

    def all(self):
        """Prints out all data available about the SolidMotor.

        Returns
        -------
        None
        """
        self.nozzle_details()
        self.grain_details()
        self.motor_details()
