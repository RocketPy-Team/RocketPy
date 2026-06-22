from .motor_prints import _MotorPrints


class _SolidMotorPrints(_MotorPrints):
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
        super().__init__(solid_motor)
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

    def all(self):
        """Prints out all data available about the SolidMotor.

        Returns
        -------
        None
        """
        self.nozzle_details()
        self.grain_details()
        self.motor_details()
