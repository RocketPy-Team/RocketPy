import logging

logger = logging.getLogger(__name__)

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
        logger.info("Nozzle Details")
        logger.info(f"Nozzle Radius: {self.solid_motor.nozzle_radius} m")
        logger.info(f"Nozzle Throat Radius: {self.solid_motor.throat_radius} m\n")

    def grain_details(self):
        """Prints out all data available about the SolidMotor grain.

        Returns
        -------
        None
        """
        logger.info("Grain Details")
        logger.info(f"Number of Grains: {self.solid_motor.grain_number}")
        logger.info(f"Grain Spacing: {self.solid_motor.grain_separation} m")
        logger.info(f"Grain Density: {self.solid_motor.grain_density} kg/m3")
        logger.info(f"Grain Outer Radius: {self.solid_motor.grain_outer_radius} m")
        logger.info(f"Grain Inner Radius: {self.solid_motor.grain_initial_inner_radius} m")
        logger.info(f"Grain Height: {self.solid_motor.grain_initial_height} m")
        logger.info(f"Grain Volume: {self.solid_motor.grain_initial_volume:.3f} m3")
        logger.info(f"Grain Mass: {self.solid_motor.grain_initial_mass:.3f} kg\n")

    def all(self):
        """Prints out all data available about the SolidMotor.

        Returns
        -------
        None
        """
        self.nozzle_details()
        self.grain_details()
        self.motor_details()
