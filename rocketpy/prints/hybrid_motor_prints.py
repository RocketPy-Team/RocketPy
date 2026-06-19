import logging

logger = logging.getLogger(__name__)

import numpy as np

from .motor_prints import _MotorPrints


class _HybridMotorPrints(_MotorPrints):
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
        super().__init__(hybrid_motor)
        self.hybrid_motor = hybrid_motor

    def nozzle_details(self):
        """Prints out all data available about the Nozzle.

        Returns
        -------
        None
        """
        # Print nozzle details
        logger.info("Nozzle Details")
        logger.info(f"Outlet Radius: {self.hybrid_motor.nozzle_radius} m")
        logger.info(f"Throat Radius: {self.hybrid_motor.solid.throat_radius} m")
        logger.info(f"Outlet Area: {np.pi * self.hybrid_motor.nozzle_radius**2:.6f} m²")
        logger.info(f"Throat Area: {np.pi * self.hybrid_motor.solid.throat_radius**2:.6f} m²")
        logger.info(f"Position: {self.hybrid_motor.nozzle_position} m\n")

    def grain_details(self):
        """Prints out all data available about the Grain.

        Returns
        -------
        None
        """
        logger.info("Grain Details")
        logger.info(f"Number of Grains: {self.hybrid_motor.solid.grain_number}")
        logger.info(f"Grain Spacing: {self.hybrid_motor.solid.grain_separation} m")
        logger.info(f"Grain Density: {self.hybrid_motor.solid.grain_density} kg/m3")
        logger.info(f"Grain Outer Radius: {self.hybrid_motor.solid.grain_outer_radius} m")
        logger.info(
            "Grain Inner Radius: "
            f"{self.hybrid_motor.solid.grain_initial_inner_radius} m"
        )
        logger.info(f"Grain Height: {self.hybrid_motor.solid.grain_initial_height} m")
        logger.info(f"Grain Volume: {self.hybrid_motor.solid.grain_initial_volume:.3f} m3")
        logger.info(f"Grain Mass: {self.hybrid_motor.solid.grain_initial_mass:.3f} kg\n")

    def all(self):
        """Prints out all data available about the HybridMotor.

        Returns
        -------
        None
        """

        self.nozzle_details()
        self.grain_details()
        self.motor_details()
