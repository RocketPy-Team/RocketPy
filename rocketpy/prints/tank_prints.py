import logging

logger = logging.getLogger(__name__)

class _TankPrints:
    """Class that holds prints methods for Tank class.

    Attributes
    ----------
    _TankPrints.tank : tank
        Tank object that will be used for the prints.

    """

    def __init__(
        self,
        tank,
    ):
        """Initializes _TankPrints class

        Parameters
        ----------
        tank: Tank
            Instance of the Tank class.

        Returns
        -------
        None
        """
        self.tank = tank

    def fluid_parameters(self):
        """Prints out the fluid parameters of the Tank.

        Returns
        -------
        None
        """
        logger.info(f"Tank '{self.tank.name}' Fluid Parameters:")
        logger.info("\nLiquid Fluid")
        self.tank.liquid.prints.all()
        logger.info("\nGas Fluid")
        self.tank.gas.prints.all()

    def mass_flux(self):
        """Prints out the mass flux of the Tank.

        Returns
        -------
        None
        """
        initial_time, final_time = self.tank.flux_time
        logger.info(f"\nTank '{self.tank.name}' Mass Flux Data:")
        logger.info(f"\nInitial Quantities at t = {initial_time:.2f} s:")
        logger.info(f"Initial Fluid Mass: {self.tank.fluid_mass(initial_time):.3e} kg")
        logger.info(f"Initial Liquid Volume: {self.tank.liquid_volume(initial_time):.3e} m^3")
        logger.info(f"Initial Liquid Level: {self.tank.liquid_height(initial_time):.3e} m")
        logger.info(f"\nFinal Quantities at t = {final_time:.2f} s:")
        logger.info(f"Final Fluid Mass: {self.tank.fluid_mass(final_time):.3e} kg")
        logger.info(f"Final Liquid Volume: {self.tank.liquid_volume(final_time):.3e} m^3")
        logger.info(f"Final Liquid Level: {self.tank.liquid_height(final_time):.3e} m")

    def all(self):
        """Prints out all data available about the Tank.

        Returns
        -------
        None
        """
        logger.info(f"\nTank '{self.tank.name}' Data:\n")
        self.tank.geometry.prints.all()
        self.fluid_parameters()
        self.mass_flux()
