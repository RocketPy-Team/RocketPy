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
        print(f"Tank '{self.tank.name}' Fluid Parameters:")
        print("\nLiquid Fluid")
        self.tank.liquid.prints.all()
        print("\nGas Fluid")
        self.tank.gas.prints.all()

    def mass_flux(self):
        """Prints out the mass flux of the Tank.

        Returns
        -------
        None
        """
        initial_time, final_time = self.tank.flux_time
        print(f"\nTank '{self.tank.name}' Mass Flux Data:")
        print(f"\nInitial Quantities at t = {initial_time:.2f} s:")
        print(f"Initial Fluid Mass: {self.tank.fluid_mass(initial_time):.3e} kg")
        print(f"Initial Liquid Volume: {self.tank.liquid_volume(initial_time):.3e} m^3")
        print(f"Initial Liquid Level: {self.tank.liquid_height(initial_time):.3e} m")
        print(f"\nFinal Quantities at t = {final_time:.2f} s:")
        print(f"Final Fluid Mass: {self.tank.fluid_mass(final_time):.3e} kg")
        print(f"Final Liquid Volume: {self.tank.liquid_volume(final_time):.3e} m^3")
        print(f"Final Liquid Level: {self.tank.liquid_height(final_time):.3e} m")

    def all(self):
        """Prints out all data available about the Tank.

        Returns
        -------
        None
        """
        print(f"\nTank '{self.tank.name}' Data:\n")
        self.tank.geometry.prints.all()
        self.fluid_parameters()
        self.mass_flux()
