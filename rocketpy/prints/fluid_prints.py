from ..mathutils.function import NUMERICAL_TYPES


class _FluidPrints:
    """Class that holds prints methods for Fluid class.

    Attributes
    ----------
    _FluidPrints.fluid : fluid
        Fluid object that will be used for the prints.

    """

    def __init__(
        self,
        fluid,
    ):
        """Initializes _FluidPrints class

        Parameters
        ----------
        fluid: Fluid
            Instance of the Fluid class.

        Returns
        -------
        None
        """
        self.fluid = fluid

    def all(self):
        """Prints out all data available about the Fluid.

        Returns
        -------
        None
        """
        print(f"Name: {self.fluid.name}")
        if isinstance(self.fluid.density, NUMERICAL_TYPES):
            print(f"Density: {self.fluid.density:.4f} kg/m^3")
        else:
            print(f"Density: {self.fluid.density_function}")
