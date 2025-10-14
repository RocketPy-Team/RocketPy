import warnings

from scipy.constants import atm, zero_Celsius


class _FluidPlots:
    """Class that holds plot methods for Fluid class.

    Attributes
    ----------
    _FluidPlots.fluid : Fluid
        Fluid object that will be used for the plots.

    """

    def __init__(self, fluid):
        """Initializes _MotorClass class.

        Parameters
        ----------
        fluid : Fluid
            Instance of the Fluid class

        Returns
        -------
        None
        """

        self.fluid = fluid

    def density_function(self, lower=None, upper=None):
        """Plots the density as a function of temperature in Kelvin
        and Pressure in Pascal.

        Parameters
        ----------
        lower: tuple
            Lower range of the temperature and pressure interval. If None
            default values are used.
        upper: tuple
            Upper range of the temperature and pressure interval. If None
            default values are used.
        """
        if lower is None:
            lower = (100, atm)
        if upper is None:
            upper = (zero_Celsius + 30, atm * 50)
        try:
            self.fluid.density_function.plot(lower, upper)
        except ValueError:
            warnings.warn("Invalid value while attempting density plot.")

    def all(self):
        """Prints out all graphs available about the Fluid. It simply calls
        all the other plotter methods in this class.

        Return
        ------
        None
        """
        self.density_function()
