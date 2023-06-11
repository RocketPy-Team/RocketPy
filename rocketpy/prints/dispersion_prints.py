__author__ = "Guilherme Fernandes Alves"
__copyright__ = "Copyright 20XX, RocketPy Team"
__license__ = "MIT"


class _DispersionPrints:
    """Class to print the dispersion results of the dispersion analysis."""

    def __init__(self, dispersion):
        self.dispersion = dispersion
        return None

    def print_results(self):
        """Print the mean and standard deviation of each parameter in the results
        dictionary or of the variables passed as argument.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        print("Monte Carlo Simulation by RocketPy")
        print("Data Source: ", self.dispersion.filename)
        print("Number of simulations: ", self.dispersion.num_of_loaded_sims)
        print("Results: \n")
        print("{:>25} {:>15} {:>15}".format("Parameter", "Mean", "Std. Dev."))
        print("-" * 60)
        for key, value in self.dispersion.processed_results.items():
            print("{:>25} {:>15.3f} {:>15.3f}".format(key, value[0], value[1]))

        return None
