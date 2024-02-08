class _MonteCarloPrints:
    """Class to print the monte carlo analysis results."""

    def __init__(self, monte_carlo):
        self.monte_carlo = monte_carlo
        return None

    def all_results(self):
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
        print("Data Source: ", self.monte_carlo.filename)
        print("Number of simulations: ", self.monte_carlo.num_of_loaded_sims)
        print("Results: \n")
        print("{:>25} {:>15} {:>15}".format("Parameter", "Mean", "Std. Dev."))
        print("-" * 60)
        for key, value in self.monte_carlo.processed_results.items():
            print("{:>25} {:>15.3f} {:>15.3f}".format(key, value[0], value[1]))

        return None
