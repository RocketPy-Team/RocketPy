class _MonteCarloPrints:
    """Class to print the monte carlo analysis results."""

    def __init__(self, monte_carlo):
        self.monte_carlo = monte_carlo

    def all(self):
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
        print(f"{'Parameter':>25} {'Mean':>15} {'Std. Dev.':>15}")
        print("-" * 60)
        for key, value in self.monte_carlo.processed_results.items():
            try:
                print(f"{key:>25} {value[0]:>15.3f} {value[1]:>15.3f}")
            except TypeError:
                print(f"{key:>25} {str(value[0]):>15} {str(value[1]):>15}")
