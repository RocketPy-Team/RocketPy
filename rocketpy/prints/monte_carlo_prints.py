import numpy as np

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
        print(f"{'Parameter':>25} {'Mean':>15} {'Std. Dev.':>15} {'95% PI Low':>15} {'95% PI High':>15}")
        print("-" * 90)
        for key, value in self.monte_carlo.processed_results.items():
            try:
                data = self.monte_carlo.results[key]
                pi_low = np.quantile(data, 0.025)
                pi_high = np.quantile(data, 0.975)
                print(f"{key:>25} {value[0]:>15.3f} {value[1]:>15.3f} {pi_low:>15.3f} {pi_high:>15.3f}")
            except TypeError:
                print(f"{key:>25} {str(value[0]):>15} {str(value[1]):>15} {'N/A':>15} {'N/A':>15}")
