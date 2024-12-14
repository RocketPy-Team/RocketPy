import numpy as np

class _MonteCarloPrints:
    """Class to print the monte carlo analysis results."""

    def __init__(self, monte_carlo):
        self.monte_carlo = monte_carlo

    def all(self):
        """Print the mean, standard deviation, and quantiles (0%,  2.5%, 50%, 97.5%, 100%) 
        of each parameter in the results dictionary or of the variables passed as argument.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        print("Monte Carlo Simulation by RocketPy")
        print("Data Source: ", self.monte_carlo.filename)
        print("Number of Simulations: ", self.monte_carlo.num_of_loaded_sims)
        print("Results: \n")
                         
        print(f"{'Parameter':>25} {'Mean':>15} {'Std. Dev.':>15} {'0% Quant':>15} {'2.5% Quant.':>15} {'50% Quant.':>15} {'97.5% Quant.':>15} {'100% Quant.':>15}")
        print("-" * 60)
        for key, value in self.monte_carlo.processed_results.items():
            try:
                pt = self.monte_carlo.results[key]
                print (f"{key:>25} {value[0]:>15.3f} {value[1]:>15.3f} {np.quantile(pt,0):>15.3f} {np.quantile(pt,0.025):>15.3f} {np.quantile(pt,0.5):>15.3f} {np.quantile(pt,0.975):>15.3f} {np.quantile(pt,1):>15.3f}")
            except TypeError:
                print (f"{key:>25} {str(value[0]):>15} {str(value[1]):>15} {str(np.quantile(pt,0)):>15} {str(np.quantile(pt,0.025)):>15} {str(np.quantile(pt,0.5)):>15} {str(np.quantile(pt,0.975)):>15} {str(np.quantile(pt,1)):>15}")