import logging

logger = logging.getLogger(__name__)

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
        logger.info("Monte Carlo Simulation by RocketPy")
        logger.info("Data Source: ", self.monte_carlo.filename)
        logger.info("Number of simulations: ", self.monte_carlo.num_of_loaded_sims)
        logger.info("Results: \n")
        logger.info(
            f"{'Parameter':>25} {'Mean':>15} {'Median':>15} {'Std. Dev.':>15} {'95% PI Lower':>15} {'95% PI Upper':>15}"
        )
        logger.info("-" * 110)
        for key, value in self.monte_carlo.processed_results.items():
            try:
                logger.info(
                    f"{key:>25} {value[0]:>15.3f} {value[1]:>15.3f} {value[2]:>15.3f} {value[3]:>15.3f} {value[4]:>15.3f}"
                )
            except TypeError:
                logger.info(
                    f"{key:>25} {'N/A':>15} {'N/A':>15} {'N/A':>15} {'N/A':>15} {'N/A':>15}"
                )

    def print_comparison(self, other_monte_carlo):
        """Print the mean and standard deviation of each parameter in the results
        dictionary or of the variables passed as argument.

        Parameters
        ----------
        other_monte_carlo : MonteCarlo
            MonteCarlo object which the current one will be compared to.

        Returns
        -------
        None

        """
        # TODO: understand why this validation is failing
        # if not isinstance(other_monte_carlo, MonteCarlo):
        #     raise TypeError(
        #         "Argument 'other_monte_carlo' must be an MonteCarlo object!"
        #     )
        original_parameters_set = set(self.monte_carlo.processed_results.keys())
        other_parameters_set = set(other_monte_carlo.processed_results.keys())
        intersection_set = original_parameters_set.intersection(other_parameters_set)
        symmetric_diff_set = original_parameters_set.symmetric_difference(
            other_parameters_set
        )
        logger.info("Comparison of Monte Carlo Simulation by RocketPy")
        logger.info("Original data Source: ", self.monte_carlo.filename)
        logger.info("Comparison data Source: ", other_monte_carlo.filename)
        logger.info("Original number of simulations: ", self.monte_carlo.num_of_loaded_sims)
        logger.info(
            "Comparison number of simulations: ", other_monte_carlo.num_of_loaded_sims
        )
        if len(symmetric_diff_set) > 0:
            logger.info(
                f"The following parameters were not in both simulations: {symmetric_diff_set}\n"
            )
        logger.info("Results: \n")
        logger.info(
            f"{'Parameter':>35} {'Source':>15} {'Mean':>15} {'Median':>15} {'Std. Dev.':>15} {'95% PI Lower':>15} {'95% PI Upper':>15}"
        )
        logger.info("-" * 140)
        for parameter in intersection_set:
            original_value = self.monte_carlo.processed_results[parameter]
            try:
                logger.info(
                    f"{parameter:>35} {'Original':>15} {original_value[0]:>15.3f} {original_value[1]:>15.3f} {original_value[2]:>15.3f} {original_value[3]:>15.3f} {original_value[4]:>15.3f}"
                )
            except TypeError:
                logger.info(
                    f"{parameter:>35} {'Original':>15} {'N/A':>15} {'N/A':>15} {'N/A':>15} {'N/A':>15} {'N/A':>15}"
                )

            other_value = other_monte_carlo.processed_results[parameter]
            try:
                logger.info(
                    f"{parameter:>35} {'Other':>15} {other_value[0]:>15.3f} {other_value[1]:>15.3f} {other_value[2]:>15.3f} {other_value[3]:>15.3f} {other_value[4]:>15.3f}"
                )
            except TypeError:
                logger.info(
                    f"{parameter:>35} {'Other':>15} {'N/A':>15} {'N/A':>15} {'N/A':>15} {'N/A':>15} {'N/A':>15}"
                )
