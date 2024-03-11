from statsmodels.api import OLS
from scipy.stats import norm
from sensitivity_analysis_utils import SimulationInfo
import numpy as np
from prettytable import PrettyTable


class TargetSensivityModel:
    """Performs sensitivity analysis from simulation data for specific
    target variables. Currently analyze each variable individually.
    Attributes
    ----------

    """

    def __init__(self, target_variables_names: list[str]) -> None:
        """Initialize target sensitivity model

        Parameters
        ----------
        target_variables : list
            List of strings contaning variables names that will be
            analyzed.

        """

        self.target_variable_names = target_variables_names
        self.info = {}

        for variable_name in self.target_variable_names:
            self.info[variable_name] = {
                "model": None,
                "results": None,
                "importance": None,
                "sd": None,
            }

        self.nominal = None
        self.parameters = None

        return

    def fit(self, simulation_info: SimulationInfo):
        """Fits a sensitivity analysis parameter importance model

        Parameters
        ----------
        simulation_info : SimulationInfo
            Dataclass containing the information from the simulations
        """
        self.nominal = simulation_info.nominal
        self.parameters = list(simulation_info.parameters_df.columns)
        self.parameters_mean = simulation_info.nominal["parameters_mean"]
        self.parameters_sd = simulation_info.nominal["parameters_sd"]
        self.nominal_target_value = {}

        ### Fits one regression model for each target variable
        # The X matrix is the same for all target variables
        nominal_parameter = np.array(
            list(simulation_info.nominal["parameters_mean"].values()), dtype=float
        )
        X = simulation_info.parameters_df
        X = X - nominal_parameter
        X = X.astype(float)
        for target_variable in self.target_variable_names:
            nominal_target = simulation_info.nominal["target_variables"][
                target_variable
            ]
            self.nominal_target_value[target_variable] = nominal_target
            Y = simulation_info.target_variables_df.loc[:, target_variable]
            Y = np.array(Y - nominal_target, dtype=float)
            self.info[target_variable]["model"] = OLS(Y, X)
            self.info[target_variable]["results"] = self.info[target_variable][
                "model"
            ].fit()

            ### Computes parameter importance
            coef = self.info[target_variable]["results"].params
            # scale attribute is the sd from the linear regression in api.OLS
            sd = self.info[target_variable]["results"].scale
            self.info[target_variable]["importance"] = {}
            total_variance = np.power(sd, 2)

            for parameter in self.parameters:
                importance = np.power(coef[parameter], 2) * np.power(
                    self.parameters_sd[parameter], 2
                )
                total_variance += importance
                self.info[target_variable]["importance"][parameter] = importance

            for parameter in self.parameters:
                self.info[target_variable]["importance"][parameter] *= (
                    100 / total_variance
                )

            self.info[target_variable]["model_error"] = (
                100 * np.power(sd, 2) / total_variance
            )
            self.info[target_variable]["sd"] = np.sqrt(total_variance)
            ###

        ###

        return

    def summary(self, digits=4, alpha=0.95) -> None:
        """Formats Parameter Importance information in a prettytable
        and prints it

        Parameters
        ----------
        digits : int, optional
            Number of decimal digits printed on tables, by default 4
        alpha: float, optional
            Significance level used for prediction intervals, by default 0.95
        """
        for target_variable in self.target_variable_names:
            results = self.info[target_variable]["results"]
            coef = results.params
            pvalues = results.pvalues

            importance_table = PrettyTable()
            importance_table.title = f"Summary {target_variable}"
            importance_table.field_names = [
                "Parameter",
                "Importance (%)",
                "Nominal mean",
                "Nominal sd",
                "Regression Coefficient",
                "p-value",
            ]
            for parameter in self.parameters:
                importance = self.info[target_variable]["importance"][parameter]
                importance_table.add_row(
                    [
                        parameter,
                        round(importance, digits),
                        round(self.parameters_mean[parameter], digits),
                        round(self.parameters_sd[parameter], digits),
                        round(coef[parameter], digits),
                        round(pvalues[parameter], digits),
                    ]
                )
            importance_table.add_row(
                [
                    "Linear Approx. Error (LAE)",
                    round(self.info[target_variable]["model_error"], digits),
                    "",
                    "",
                    "",
                    "",
                ]
            )
            importance_table.sortby = "Importance (%)"
            importance_table.reversesort = True

            print(importance_table)

            table = PrettyTable()
            nominal_value = round(self.nominal_target_value[target_variable], digits)
            target_sd = self.info[target_variable]["sd"]
            norm_quantile = norm.ppf((1 + alpha) / 2)
            table.add_row([f"Nominal value: {nominal_value}"])
            total_variance = np.power(self.info[target_variable]["sd"], 2)
            table.add_row([f"Variance: {round(total_variance, digits)}"])
            ci_lower = round(nominal_value - norm_quantile * target_sd, digits)
            ci_upper = round(nominal_value + norm_quantile * target_sd, digits)
            table.add_row(
                [
                    f"{round(100 * alpha, 0)}% Prediction Interval: [{ci_lower}, {ci_upper}]"
                ]
            )
            column_width = len(importance_table._hrule)
            # Make tables borders match
            table.field_names = [(column_width - 4) * " "]
            print(table)

        return


class TrajectorySensivityModel:
    """Performs sensitivity analysis from simulation data for the whole
    trajectory. Not Implemented. Baby steps.
    """

    def __init__(self, target_variables: list) -> None:
        """Initializes sensitivity model

        Args:
            target_variable (list): _description_
        """
        pass

    def fit(self):
        pass
