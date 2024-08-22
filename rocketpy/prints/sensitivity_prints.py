from scipy.stats import norm

from rocketpy.tools import import_optional_dependency


class _SensitivityModelPrints:

    def __init__(self, model):
        self.model = model

    def summary(self, digits=4, alpha=0.95):
        """Formats parameter sensitivity information in a prettytable
        and prints it

        Parameters
        ----------
        digits : int, optional
            Number of decimal digits printed on tables, by default 4
        alpha: float, optional
            Significance level used for prediction intervals, by default 0.95

        Returns
        -------
        None
        """

        self.model._raise_error_if_not_fitted()

        pt = import_optional_dependency("prettytable")

        if self.model._nominal_parameters_passed:
            nominal_mean_text = "Nominal mean"
            nominal_sd_text = "Nominal sd"
        else:
            nominal_mean_text = "Estimated mean"
            nominal_sd_text = "Estimated sd"
        for target_variable in self.model.target_variables_names:
            model = self.model.target_variables_info[target_variable]["model"]
            coef = model.params[1:]  # skipping intercept
            p_values = model.pvalues

            sensitivity_table = pt.PrettyTable()
            sensitivity_table.title = f"Summary {target_variable}"

            sensitivity_table.field_names = [
                "Parameter",
                "Sensitivity (%)",
                nominal_mean_text,
                nominal_sd_text,
                "Regression Coefficient",
                "p-value",
            ]

            for i in range(self.model.n_parameters):
                parameter = self.model.parameters_names[i]
                beta = coef[i]
                p_val = p_values[i]
                sensitivity = self.model.target_variables_info[target_variable][
                    "sensitivity"
                ][parameter]
                sensitivity_table.add_row(
                    [
                        parameter,
                        round(100 * sensitivity, digits),
                        round(
                            self.model.parameters_info[parameter]["nominal_mean"],
                            digits,
                        ),
                        round(
                            self.model.parameters_info[parameter]["nominal_sd"], digits
                        ),
                        round(beta, digits),
                        round(p_val, digits),
                    ]
                )
            sensitivity_table.add_row(
                [
                    "Linear Approx. Error (LAE)",
                    round(
                        100 * self.model.target_variables_info[target_variable]["LAE"],
                        digits,
                    ),
                    "",
                    "",
                    "",
                    "",
                ]
            )
            sensitivity_table.sortby = "Sensitivity (%)"
            sensitivity_table.reversesort = True

            print(sensitivity_table)

            table = pt.PrettyTable()
            nominal_value = round(
                self.model.target_variables_info[target_variable]["nominal_value"],
                digits,
            )
            norm_quantile = norm.ppf((1 + alpha) / 2)
            if self.model._nominal_target_passed:
                table.add_row([f"Nominal value: {nominal_value}"])
            else:
                table.add_row([f"Estimated value: {nominal_value}"])
            target_sd = self.model.target_variables_info[target_variable]["sd"]
            table.add_row([f"Std: {round(target_sd, digits)}"])
            ci_lower = round(nominal_value - norm_quantile * target_sd, digits)
            ci_upper = round(nominal_value + norm_quantile * target_sd, digits)
            table.add_row(
                [
                    f"{round(100 * alpha, 0)}% Prediction Interval: [{ci_lower}, {ci_upper}]"
                ]
            )
            column_width = len(sensitivity_table._hrule)
            # Make tables borders match
            table.field_names = [(column_width - 4) * " "]
            print(table)

    def all(self):
        """Prints all sensitivity analysis plots

        Returns
        -------
        None
        """
        self.summary()
