from scipy.stats import norm

from rocketpy.tools import import_optional_dependency


class _SensitivityModelPrints:
    def __init__(self, model):
        self.model = model

    def _create_sensitivity_table(
        self, target_variable, digits, nominal_mean_text, nominal_sd_text, pt
    ):
        sensitivity_table = pt.PrettyTable()
        sensitivity_table.title = f"Summary {target_variable}"
        sensitivity_table.field_names = [
            "Parameter",
            "Sensitivity (%)",
            nominal_mean_text,
            nominal_sd_text,
            "Effect per sd",
        ]

        model = self.model.target_variables_info[target_variable]["model"]
        coef = model.params[1:]  # skipping intercept

        for i in range(self.model.n_parameters):
            parameter = self.model.parameters_names[i]
            beta = coef[i]
            effect_per_sd = beta * self.model.parameters_info[parameter]["nominal_sd"]
            sensitivity = self.model.target_variables_info[target_variable][
                "sensitivity"
            ][parameter]
            sensitivity_table.add_row(
                [
                    parameter,
                    round(100 * sensitivity, digits),
                    round(
                        self.model.parameters_info[parameter]["nominal_mean"], digits
                    ),
                    round(self.model.parameters_info[parameter]["nominal_sd"], digits),
                    round(effect_per_sd, digits),
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
            ]
        )
        sensitivity_table.sortby = "Sensitivity (%)"
        sensitivity_table.reversesort = True

        return sensitivity_table

    def _create_prediction_interval_table(self, target_variable, digits, alpha, pt):
        table = pt.PrettyTable()
        nominal_value = round(
            self.model.target_variables_info[target_variable]["nominal_value"], digits
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
            [f"{round(100 * alpha, 0)}% Prediction Interval: [{ci_lower}, {ci_upper}]"]
        )

        return table

    def summary(self, digits=4, alpha=0.95):
        """Formats parameter sensitivity information in a prettytable and prints it."""
        self.model._raise_error_if_not_fitted()
        pt = import_optional_dependency("prettytable")

        nominal_mean_text = (
            "Nominal mean"
            if self.model._nominal_parameters_passed
            else "Estimated mean"
        )
        nominal_sd_text = (
            "Nominal sd" if self.model._nominal_parameters_passed else "Estimated sd"
        )

        for target_variable in self.model.target_variables_names:
            sensitivity_table = self._create_sensitivity_table(
                target_variable, digits, nominal_mean_text, nominal_sd_text, pt
            )
            prediction_table = self._create_prediction_interval_table(
                target_variable, digits, alpha, pt
            )

            # Calculate column width based on the length of the string representation
            column_width = len(sensitivity_table.get_string().splitlines()[0])
            prediction_table.field_names = [
                (column_width - 4) * " "
            ]  # Make tables borders match

            print(sensitivity_table)
            print(prediction_table)

    def all(self):
        """Prints all sensitivity analysis plots"""
        self.summary()
