import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from ..tools import check_requirement_version, import_optional_dependency


class SensitivityModel:
    """Performs a 'local variance based first-order
    sensitivity analysis' considering independent input parameters.

    The core reference for global variance based sensitivity analysis is
    [1]. Our method implements a local version that only considers first
    order terms, which correspond to linear terms. Albeit the flight
    function is nonlinear, the linear hypothesis might be adequate when
    performing *local* sensitivity analysis.

    The model is fit using separate multiple linear regression for each
    target variable passed and using the parameters as covariates.

    References
    ----------
    [1] Sobol, Ilya M. "Global sensitivity indices for nonlinear mathematical models and their Monte Carlo estimates." Mathematics and computers in simulation 55.1-3 (2001): 271-280.
    """

    def __init__(
        self,
        parameters_names: list[str],
        target_variables_names: list[str],
    ):
        self.__check_requirements()
        self.n_parameters = len(parameters_names)
        self.parameters_names = parameters_names
        self.parameters_info = {
            parameter: {
                "nominal_mean": None,
                "nominal_sd": None,
            }
            for parameter in parameters_names
        }

        self.n_target_variables = len(target_variables_names)
        self.target_variables_names = target_variables_names
        self.target_variables_info = {
            variable: {
                "nominal_value": None,
                "sd": None,
                "var": None,
                "model": None,
                "sensitivity": {parameter: None for parameter in self.parameters_names},
                "LAE": None,  # Linear Approximation Error
            }
            for variable in target_variables_names
        }

        self.number_of_samples = None

        # Flags for nominal parameters
        self._nominal_parameters_passed = False
        self._nominal_target_passed = False

        self._fitted = False

        return

    def set_parameters_nominal(
        self,
        parameters_nominal_mean: np.array,
        parameters_nominal_sd: np.array,
    ):
        """Set parameters nominal mean and standard deviation

        Parameters
        ----------
        parameters_nominal_mean : np.array
            An array contaning the nominal mean for parameters in the
            order specified in parameters names at initialization
        parameters_nominal_sd : np.array
            An array contaning the nominal standard deviation for
            parameters in the order specified in parameters names at
            initialization
        """
        if len(parameters_nominal_mean) != self.n_parameters:
            raise ValueError(
                "Nominal mean array length does not match number of parameters passed at initilization."
            )
        if len(parameters_nominal_sd) != self.n_parameters:
            raise ValueError(
                "Nominal sd array length does not match number of parameters passed at initilization."
            )

        for i in range(self.n_parameters):
            parameter = self.parameters_names[i]
            self.parameters_info[parameter]["nominal_mean"] = parameters_nominal_mean[i]
            self.parameters_info[parameter]["nominal_sd"] = parameters_nominal_sd[i]

        self._nominal_parameters_passed = True

        return

    def set_target_variables_nominal(
        self,
        target_variables_nominal_value: np.array,
    ):
        """Set target variables nominal value (mean)

        Parameters
        ----------
        target_variables_nominal_value: np.array
            An array contaning the nominal mean for target variables in
            the order specified in target variables names at
            initialization
        """
        if len(target_variables_nominal_value) != self.n_target_variables:
            raise ValueError(
                "Target variables array length does not match number of target variables passed at initilization."
            )
        for i in range(self.n_target_variables):
            target_variable = self.target_variables_names[i]
            self.target_variables_info[target_variable][
                "nominal_value"
            ] = target_variables_nominal_value[i]

        self._nominal_target_passed = True

        return

    def _estimate_parameter_nominal(
        self,
        parameters_matrix: np.matrix,
    ):
        """Estimates parameters nominal values

        Parameters
        ----------
        parameters_matrix : np.matrix
            Data matrix whose columns correspond to parameters values
            ordered as passed in initialization

        """
        if parameters_matrix.shape[1] != self.n_parameters:
            raise ValueError(
                "Number of columns (parameters) does not match number of parameters passed at initialization."
            )
        for i in range(self.n_parameters):
            parameter = self.parameters_names[i]
            self.parameters_info[parameter]["nominal_mean"] = np.mean(
                parameters_matrix[:, i]
            )
            self.parameters_info[parameter]["nominal_sd"] = np.std(
                parameters_matrix[:, i]
            )

        return

    def _estimate_target_nominal(
        self,
        target_data: np.matrix,
    ):
        """Estimates target variables nominal values

        Parameters
        ----------
        target_data : np.array | np.matrix
            Data matrix or array. In the case of a matrix, the columns
            correspond to target variable values ordered as passed in
            initialization

        """
        if target_data.ndim == 1:
            if self.n_target_variables > 1:
                raise ValueError(
                    "Single target variable passed but more than one target variable was passed at initialization."
                )
            target_variable = self.target_variables_names[0]
            self.target_variables_info[target_variable]["nominal_value"] = np.mean(
                target_data[:]
            )

        else:
            if len(target_variable.shape[1]) != self.n_target_variables:
                raise ValueError(
                    "Number of columns (variables) does not match number of target variables passed at initilization."
                )
            for i in range(self.n_target_variables):
                target_variable = self.target_variables_names[i]
                self.target_variables_info[target_variable]["nominal_value"] = np.mean(
                    target_data[:, i]
                )

        return

    def fit(
        self,
        parameters_matrix: np.matrix,
        target_data: np.matrix,
    ):
        """Fits sensitivity model

        Parameters
        ----------
        parameters_matrix : np.matrix
            Data matrix whose columns correspond to parameters values
            ordered as passed in initialization

        target_data : np.array | np.matrix
            Data matrix or array. In the case of a matrix, the columns
            correspond to target variable values ordered as passed in
            initialization
        """
        # imports statsmodels for OLS method
        sm = import_optional_dependency("statsmodels.api")

        # Checks if data is in conformity with initialization info
        self.__check_conformity(parameters_matrix, target_data)

        # If nominal parameters are not set previous to fit, then we
        # must estimate them
        if not self._nominal_parameters_passed:
            self._estimate_parameter_nominal(parameters_matrix)
        if not self._nominal_target_passed:
            self._estimate_target_nominal(target_data)

        self.number_of_samples = parameters_matrix.shape[0]

        # Estimation setup
        parameters_mean = np.empty(self.n_parameters)
        parameters_sd = np.empty(self.n_parameters)
        for i in range(self.n_parameters):
            parameter = self.parameters_names[i]
            parameters_mean[i] = self.parameters_info[parameter]["nominal_mean"]
            parameters_sd[i] = self.parameters_info[parameter]["nominal_sd"]

        offset_matrix = np.repeat(parameters_mean, self.number_of_samples)
        offset_matrix = offset_matrix.reshape(
            self.n_parameters, self.number_of_samples
        ).T
        X = parameters_matrix - offset_matrix

        # When target data is a 1d-array, transform to 2d-array
        if target_data.ndim == 1:
            target_data = target_data.reshape(self.number_of_samples, 1)

        # Estimation
        for i in range(self.n_target_variables):
            target_variable = self.target_variables_names[i]
            nominal_value = self.target_variables_info[target_variable]["nominal_value"]
            Y = np.array(target_data[:, i] - nominal_value)
            ols_model = sm.OLS(Y, X)
            fitted_model = ols_model.fit()
            self.target_variables_info[target_variable]["model"] = fitted_model

            # Compute sensitivity
            beta = fitted_model.params
            sd_eps = fitted_model.scale
            var_Y = sd_eps**2
            for k in range(self.n_parameters):
                parameter = self.parameters_names[k]
                sensitivity = np.power(beta[k], 2) * np.power(parameters_sd[k], 2)
                self.target_variables_info[target_variable]["sensitivity"][
                    parameter
                ] = sensitivity
                var_Y += sensitivity

            self.target_variables_info[target_variable]["var"] = var_Y
            self.target_variables_info[target_variable]["sd"] = np.sqrt(var_Y)

            for k in range(self.n_parameters):
                parameter = self.parameters_names[k]
                self.target_variables_info[target_variable]["sensitivity"][
                    parameter
                ] /= var_Y
            self.target_variables_info[target_variable]["LAE"] = sd_eps**2
            self.target_variables_info[target_variable]["LAE"] /= var_Y

        self._fitted = True
        return

    def plot(self, target_variable="all"):
        self.__check_if_fitted()

        if (target_variable not in self.target_variables_names) and (
            target_variable != "all"
        ):
            raise ValueError(
                f"Target variable {target_variable} was not listed in initialization!"
            )

        # Parameters bars are blue colored
        # LAE bar is red colored
        bar_colors = self.n_parameters * ["blue"]
        bar_colors.append("red")

        if target_variable == "all":
            fig, axs = plt.subplots(self.n_target_variables, 1, sharex=True)
            fig.supxlabel("Parameters and LAE")
            fig.supylabel("Sensitivity (%)")
            x = [parameter for parameter in self.parameters_names]
            x.append("LAE")
            for i in range(self.n_target_variables):
                current_target_variable = self.target_variables_names[i]
                y = np.empty(self.n_parameters + 1)
                for j in range(self.n_parameters):
                    parameter = x[j]
                    y[j] = (
                        100
                        * self.target_variables_info[current_target_variable][
                            "sensitivity"
                        ][parameter]
                    )
                y[self.n_parameters] = (
                    100 * self.target_variables_info[current_target_variable]["LAE"]
                )
                axs[i].bar(x, y, color=bar_colors)
                axs[i].set_title(current_target_variable)

            return

        fig, axs = plt.subplots()
        fig.supxlabel("Parameters and LAE")
        fig.supylabel("Sensitivity (%)")
        x = [parameter for parameter in self.parameters_names]
        x.append("LAE")
        y = np.empty(self.n_parameters + 1)
        for j in range(self.n_parameters):
            parameter = x[j]
            y[j] = (
                100
                * self.target_variables_info[target_variable]["sensitivity"][parameter]
            )
        y[self.n_parameters] = 100 * self.target_variables_info[target_variable]["LAE"]
        axs.bar(x, y, color=bar_colors)
        axs.set_title(target_variable)

        return

    def summary(self, digits=4, alpha=0.95) -> None:
        """Formats parameter sensitivity information in a prettytable
        and prints it

        Parameters
        ----------
        digits : int, optional
            Number of decimal digits printed on tables, by default 4
        alpha: float, optional
            Significance level used for prediction intervals, by default 0.95
        """

        self.__check_if_fitted()

        pt = import_optional_dependency("prettytable")

        if self._nominal_parameters_passed:
            nominal_mean_text = "Nominal mean"
            nominal_sd_text = "Nominal sd"
        else:
            nominal_mean_text = "Estimated mean"
            nominal_sd_text = "Estimated sd"
        for target_variable in self.target_variables_names:
            model = self.target_variables_info[target_variable]["model"]
            coef = model.params
            pvalues = model.pvalues

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

            for i in range(self.n_parameters):
                parameter = self.parameters_names[i]
                beta = coef[i]
                pval = pvalues[i]
                sensitivity = self.target_variables_info[target_variable][
                    "sensitivity"
                ][parameter]
                sensitivity_table.add_row(
                    [
                        parameter,
                        round(100 * sensitivity, digits),
                        round(self.parameters_info[parameter]["nominal_mean"], digits),
                        round(self.parameters_info[parameter]["nominal_sd"], digits),
                        round(beta, digits),
                        round(pval, digits),
                    ]
                )
            sensitivity_table.add_row(
                [
                    "Linear Approx. Error (LAE)",
                    round(
                        100 * self.target_variables_info[target_variable]["LAE"],
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
                self.target_variables_info[target_variable]["nominal_value"], digits
            )
            norm_quantile = norm.ppf((1 + alpha) / 2)
            if self._estimate_target_nominal:
                table.add_row([f"Nominal value: {nominal_value}"])
            else:
                table.add_row([f"Estimated value: {nominal_value}"])
            target_sd = self.target_variables_info[target_variable]["sd"]
            total_variance = np.power(target_sd, 2)
            table.add_row([f"Variance: {round(total_variance, digits)}"])
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

        return

    def __check_conformity(
        self,
        parameters_matrix: np.matrix,
        target_data: np.matrix,
    ):
        """Checks if matrices used for fitting conform with the
        information passed at initialization

        Parameters
        ----------
        parameters_matrix : np.matrix
            Data matrix whose columns correspond to parameters values
            ordered as passed in initialization
        target_data : np.array | np.matrix
            Data matrix or array. In the case of a matrix, the columns
            correspond to target variable values ordered as passed in
            initialization

        """
        if parameters_matrix.shape[1] != self.n_parameters:
            raise ValueError(
                "Number of columns (parameters) does not match number of parameters passed at initialization."
            )
        if target_data.ndim == 1:
            n_samples_y = len(target_data)
            if self.n_target_variables > 1:
                raise ValueError(
                    "Single target variable passed but more than one target variable was passed at initialization."
                )
        else:
            n_samples_y = target_data.shape[0]
            if target_data.shape[1] != self.n_target_variables:
                raise ValueError(
                    "Number of columns (variables) does not match number of target variables passed at initilization."
                )
        if n_samples_y != parameters_matrix.shape[0]:
            raise ValueError(
                "Number of samples does not match between parameter matrix and target data."
            )

        return

    def __check_if_fitted(self):
        if not self._fitted:
            raise Exception("SensitivityModel must be fitted!")
        return

    def __check_requirements(self):
        """Check if extra requirements are installed. If not, print a message
        informing the user that some methods may not work and how to install
        the extra requirements.

        Returns
        -------
        None
        """
        sensitivity_require = {  # The same as in the setup.py file
            "statsmodels": "",
            "prettytable": "",
        }
        has_error = False
        for module_name, version in sensitivity_require.items():
            version = ">=0" if not version else version
            try:
                check_requirement_version(module_name, version)
            except (ValueError, ImportError) as e:
                has_error = True
                print(
                    f"The following error occurred while importing {module_name}: {e}"
                )
        if has_error:
            print(
                "Given the above errors, some methods may not work. Please run "
                + "'pip install rocketpy[sensitivity]' to install extra requirements."
            )
        return None
