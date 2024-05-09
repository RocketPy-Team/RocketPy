import numpy as np
import matplotlib.pyplot as plt

try:
    from statsmodels.api import OLS
except ImportError:
    raise ImportError("ImportanceModel requires the 'statsmodels' package.")


class ImportanceModel:
    """ """

    def __init__(
        self,
        parameters_names: list[str],
        target_variables_names: list[str],
    ):
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
                "importance": {parameter: None for parameter in self.parameters_names},
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
            target_variable = self.target_variables_names[0]
            self.target_variables_info[target_variable]["nominal_value"] = np.mean(
                target_data[:]
            )

        else:
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
        """Fits importance model

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
        # Checks if data is in conformity with initialization info
        self._check_conformity(parameters_matrix, target_data)

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
            ols_model = OLS(Y, X)
            fitted_model = ols_model.fit()
            self.target_variables_info[target_variable]["model"] = fitted_model

            # Compute importance
            beta = fitted_model.params
            sd_eps = fitted_model.scale
            var_Y = sd_eps**2
            for k in range(self.n_parameters):
                parameter = self.parameters_names[k]
                importance = np.power(beta[k], 2) * np.power(parameters_sd[k], 2)
                self.target_variables_info[target_variable]["importance"][
                    parameter
                ] = importance
                var_Y += importance

            self.target_variables_info[target_variable]["var"] = var_Y
            self.target_variables_info[target_variable]["sd"] = np.sqrt(var_Y)

            for k in range(self.n_parameters):
                parameter = self.parameters_names[k]
                self.target_variables_info[target_variable]["importance"][
                    parameter
                ] /= var_Y
            self.target_variables_info[target_variable]["LAE"] = sd_eps**2
            self.target_variables_info[target_variable]["LAE"] /= var_Y

        self._fitted = True
        return

    def print(self):
        pass

    def plot(self, target_variable="all"):
        if not self._fitted:
            raise Exception("ImportanceModel must be fitted before plotting!")

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
            fig.supylabel("Importance (%)")
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
                            "importance"
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
        fig.supylabel("Importance (%)")
        x = [parameter for parameter in self.parameters_names]
        x.append("LAE")
        y = np.empty(self.n_parameters + 1)
        for j in range(self.n_parameters):
            parameter = x[j]
            y[j] = (
                100
                * self.target_variables_info[target_variable]["importance"][parameter]
            )
        y[self.n_parameters] = 100 * self.target_variables_info[target_variable]["LAE"]
        axs.bar(x, y, color=bar_colors)
        axs.set_title(target_variable)

        return

    def _check_conformity(
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
        pass
