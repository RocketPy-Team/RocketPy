import numpy as np

from rocketpy.plots.sensitivity_plots import _SensitivityModelPlots
from rocketpy.prints.sensitivity_prints import _SensitivityModelPrints
from rocketpy.tools import check_requirement_version, import_optional_dependency


class SensitivityModel:
    """Performs a 'local variance based first-order
    sensitivity analysis' considering independent input parameters.

    The main reference for global variance based sensitivity analysis is
    [1]. Our method implements a local version that only considers first
    order terms, which correspond to linear terms. Albeit the flight
    function is nonlinear, the linear hypothesis might be adequate when
    performing *local* sensitivity analysis.

    The model is fit using separate multiple linear regression for each
    target variable passed and using the parameters as covariates.

    References
    ----------
    [1] Sobol, Ilya M. "Global sensitivity indices for nonlinear mathematical
    models and their Monte Carlo estimates." Mathematics and computers
    in simulation 55.1-3 (2001): 271-280.
    """

    def __init__(
        self,
        parameters_names,
        target_variables_names,
    ):
        """Initializes sensitivity model

        Parameters
        ----------
        parameter_names: list[str]
            A list containing the names of the parameters used in the
            analysis. Note that the order is important and must match the
            order passed in the parameter data matrix.
        target_variables_names: list[str]
            A list containing the names of the target variables used in the
            analysis. Note that the order is important and must match the
            order passed in the target variables data matrix.

        Returns
        -------
        None
        """
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

        self.prints = _SensitivityModelPrints(self)
        self.plots = _SensitivityModelPlots(self)

    def set_parameters_nominal(self, parameters_nominal_mean, parameters_nominal_sd):
        """Set parameters nominal mean and standard deviation

        Parameters
        ----------
        parameters_nominal_mean : np.array
            An array containing the nominal mean for parameters in the
            order specified in parameters names at initialization
        parameters_nominal_sd : np.array
            An array containing the nominal standard deviation for
            parameters in the order specified in parameters names at
            initialization

        Returns
        -------
        None
        """
        if len(parameters_nominal_mean) != self.n_parameters:
            raise ValueError(
                "Nominal mean array length does not match number of "
                "parameters passed at initialization."
            )
        if len(parameters_nominal_sd) != self.n_parameters:
            raise ValueError(
                "Nominal sd array length does not match number of parameters "
                "passed at initialization."
            )

        for i, parameter in enumerate(self.parameters_names):
            self.parameters_info[parameter]["nominal_mean"] = parameters_nominal_mean[i]
            self.parameters_info[parameter]["nominal_sd"] = parameters_nominal_sd[i]

        self._nominal_parameters_passed = True

    def set_target_variables_nominal(self, target_variables_nominal_value):
        """Set target variables nominal value (mean)

        Parameters
        ----------
        target_variables_nominal_value: np.array
            An array containing the nominal mean for target variables in
            the order specified in target variables names at
            initialization

        Returns
        -------
        None
        """
        if len(target_variables_nominal_value) != self.n_target_variables:
            raise ValueError(
                "Target variables array length does not match number of \
                 target variables passed at initialization."
            )
        for i, target_variable in enumerate(self.target_variables_names):
            self.target_variables_info[target_variable]["nominal_value"] = (
                target_variables_nominal_value[i]
            )
        for i, target_variable in enumerate(self.target_variables_names):
            self.target_variables_info[target_variable]["nominal_value"] = (
                target_variables_nominal_value[i]
            )

        self._nominal_target_passed = True

    def _estimate_parameter_nominal(self, parameters_matrix):
        """Estimates parameters nominal values

        Parameters
        ----------
        parameters_matrix : np.matrix
            Data matrix whose columns correspond to parameters values
            ordered as passed in initialization

        Returns
        -------
        None
        """

        for i, parameter in enumerate(self.parameters_names):
            self.parameters_info[parameter]["nominal_mean"] = np.mean(
                parameters_matrix[:, i]
            )
            self.parameters_info[parameter]["nominal_sd"] = np.std(
                parameters_matrix[:, i]
            )

    def _estimate_target_nominal(self, target_data):
        """Estimates target variables nominal values

        Parameters
        ----------
        target_data : np.array | np.matrix
            Data matrix or array. In the case of a matrix, the columns
            correspond to target variable values ordered as passed in
            initialization

        Returns
        -------
        None
        """
        if target_data.ndim == 1:
            target_variable = self.target_variables_names[0]
            self.target_variables_info[target_variable]["nominal_value"] = np.mean(
                target_data[:]
            )

        else:
            for i, target_variable in enumerate(self.target_variables_names):
                self.target_variables_info[target_variable]["nominal_value"] = np.mean(
                    target_data[:, i]
                )

    def fit(self, parameters_matrix, target_data):
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

        Returns
        -------
        None
        """
        sm = import_optional_dependency("statsmodels.api")

        self.__check_conformity(parameters_matrix, target_data)

        # If nominal parameters are not set previous to fit, we must estimate them
        if not self._nominal_parameters_passed:
            self._estimate_parameter_nominal(parameters_matrix)
        if not self._nominal_target_passed:
            self._estimate_target_nominal(target_data)

        self.number_of_samples = parameters_matrix.shape[0]

        # Estimation setup
        parameters_mean = np.empty(self.n_parameters)
        parameters_sd = np.empty(self.n_parameters)
        for i, parameter in enumerate(self.parameters_names):
            parameters_mean[i] = self.parameters_info[parameter]["nominal_mean"]
            parameters_sd[i] = self.parameters_info[parameter]["nominal_sd"]

        offset_matrix = np.repeat(parameters_mean, self.number_of_samples)
        offset_matrix = offset_matrix.reshape(
            self.n_parameters, self.number_of_samples
        ).T
        X = parameters_matrix - offset_matrix
        X = sm.add_constant(X)
        # When target data is a 1d-array, transform to 2d-array
        if target_data.ndim == 1:
            target_data = target_data.reshape(self.number_of_samples, 1)

        self.__estimation_loop(target_data, sm, parameters_sd, X)

        self._fitted = True

    def __estimation_loop(self, target_data, sm, parameters_sd, X):
        for i, target_variable in enumerate(self.target_variables_names):
            nominal_value = self.target_variables_info[target_variable]["nominal_value"]
            Y = np.array(target_data[:, i] - nominal_value)
            ols_model = sm.OLS(Y, X)
            fitted_model = ols_model.fit()
            self.target_variables_info[target_variable]["model"] = fitted_model

            # Compute sensitivity
            beta = fitted_model.params[1:]  # skipping the intercept
            var_eps = fitted_model.scale
            var_y = var_eps
            for k, parameter in enumerate(self.parameters_names):
                sensitivity = np.power(beta[k], 2) * np.power(parameters_sd[k], 2)
                self.target_variables_info[target_variable]["sensitivity"][
                    parameter
                ] = sensitivity
                var_y += sensitivity

            self.target_variables_info[target_variable]["var"] = var_y
            self.target_variables_info[target_variable]["sd"] = np.sqrt(var_y)

            for k, parameter in enumerate(self.parameters_names):
                self.target_variables_info[target_variable]["sensitivity"][
                    parameter
                ] /= var_y
            self.target_variables_info[target_variable]["LAE"] = var_eps
            self.target_variables_info[target_variable]["LAE"] /= var_y

    def info(self):
        self.prints.all()

    def all_info(self):
        self.prints.all()
        self.plots.all()

    def __check_conformity(
        self,
        parameters_matrix,
        target_data,
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

        Returns
        -------
        None
        """
        if parameters_matrix.shape[1] != self.n_parameters:
            raise ValueError(
                "Number of columns (parameters) does not match number of \
                 parameters passed at initialization."
            )
        if target_data.ndim == 1:
            n_samples_y = len(target_data)
            if self.n_target_variables > 1:
                raise ValueError(
                    "Single target variable passed but more than one target \
                    variable was passed at initialization."
                )
        else:
            n_samples_y = target_data.shape[0]
            if target_data.shape[1] != self.n_target_variables:
                raise ValueError(
                    "Number of columns (variables) does not match number of \
                     target variables passed at initialization."
                )
        if n_samples_y != parameters_matrix.shape[0]:
            raise ValueError(
                "Number of samples does not match between parameter matrix \
                 and target data."
            )

    def _raise_error_if_not_fitted(self):
        """Checks if model is fitted or raise an error.

        Returns
        -------
        None
        """
        if not self._fitted:
            raise ValueError("SensitivityModel must be fitted!")

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
