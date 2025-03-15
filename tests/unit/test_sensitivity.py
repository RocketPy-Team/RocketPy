from unittest.mock import patch

import numpy as np
import pytest

from rocketpy.sensitivity import SensitivityModel


def test_initialization():
    parameters_names = ["param1", "param2"]
    target_variables_names = ["target1", "target2"]

    model = SensitivityModel(parameters_names, target_variables_names)

    assert model.n_parameters == 2
    assert model.parameters_names == parameters_names
    assert model.n_target_variables == 2
    assert model.target_variables_names == target_variables_names
    assert not model._fitted


def test_set_parameters_nominal():
    parameters_names = ["param1", "param2"]
    target_variables_names = ["target1", "target2"]
    model = SensitivityModel(parameters_names, target_variables_names)

    parameters_nominal_mean = np.array([1.0, 2.0])
    parameters_nominal_sd = np.array([0.1, 0.2])

    model.set_parameters_nominal(parameters_nominal_mean, parameters_nominal_sd)

    assert model.parameters_info["param1"]["nominal_mean"] == 1.0
    assert model.parameters_info["param2"]["nominal_sd"] == 0.2

    # check dimensions mismatch error raise
    incorrect_nominal_mean = np.array([1.0])
    with pytest.raises(ValueError):
        model.set_parameters_nominal(incorrect_nominal_mean, parameters_nominal_sd)

    incorrect_nominal_sd = np.array([0.1])
    with pytest.raises(ValueError):
        model.set_parameters_nominal(parameters_nominal_mean, incorrect_nominal_sd)


def test_set_target_variables_nominal():
    parameters_names = ["param1", "param2"]
    target_variables_names = ["target1", "target2"]
    model = SensitivityModel(parameters_names, target_variables_names)

    target_variables_nominal_value = np.array([10.0, 20.0])

    model.set_target_variables_nominal(target_variables_nominal_value)

    assert model.target_variables_info["target1"]["nominal_value"] == 10.0
    assert model.target_variables_info["target2"]["nominal_value"] == 20.0

    # check dimensions mismatch error raise
    incorrect_nominal_value = np.array([10.0])
    with pytest.raises(ValueError):
        model.set_target_variables_nominal(incorrect_nominal_value)


def test_fit_method_one_target():
    parameters_names = ["param1", "param2"]
    target_variables_names = ["target1"]
    model = SensitivityModel(parameters_names, target_variables_names)

    parameters_matrix = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    target_data = np.array([10.0, 12.0, 14.0])

    model.fit(parameters_matrix, target_data)

    assert model._fitted
    assert model.number_of_samples == 3


def test_fit_method_multiple_target():
    parameters_names = ["param1", "param2"]
    target_variables_names = ["target1", "target2"]
    model = SensitivityModel(parameters_names, target_variables_names)

    parameters_matrix = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    target_data = np.array([[10.0, 12.0, 14.0], [11.0, 13.0, 17.0]]).T

    model.fit(parameters_matrix, target_data)

    assert model._fitted
    assert model.number_of_samples == 3


def test_fit_raises_error_on_mismatched_dimensions():
    parameters_names = ["param1", "param2"]
    target_variables_names = ["target1"]
    model = SensitivityModel(parameters_names, target_variables_names)

    parameters_matrix = np.array([[1.0, 2.0], [2.0, 3.0]])
    target_data = np.array([10.0, 12.0, 14.0])

    with pytest.raises(ValueError):
        model.fit(parameters_matrix, target_data)


def test_check_conformity():
    parameters_names = ["param1", "param2"]
    target_variables_names = ["target1", "target2"]
    model = SensitivityModel(parameters_names, target_variables_names)

    parameters_matrix = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    target_data = np.array([[10.0, 20.0], [12.0, 22.0], [14.0, 24.0]])

    model._SensitivityModel__check_conformity(parameters_matrix, target_data)


def test_check_conformity_raises_error():
    parameters_names = ["param1", "param2"]
    target_variables_names = ["target1", "target2"]
    model = SensitivityModel(parameters_names, target_variables_names)

    parameters_matrix = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    target_data = np.array([[10.0, 20.0], [12.0, 22.0]])

    with pytest.raises(ValueError):
        model._SensitivityModel__check_conformity(parameters_matrix, target_data)

    parameters_matrix2 = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])

    with pytest.raises(ValueError):
        model._SensitivityModel__check_conformity(parameters_matrix2, target_data)

    target_data2 = np.array([10.0, 12.0])

    with pytest.raises(ValueError):
        model._SensitivityModel__check_conformity(parameters_matrix, target_data2)

    target_variables_names = ["target1"]
    model = SensitivityModel(parameters_names, target_variables_names)

    target_data = np.array([[10.0, 20.0], [12.0, 22.0], [14.0, 24.0]])

    with pytest.raises(ValueError):
        model._SensitivityModel__check_conformity(parameters_matrix, target_data)


@patch("matplotlib.pyplot.show")
def test_prints_and_plots(mock_show):  # pylint: disable=unused-argument
    parameters_names = ["param1", "param2"]
    target_variables_names = ["target1"]
    model = SensitivityModel(parameters_names, target_variables_names)

    parameters_matrix = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    target_data = np.array([10.0, 12.0, 14.0])

    # tests if an error is raised if summary is called before print
    with pytest.raises(ValueError):
        model.info()

    model.fit(parameters_matrix, target_data)
    assert model.all_info() is None

    nominal_target = np.array([12.0])
    model.set_target_variables_nominal(nominal_target)
    assert model.all_info() is None
