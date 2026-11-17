"""Invalid stochastic inputs raise RocketPy exceptions that are still
``AssertionError`` for callers written before they existed."""

from types import SimpleNamespace

import pytest

import rocketpy
from rocketpy.exceptions import (
    RocketPyError,
    StochasticTypeError,
    StochasticValueError,
)
from rocketpy.stochastic import (
    StochasticEnvironment,
    StochasticFlight,
    StochasticFreeFormFins,
    StochasticParachute,
    StochasticRocket,
)
from rocketpy.stochastic.stochastic_model import StochasticModel

ENSEMBLE_ENVIRONMENT = SimpleNamespace(
    atmospheric_model_type="Ensemble", num_ensemble_members=3
)


def _model_with_mass(**inputs):
    return StochasticModel(SimpleNamespace(mass=1.0), **inputs)


def _free_form_fins(fins, shape_points):
    return StochasticFreeFormFins(free_form_fins=fins, shape_points=shape_points)


@pytest.mark.parametrize("error", [StochasticTypeError, StochasticValueError])
def test_stochastic_errors_keep_their_old_and_new_bases(error):
    assert issubclass(error, (RocketPyError, AssertionError))
    assert getattr(rocketpy, error.__name__) is error


def test_stochastic_errors_are_the_matching_builtin():
    assert issubclass(StochasticTypeError, TypeError)
    assert issubclass(StochasticValueError, ValueError)


@pytest.mark.parametrize(
    "build, error",
    [
        pytest.param(
            lambda: _model_with_mass(mass="heavy"), StochasticTypeError, id="model_str"
        ),
        pytest.param(
            lambda: _model_with_mass(mass=(1.0, 0.1, "normal", "extra")),
            StochasticValueError,
            id="model_tuple_length",
        ),
        pytest.param(
            lambda: _model_with_mass(mass=("a", 0.1)),
            StochasticTypeError,
            id="model_tuple_first_item",
        ),
        pytest.param(
            lambda: _model_with_mass(mass=(1.0, 0.1, 5)),
            StochasticTypeError,
            id="model_tuple_distribution_name",
        ),
        pytest.param(
            lambda: _model_with_mass(mass=(1.0, [0.1])),
            StochasticTypeError,
            id="model_tuple_second_item",
        ),
        pytest.param(
            lambda: _model_with_mass()._validate_custom_sampler("mass", object()),
            StochasticTypeError,
            id="custom_sampler",
        ),
        pytest.param(
            lambda: _model_with_mass()._validate_airfoil([("naca", "x", "y")]),
            StochasticValueError,
            id="airfoil_tuple_length",
        ),
        pytest.param(
            lambda: _model_with_mass()._validate_airfoil("naca"),
            StochasticTypeError,
            id="airfoil_not_a_list",
        ),
        pytest.param(
            lambda: _model_with_mass()._validate_positive_int_list("m", [-1]),
            StochasticValueError,
            id="negative_int_list",
        ),
        pytest.param(
            lambda: _model_with_mass()._validate_1d_array_like("m", "path"),
            StochasticTypeError,
            id="array_like_not_a_list",
        ),
        pytest.param(
            lambda: StochasticEnvironment._validate_ensemble(
                SimpleNamespace(),
                [0],
                SimpleNamespace(atmospheric_model_type="Forecast"),
            ),
            StochasticValueError,
            id="ensemble_on_environment_without_members",
        ),
        pytest.param(
            lambda: StochasticEnvironment._validate_ensemble(
                SimpleNamespace(), 0, ENSEMBLE_ENVIRONMENT
            ),
            StochasticTypeError,
            id="ensemble_member_not_a_list",
        ),
        pytest.param(
            lambda: StochasticEnvironment._validate_ensemble(
                SimpleNamespace(), [3], ENSEMBLE_ENVIRONMENT
            ),
            StochasticValueError,
            id="ensemble_member_out_of_range",
        ),
    ],
)
def test_stochastic_model_inputs_raise_the_matching_error(build, error):
    with pytest.raises(error):
        build()
    with pytest.raises(AssertionError):
        build()


def test_wrong_flight_inputs_raise_the_matching_error(flight_calisto_robust):
    with pytest.raises(StochasticTypeError):
        StochasticFlight(flight=flight_calisto_robust, terminate_on_apogee="yes")
    with pytest.raises(StochasticValueError):
        StochasticFlight(flight=flight_calisto_robust, initial_solution=(0.0, 1.0))
    with pytest.raises(AssertionError):
        StochasticFlight(flight=flight_calisto_robust, initial_solution=(0.0, 1.0))


def test_wrong_parachute_trigger_raises_the_matching_error(calisto_main_chute):
    with pytest.raises(StochasticTypeError):
        StochasticParachute(calisto_main_chute, trigger="banana")
    with pytest.raises(AssertionError):
        StochasticParachute(calisto_main_chute, trigger="banana")


@pytest.mark.parametrize(
    "shape_points, error",
    [
        ("outline", StochasticTypeError),
        ([(0.0, 0.0), (0.1, 0.0)], StochasticValueError),
        ((0.001,), StochasticValueError),
        (([(0.0, 0.0), (0.1, 0.0), (0.1, 0.1)], "x"), StochasticTypeError),
    ],
    ids=["not_an_outline", "too_few_points", "tuple_too_short", "std_is_a_string"],
)
def test_wrong_free_form_fins_shape_raises_the_matching_error(
    calisto_free_form_fins, shape_points, error
):
    with pytest.raises(error):
        _free_form_fins(calisto_free_form_fins, shape_points)
    with pytest.raises(AssertionError):
        _free_form_fins(calisto_free_form_fins, shape_points)


def test_wrong_rocket_components_raise_the_matching_error(calisto_robust, calisto_tail):
    stochastic = StochasticRocket(rocket=calisto_robust)

    with pytest.raises(StochasticTypeError):
        stochastic.add_free_form_fins(calisto_tail)
    with pytest.raises(StochasticTypeError):
        stochastic.set_rail_buttons("buttons")
    with pytest.raises(AssertionError):
        stochastic.set_rail_buttons("buttons")
