import matplotlib as plt
import numpy as np

plt.rcParams.update({"figure.max_open_warning": 0})


def test_stochastic_environment_create_object_with_wind_x(stochastic_environment):
    """Tests the stochastic environment object by checking if the wind velocity
    can be generated properly. The goal is to check if the create_object()
    method is being called without any problems.

    Parameters
    ----------
    stochastic_environment : StochasticEnvironment
        The stochastic environment object, this is a pytest fixture.
    """
    wind_x_at_1000m = []
    for _ in range(10):
        random_env = stochastic_environment.create_object()
        wind_x_at_1000m.append(random_env.wind_velocity_x(1000))

    assert np.isclose(np.mean(wind_x_at_1000m), 0, atol=0.1)
    assert np.isclose(np.std(wind_x_at_1000m), 0, atol=0.1)
    # TODO: add a new test for the special case of ensemble member


def test_stochastic_solid_motor_create_object_with_impulse(stochastic_solid_motor):
    """Tests the stochastic solid motor object by checking if the total impulse
    can be generated properly. The goal is to check if the create_object()
    method is being called without any problems.

    Parameters
    ----------
    stochastic_solid_motor : StochasticSolidMotor
        The stochastic solid motor object, this is a pytest fixture.
    """
    total_impulse = [
        stochastic_solid_motor.create_object().total_impulse for _ in range(200)
    ]

    assert np.isclose(np.mean(total_impulse), 6500, rtol=0.3)
    assert np.isclose(np.std(total_impulse), 1000, rtol=0.4)


def test_stochastic_calisto_create_object_with_static_margin(stochastic_calisto):
    """Tests the stochastic calisto object by checking if the static margin
    can be generated properly. The goal is to check if the create_object()
    method is being called without any problems.

    Parameters
    ----------
    stochastic_calisto : StochasticCalisto
        The stochastic calisto object, this is a pytest fixture.
    """

    all_margins = []
    for _ in range(10):
        random_rocket = stochastic_calisto.create_object()
        all_margins.append(random_rocket.static_margin(0))

    assert np.isclose(np.mean(all_margins), 2.2625350013000434, rtol=0.15)
    assert np.isclose(np.std(all_margins), 0.1, atol=0.2)
