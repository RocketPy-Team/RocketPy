from rocketpy import AirBrakes, AirBrakesController
from rocketpy.rocket.rocket import Rocket


def test_str(stochastic_calisto):
    assert isinstance(str(stochastic_calisto), str)


def test_air_brakes_samples_are_isolated(stochastic_calisto):
    """Each sampled rocket must get its own air brakes (wired into the
    surface loop) and its own controller with a deep-copied context, so
    Monte Carlo samples do not share mutable state."""
    air_brakes = AirBrakes(
        drag_coefficient_curve="data/rockets/calisto/air_brakes_cd.csv",
        reference_area=0.01,
    )
    controller = AirBrakesController(
        controller_function=lambda **kwargs: None,
        air_brakes=air_brakes,
        sampling_rate=10,
        context={"observed_variables": []},
    )
    stochastic_calisto.add_air_brakes(air_brakes, controller)

    first = stochastic_calisto.create_object()
    second = stochastic_calisto.create_object()

    # air brakes wired into the surface loop of each sample
    for rocket in (first, second):
        assert len(rocket.air_brakes) == 1
        assert rocket.air_brakes[0] in [
            surface for surface, _ in rocket.aerodynamic_surfaces
        ]
        assert isinstance(rocket._controllers[0], AirBrakesController)
        assert rocket._controllers[0].air_brakes is rocket.air_brakes[0]

    # contexts are deep copies: mutating one sample touches nothing else
    first._controllers[0].context["observed_variables"].append("junk")
    assert second._controllers[0].context == {"observed_variables": []}
    assert controller.context == {"observed_variables": []}


def test_create_object(stochastic_calisto):
    """Test create object method of StochasticRocket class.

    This test checks if the create_object method of the StochasticCalisto
    class creates a StochasticCalisto object from the randomly generated
    input arguments.

    Parameters
    ----------
    stochastic_calisto : StochasticCalisto
        StochasticCalisto object to be tested.

    Returns
    -------
    None
    """
    obj = stochastic_calisto.create_object()
    assert isinstance(obj, Rocket)
