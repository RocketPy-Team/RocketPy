"""This module contains fixtures for the stochastic module. The fixtures are
used to test the stochastic objects that will be used in the Monte Carlo
simulations. It is a team effort to keep it as documented as possible."""

import pytest

from rocketpy.stochastic import (
    StochasticEnvironment,
    StochasticFlight,
    StochasticNoseCone,
    StochasticParachute,
    StochasticRailButtons,
    StochasticRocket,
    StochasticTail,
    StochasticTrapezoidalFins,
)


@pytest.fixture
def stochastic_environment(example_spaceport_env):
    """This fixture is used to create a stochastic environment object for the
    Calisto flight.

    Parameters
    ----------
    example_spaceport_env : Environment
        This is another fixture.

    Returns
    -------
    StochasticEnvironment
        The stochastic environment object
    """
    return StochasticEnvironment(
        environment=example_spaceport_env,
        elevation=(1400, 10, "normal"),
        gravity=None,
        latitude=None,
        longitude=None,
        ensemble_member=None,
        wind_velocity_x_factor=(1.0, 0.033, "normal"),
        wind_velocity_y_factor=(1.0, 0.033, "normal"),
    )


@pytest.fixture
def stochastic_environment_custom_sampler(example_spaceport_env, elevation_sampler):
    """This fixture is used to create a stochastic environment object for the
    Calisto flight using a custom sampler for the elevation.

    Parameters
    ----------
    example_spaceport_env : Environment
        This is another fixture.

    elevation_sampler: CustomSampler
        This is another fixture.

    Returns
    -------
    StochasticEnvironment
        The stochastic environment object
    """
    return StochasticEnvironment(
        environment=example_spaceport_env,
        elevation=elevation_sampler,
        gravity=None,
        latitude=None,
        longitude=None,
        ensemble_member=None,
        wind_velocity_x_factor=(1.0, 0.033, "normal"),
        wind_velocity_y_factor=(1.0, 0.033, "normal"),
    )


@pytest.fixture
def stochastic_nose_cone(calisto_nose_cone):
    """This fixture is used to create a StochasticNoseCone object for the
    Calisto rocket.

    Parameters
    ----------
    calisto_nose_cone : NoseCone
        This is another fixture.

    Returns
    -------
    StochasticNoseCone
        The stochastic nose cone object
    """
    return StochasticNoseCone(
        nosecone=calisto_nose_cone,
        length=0.001,
    )


@pytest.fixture
def stochastic_trapezoidal_fins(calisto_trapezoidal_fins):
    """This fixture is used to create a StochasticTrapezoidalFins object for the
    Calisto rocket.

    Parameters
    ----------
    calisto_trapezoidal_fins : TrapezoidalFins
        This is another fixture.

    Returns
    -------
    StochasticTrapezoidalFins
        The stochastic trapezoidal fins object
    """
    return StochasticTrapezoidalFins(
        trapezoidal_fins=calisto_trapezoidal_fins,
        root_chord=0.0005,
        tip_chord=0.0005,
        span=0.0005,
    )


@pytest.fixture
def stochastic_tail(calisto_tail):
    """This fixture is used to create a StochasticTail object for the
    Calisto rocket.

    Parameters
    ----------
    calisto_tail : Tail
        This is another fixture.

    Returns
    -------
    StochasticTail
        The stochastic tail object
    """
    return StochasticTail(
        tail=calisto_tail,
        top_radius=0.001,
        bottom_radius=0.001,
        length=0.001,
    )


@pytest.fixture
def stochastic_rail_buttons(calisto_rail_buttons):
    """This fixture is used to create a StochasticRailButtons object for the
    Calisto rocket.

    Parameters
    ----------
    calisto_rail_buttons : RailButtons
        This is another fixture.

    Returns
    -------
    StochasticRailButtons
        The stochastic rail buttons object
    """
    return StochasticRailButtons(
        rail_buttons=calisto_rail_buttons, buttons_distance=0.001
    )


@pytest.fixture
def stochastic_main_parachute(calisto_main_chute):
    """This fixture is used to create a StochasticParachute object for the
    Calisto rocket.

    Parameters
    ----------
    calisto_main_chute : Parachute
        This is another fixture.

    Returns
    -------
    StochasticParachute
        The stochastic parachute object
    """
    return StochasticParachute(
        parachute=calisto_main_chute,
        cd_s=0.1,
        lag=0.1,
    )


@pytest.fixture
def stochastic_drogue_parachute(calisto_drogue_chute):
    """This fixture is used to create a StochasticParachute object for the
    Calisto rocket. This time, the drogue parachute is created.

    Parameters
    ----------
    calisto_drogue_chute : Parachute
        This is another fixture.

    Returns
    -------
    StochasticParachute
        The stochastic parachute object
    """
    return StochasticParachute(
        parachute=calisto_drogue_chute,
        cd_s=0.07,
        lag=0.2,
    )


@pytest.fixture
def stochastic_calisto(
    calisto_robust,
    stochastic_nose_cone,
    stochastic_trapezoidal_fins,
    stochastic_tail,
    stochastic_solid_motor,
    stochastic_rail_buttons,
    stochastic_main_parachute,
    stochastic_drogue_parachute,
):
    """This fixture creates a StochasticRocket object for the Calisto rocket.
    The fixture will already have the stochastic nose cone, trapezoidal fins,
    tail, solid motor, rail buttons, main parachute, and drogue parachute.

    Returns
    -------
    StochasticRocket
        The stochastic rocket object
    """
    rocket = StochasticRocket(
        rocket=calisto_robust,
        radius=0.0127 / 2000,
        mass=(15.426, 0.5, "normal"),
        inertia_11=(6.321, 0),
        inertia_22=0.01,
        inertia_33=0.01,
        center_of_mass_without_motor=0,
    )
    rocket.add_motor(stochastic_solid_motor, position=0.001)
    rocket.add_nose(stochastic_nose_cone, position=(1.134, 0.001))
    rocket.add_trapezoidal_fins(stochastic_trapezoidal_fins, position=(0.001, "normal"))
    rocket.add_tail(stochastic_tail)
    rocket.set_rail_buttons(
        stochastic_rail_buttons, lower_button_position=(-0.618, 0.001, "normal")
    )
    rocket.add_parachute(stochastic_main_parachute)
    rocket.add_parachute(stochastic_drogue_parachute)
    return rocket


@pytest.fixture
def stochastic_flight(flight_calisto_robust):
    """This fixture creates a StochasticFlight object for the Calisto flight.

    Parameters
    ----------
    flight_calisto_robust : Flight
        This is another fixture.

    Returns
    -------
    StochasticFlight
        The stochastic flight object
    """
    return StochasticFlight(
        flight=flight_calisto_robust,
        inclination=(84.7, 1),
        heading=(53, 2),
    )
