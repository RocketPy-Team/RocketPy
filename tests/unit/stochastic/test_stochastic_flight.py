from rocketpy.simulation.flight import Flight
from rocketpy.stochastic import StochasticFlight


def test_stochastic_flight_create_object(stochastic_flight):
    obj = stochastic_flight.create_object()
    assert isinstance(obj, Flight)


def test_stochastic_flight_inherited_attributes(calisto_robust, example_spaceport_env):
    flight = Flight(
        rocket=calisto_robust,
        environment=example_spaceport_env,
        rail_length=5.2,
        max_time_step=10,
        min_time_step=0.01,
        rtol=1e-4,
        atol=1e-6,
        name="FlightName",
        equations_of_motion="solid_propulsion",
        ode_solver="BDF",
        simulation_mode="3 DOF",
    )
    stochastic_flight = StochasticFlight(flight=flight)

    obj = stochastic_flight.create_object()
    assert flight.max_time_step == obj.max_time_step
    assert flight.min_time_step == obj.min_time_step
    assert flight.rtol == obj.rtol
    assert flight.atol == obj.atol
    assert flight.name == obj.name
    assert flight.equations_of_motion == obj.equations_of_motion
    assert flight.ode_solver == obj.ode_solver
    assert flight.simulation_mode == obj.simulation_mode


def test_stochastic_flight_optional_attributes(flight_calisto_robust):
    stochastic_flight = StochasticFlight(
        flight=flight_calisto_robust,
        terminate_on_apogee=True,
        time_overshoot=True,
        max_time=987.6,
    )
    obj = stochastic_flight.create_object()
    assert obj.terminate_on_apogee is True
    assert obj.time_overshoot is True
    assert obj.max_time == 987.6


def test_dict_generator_skips_initial_solution_tuple(flight_calisto_robust):
    """Regression for #1109: tuple initial_solution must not be sampled."""
    initial_solution = tuple(float(i) for i in range(14))
    stochastic_flight = StochasticFlight(
        flight=flight_calisto_robust,
        initial_solution=initial_solution,
        rail_length=(5.2, 0.1),
    )
    generated = next(stochastic_flight.dict_generator())
    assert "initial_solution" not in generated
    assert stochastic_flight.initial_solution == initial_solution


def test_dict_generator_skips_initial_solution_list(flight_calisto_robust):
    """List-form initial_solution must not be randomly subset-sampled."""
    initial_solution = [float(i) for i in range(14)]
    stochastic_flight = StochasticFlight(
        flight=flight_calisto_robust,
        initial_solution=initial_solution,
        inclination=[85, 86, 87],
    )
    generated = next(stochastic_flight.dict_generator())
    assert "initial_solution" not in generated
    assert stochastic_flight.initial_solution == initial_solution


def test_create_object_matches_last_rnd_dict(flight_calisto_robust):
    """Regression for #1090: create_object must use one dict_generator draw.

    Spreads are set on all three flight inputs so a second draw would diverge
    from ``last_rnd_dict``.
    """
    stochastic_flight = StochasticFlight(
        flight=flight_calisto_robust,
        rail_length=(5.2, 0.5),
        inclination=(84.7, 1),
        heading=(53, 2),
    )
    stochastic_flight._set_stochastic(4242)

    flight = stochastic_flight.create_object()
    sampled = stochastic_flight.last_rnd_dict

    assert flight.rail_length == sampled["rail_length"]
    assert flight.inclination == sampled["inclination"]
    assert flight.heading == sampled["heading"]


def test_monte_carlo_single_simulation_matches_flight_last_rnd_dict(
    monte_carlo_calisto,
):
    """Regression for #1090: MonteCarlo must fly the same sample it logs."""
    monte_carlo_calisto.flight._set_stochastic(4242)

    flight = monte_carlo_calisto._MonteCarlo__run_single_simulation()
    sampled = monte_carlo_calisto.flight.last_rnd_dict

    assert flight.rail_length == sampled["rail_length"]
    assert flight.inclination == sampled["inclination"]
    assert flight.heading == sampled["heading"]
