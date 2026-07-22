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
