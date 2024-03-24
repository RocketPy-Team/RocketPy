from rocketpy.simulation.flight import Flight


def test_stochastic_flight_create_object(stochastic_flight):
    obj = stochastic_flight.create_object()
    assert isinstance(obj, Flight)
