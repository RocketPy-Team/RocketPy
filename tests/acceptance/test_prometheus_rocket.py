from rocketpy import Flight
from rocketpy.simulation.flight_data_importer import FlightDataImporter


def test_prometheus_rocket_data_asserts_acceptance(
    environment_spaceport_america_2023, prometheus_rocket
):
    """Tests the Prometheus rocket flight data against acceptance criteria.

    This function simulates a rocket flight using the given environment and
    rocket parameters, then compares the simulated apogee with real flight data
    to ensure the relative error is within acceptable thresholds.

    Parameters
    ----------
    environment_spaceport_america_2023 : Environment
        An environment configuration for Spaceport America in 2023.
    prometheus_rocket : Rocket
        The Prometheus rocket configuration.

    Raises
    ------
    AssertionError
        If the relative error between the simulated apogee and the real apogee
        exceeds the threshold.
    """
    # Define relative error threshold (defined manually based on data)
    apogee_threshold = 7.5 / 100

    # Simulate the flight
    test_flight = Flight(
        rocket=prometheus_rocket,
        environment=environment_spaceport_america_2023,
        inclination=80,
        heading=75,
        rail_length=5.18,
    )

    # Read the flight data
    columns_map = {
        "time": "time",
        "altitude": "z",
        "height": "altitude",
        "acceleration": "acceleration",
        "pressure": "pressure",
        "accel_x": "ax",
        "accel_y": "ay",
        "accel_z": "az",
        "latitude": "latitude",
        "longitude": "longitude",
    }

    altimeter_data = FlightDataImporter(
        name="Telemetry Mega",
        paths="data/rockets/prometheus/2022-06-24-serial-5115-flight-0001-TeleMetrum.csv",
        columns_map=columns_map,
        units=None,
        interpolation="linear",
        extrapolation="zero",
        delimiter=",",
        encoding="utf-8",
    )

    # Calculate errors and assert values
    real_apogee = altimeter_data.altitude.max
    rocketpy_apogee = test_flight.apogee - test_flight.env.elevation
    a_error = abs(real_apogee - rocketpy_apogee)
    r_error = a_error / real_apogee

    assert r_error < apogee_threshold, f"Apogee relative error is {r_error * 100:.2f}%"
