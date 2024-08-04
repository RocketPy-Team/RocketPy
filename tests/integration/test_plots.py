# pylint: disable=unused-argument
import os
from unittest.mock import patch

from rocketpy import Flight
from rocketpy.plots.compare import CompareFlights


@patch("matplotlib.pyplot.show")
def test_compare_flights(mock_show, calisto, example_plain_env):
    """Tests the CompareFlights class. It simply ensures that all the methods
    are being called without errors. It does not test the actual plots, which
    would be very difficult to do.

    Parameters
    ----------
    mock_show :
        Mocks the matplotlib.pyplot.show() function to avoid showing the plots.
    calisto : rocketpy.Rocket
        Rocket object to be used in the tests. See conftest.py for more details.
    example_plain_env : rocketpy.Environment
        Environment object to be used in the tests. See conftest.py for more details.

    Returns
    -------
    None
    """
    example_plain_env.set_atmospheric_model(
        type="custom_atmosphere",
        pressure=None,
        temperature=300,
        wind_u=[(0, 5), (1000, 10)],
        wind_v=[(0, -2), (500, 3), (1600, 2)],
    )

    calisto.set_rail_buttons(-0.5, 0.2)
    inclinations = [60, 70, 80, 90]
    headings = [0, 45, 90, 180]
    flights = []
    # Create (4 * 4) = 16 different flights to be compared
    for heading in headings:
        for inclination in inclinations:
            flight = Flight(
                environment=example_plain_env,
                rocket=calisto,
                rail_length=5,
                inclination=inclination,
                heading=heading,
                name=f"Incl {inclination} Head {heading}",
            )
            flights.append(flight)

    comparison = CompareFlights(flights)

    assert comparison.all() is None
    assert comparison.trajectories_2d(plane="xz", legend=False) is None
    assert comparison.trajectories_2d(plane="yz", legend=True) is None

    # Test save fig and then remove file
    assert comparison.positions(filename="test.png") is None
    os.remove("test.png")

    # Test xlim and ylim arguments
    assert comparison.positions(x_lim=[0, 100], y_lim=[0, 1000]) is None
    assert comparison.positions(x_lim=[0, "apogee"]) is None
