# pylint: disable=unused-argument
import os
from unittest.mock import patch

import matplotlib.pyplot as plt

from rocketpy import Flight
from rocketpy.plots.compare import CompareFlights


@patch("matplotlib.pyplot.show")
def test_compare(mock_show, flight_calisto):
    """Here we want to test the 'x_attributes' argument, which is the only one
    that is not tested in the other tests.

    Parameters
    ----------
    mock_show :
        Mocks the matplotlib.pyplot.show() function to avoid showing the plots.
    flight_calisto : rocketpy.Flight
        Flight object to be used in the tests. See conftest.py for more details.
    """
    flight = flight_calisto

    objects = [flight, flight, flight]

    comparison = CompareFlights(objects)

    fig, _ = comparison.create_comparison_figure(
        y_attributes=["z"],
        n_rows=1,
        n_cols=1,
        figsize=(10, 10),
        legend=False,
        title="Test",
        x_labels=["Time (s)"],
        y_labels=["Altitude (m)"],
        x_lim=(0, 3),
        y_lim=(0, 1000),
        x_attributes=["time"],
    )

    assert isinstance(fig, plt.Figure)


@patch("matplotlib.pyplot.show")
@patch("matplotlib.figure.Figure.show")
def test_compare_flights(mock_show, mock_figure_show, calisto, example_plain_env):
    """Tests the CompareFlights class. It simply ensures that all the methods
    are being called without errors. It does not test the actual plots, which
    would be very difficult to do.

    Parameters
    ----------
    mock_show :
        Mocks the matplotlib.pyplot.show() function to avoid showing the plots.
    mock_figure_show :
        Mocks the matplotlib.figure.Figure.show() function to avoid showing the plots.
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
    inclinations = [60, 90]
    headings = [0, 180]
    flights = []
    # Create (2 * 2) = 4 different flights to be compared
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
