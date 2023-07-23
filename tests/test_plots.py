import os
from unittest.mock import patch

import matplotlib.pyplot as plt

from rocketpy import Flight
from rocketpy.plots.compare import Compare, CompareFlights


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

    comparison = Compare(object_list=objects)

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

    assert isinstance(fig, plt.Figure) == True


@patch("matplotlib.pyplot.show")
def test_compare_flights(mock_show, calisto, example_env):
    """Tests the CompareFlights class. It simply ensures that all the methods
    are being called without errors. It does not test the actual plots, which
    would be very difficult to do.

    Parameters
    ----------
    mock_show :
        Mocks the matplotlib.pyplot.show() function to avoid showing the plots.
    calisto : rocketpy.Rocket
        Rocket object to be used in the tests. See conftest.py for more details.
    example_env : rocketpy.Environment
        Environment object to be used in the tests. See conftest.py for more details.

    Returns
    -------
    None
    """
    example_env.set_atmospheric_model(
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
                environment=example_env,
                rocket=calisto,
                rail_length=5,
                inclination=inclination,
                heading=heading,
                name=f"Incl {inclination} Head {heading}",
            )
            flights.append(flight)

    comparison = CompareFlights(flights)

    assert comparison.all() == None
    assert comparison.trajectories_2d(plane="xz", legend=False) == None
    assert comparison.trajectories_2d(plane="yz", legend=True) == None

    # Test save fig and then remove file
    assert comparison.positions(filename="test.png") == None
    os.remove("test.png")

    # Test xlim and ylim arguments
    assert comparison.positions(x_lim=[0, 100], y_lim=[0, 1000]) == None
    assert comparison.positions(x_lim=[0, "apogee"]) == None
