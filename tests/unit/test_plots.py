import os
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import pytest

from rocketpy.plots.compare import Compare
from rocketpy.plots.plot_helpers import show_or_save_fig, show_or_save_plot


@patch("matplotlib.pyplot.show")
def test_compare(mock_show, flight_calisto):  # pylint: disable=unused-argument
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

    assert isinstance(fig, plt.Figure)


@patch("matplotlib.pyplot.show")
@pytest.mark.parametrize("filename", [None, "test.png"])
def test_show_or_save_plot(mock_show, filename):
    """This test is to check if the show_or_save_plot function is
    working properly.

    Parameters
    ----------
    mock_show :
        Mocks the matplotlib.pyplot.show() function to avoid showing
        the plots.
    filename : str
        Name of the file to save the plot. If None, the plot will be
        shown instead.
    """
    plt.subplots()
    show_or_save_plot(filename)

    if filename is None:
        mock_show.assert_called_once()
    else:
        assert os.path.exists(filename)
        os.remove(filename)


@pytest.mark.parametrize("filename", [None, "test.png"])
def test_show_or_save_fig(filename):
    """This test is to check if the show_or_save_fig function is
    working properly.

    Parameters
    ----------
    filename : str
        Name of the file to save the plot. If None, the plot will be
        shown instead.
    """
    fig, _ = plt.subplots()

    fig.show = MagicMock()
    show_or_save_fig(fig, filename)

    if filename is None:
        fig.show.assert_called_once()
    else:
        assert os.path.exists(filename)
        os.remove(filename)
