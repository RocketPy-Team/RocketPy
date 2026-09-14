"""Unit tests for the ellipse plots in ``rocketpy.plots.monte_carlo_plots``.

Background-map fetching is covered separately in
``test_monte_carlo_plots_background.py``. Everything here runs with
``background=None`` so no tile provider is contacted.
"""

from unittest.mock import patch

import numpy as np
import pytest

from rocketpy.plots.monte_carlo_plots import _MonteCarloPlots
from rocketpy.simulation import MonteCarlo

APOGEE = {"apogee_x": [100, 200, 300], "apogee_y": [100, 200, 300]}
IMPACT = {"x_impact": [1000, 2000, 3000], "y_impact": [1000, 2000, 3000]}


class MockMonteCarlo(MonteCarlo):
    """A MonteCarlo carrying only the results the ellipse plots read."""

    def __init__(self, results, filename="test"):
        # pylint: disable=super-init-not-called
        self.filename = filename
        self.results = results
        self.plots = _MonteCarloPlots(self)


@pytest.fixture(name="square_image")
def square_image_fixture(tmp_path):
    """A small on-disk image for the ``image`` background path."""
    imageio = pytest.importorskip("imageio")
    path = tmp_path / "launch_site.png"
    imageio.imwrite(path, np.zeros((8, 8, 3), dtype=np.uint8))
    return str(path)


@patch("matplotlib.pyplot.show")
def test_ellipses_without_apogee_data_plots_the_impact_points(mock_show, caplog):  # pylint: disable=unused-argument
    """A results file may hold impacts and no apogees.

    The apogee lookup is allowed to miss; the method warns and draws what is
    left rather than failing.
    """
    monte_carlo = MockMonteCarlo(dict(IMPACT))

    assert monte_carlo.plots.ellipses() is None
    assert "No apogee data found" in caplog.text


@patch("matplotlib.pyplot.show")
def test_ellipses_without_impact_data_plots_the_apogee_points(mock_show, caplog):  # pylint: disable=unused-argument
    """The mirror case: apogees recorded, impacts missing."""
    monte_carlo = MockMonteCarlo(dict(APOGEE))

    assert monte_carlo.plots.ellipses() is None
    assert "No impact data found" in caplog.text


def test_ellipses_refuses_results_with_neither_apogee_nor_impact():
    """With both lookups missing there is nothing to draw an ellipse around."""
    monte_carlo = MockMonteCarlo({"t_final": [10, 11, 12]})

    with pytest.raises(ValueError, match="No apogee or impact data found"):
        monte_carlo.plots.ellipses()


def test_ellipses_reports_an_image_path_that_does_not_exist(tmp_path):
    """The path is the user's input, so the failure names it rather than the read."""
    monte_carlo = MockMonteCarlo({**APOGEE, **IMPACT})

    with pytest.raises(FileNotFoundError, match="image file was not found"):
        monte_carlo.plots.ellipses(image=str(tmp_path / "absent.png"))


@patch("matplotlib.pyplot.show")
def test_ellipses_draws_over_an_image_and_marks_the_actual_landing_point(  # pylint: disable=unused-argument
    mock_show, square_image
):
    """``image`` and ``actual_landing_point`` are separate optional branches."""
    monte_carlo = MockMonteCarlo({**APOGEE, **IMPACT})

    assert (
        monte_carlo.plots.ellipses(
            image=square_image, actual_landing_point=(1500, 1500)
        )
        is None
    )


@patch("matplotlib.pyplot.show")
def test_ellipses_comparison_without_apogee_data(mock_show, caplog):  # pylint: disable=unused-argument
    """The comparison reads four series at once, so one miss drops all four."""
    monte_carlo = MockMonteCarlo(dict(IMPACT))
    other = MockMonteCarlo(dict(IMPACT), filename="other")

    assert monte_carlo.plots.ellipses_comparison(other) is None
    assert "No apogee data found" in caplog.text


@patch("matplotlib.pyplot.show")
def test_ellipses_comparison_without_impact_data(mock_show, caplog):  # pylint: disable=unused-argument
    """The mirror case for the impact series."""
    monte_carlo = MockMonteCarlo(dict(APOGEE))
    other = MockMonteCarlo(dict(APOGEE), filename="other")

    assert monte_carlo.plots.ellipses_comparison(other) is None
    assert "No impact data found" in caplog.text


@patch("matplotlib.pyplot.show")
def test_ellipses_comparison_draws_over_an_image(mock_show, square_image):  # pylint: disable=unused-argument
    """The comparison takes the same ``image`` branch as ``ellipses``."""
    monte_carlo = MockMonteCarlo({**APOGEE, **IMPACT})
    other = MockMonteCarlo({**APOGEE, **IMPACT}, filename="other")

    assert monte_carlo.plots.ellipses_comparison(other, image=square_image) is None
