# pylint: disable=unused-argument,assignment-from-no-return
import os
import urllib.error
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import UnidentifiedImageError

from rocketpy.plots.monte_carlo_plots import _MonteCarloPlots
from rocketpy.simulation import MonteCarlo
from rocketpy.tools import import_optional_dependency

plt.rcParams.update({"figure.max_open_warning": 0})

pytest.importorskip(
    "contextily", reason="This test requires contextily to be installed"
)


class MockMonteCarlo(MonteCarlo):
    """Create a mock class to test the method without running a real simulation.

    This class creates a MonteCarlo object with simulated results data for testing
    background map options. Only includes the minimal attributes needed for plotting.
    """

    def __init__(self, environment=None, filename="test"):
        """Initialize MockMonteCarlo with simulated data.

        Parameters
        ----------
        environment : object, optional
            Environment object with latitude and longitude attributes.
            If None, no environment attribute will be set.
        filename : str, optional
            Filename for the MonteCarlo object. Defaults to "test".
        """
        # pylint: disable=super-init-not-called
        # Set attributes needed for plotting background maps
        self.filename = filename
        self.results = {
            "apogee_x": [100, 200, 300],
            "apogee_y": [100, 200, 300],
            "x_impact": [1000, 2000, 3000],
            "y_impact": [1000, 2000, 3000],
        }
        if environment is not None:
            self.environment = environment
        self.plots = _MonteCarloPlots(self)


class SimpleEnvironment:
    """Simple environment object with latitude and longitude attributes."""

    def __init__(self, latitude=32.990254, longitude=-106.974998):
        """Initialize SimpleEnvironment with latitude and longitude.

        Parameters
        ----------
        latitude : float, optional
            Latitude in degrees. Defaults to Spaceport America coordinates.
        longitude : float, optional
            Longitude in degrees. Defaults to Spaceport America coordinates.
        """
        self.latitude = latitude
        self.longitude = longitude


@pytest.mark.parametrize(
    "background_type",
    [None, "satellite", "street", "terrain", "CartoDB.Positron"],
)
@patch("matplotlib.pyplot.show")
def test_ellipses_background_types_display_successfully(mock_show, background_type):
    """Test that different background map types display without errors.

    This parameterized test verifies that the ellipses method works with:
    - None (no background map)
    - "satellite" (Esri.WorldImagery)
    - "street" (OpenStreetMap.Mapnik)
    - "terrain" (Esri.WorldTopoMap)
    - Custom provider (e.g., CartoDB.Positron)

    Parameters
    ----------
    mock_show : unittest.mock.MagicMock
        Mocks the matplotlib.pyplot.show() function to avoid displaying plots.
    background_type : str or None
        The background map type to test.
    """
    mock_monte_carlo = MockMonteCarlo(environment=SimpleEnvironment())

    result = mock_monte_carlo.plots.ellipses(background=background_type)

    assert result is None


@patch("matplotlib.pyplot.show")
def test_ellipses_image_takes_precedence_over_background(mock_show, tmp_path):
    """Test that image parameter takes precedence over background parameter.

    When both image and background are provided, the image should be used
    and the background map should not be downloaded.

    Parameters
    ----------
    mock_show : unittest.mock.MagicMock
        Mocks the matplotlib.pyplot.show() function to avoid displaying plots.
    tmp_path : pathlib.Path
        Pytest fixture providing a temporary directory.
    """

    mock_monte_carlo = MockMonteCarlo(environment=SimpleEnvironment())
    dummy_image_path = tmp_path / "dummy_image.png"
    dummy_image_path.write_bytes(b"dummy")
    mock_image = np.zeros((100, 100, 3), dtype=np.uint8)  # RGB image

    with patch("imageio.imread") as mock_imread:
        mock_imread.return_value = mock_image
        result = mock_monte_carlo.plots.ellipses(
            image=str(dummy_image_path), background="satellite"
        )
        assert result is None
        mock_imread.assert_called_once()


@patch("matplotlib.pyplot.show")
def test_ellipses_background_raises_error_when_no_environment(mock_show):
    """Test that ValueError is raised when environment attribute is missing.

    Parameters
    ----------
    mock_show : unittest.mock.MagicMock
        Mocks the matplotlib.pyplot.show() function to avoid displaying plots.
    """

    mock_monte_carlo = MockMonteCarlo(environment=None)

    with pytest.raises(ValueError) as exc_info:
        mock_monte_carlo.plots.ellipses(background="satellite")

    error_message = str(exc_info.value).lower()
    assert "environment" in error_message
    assert "automatically fetching the background map" in error_message


@patch("matplotlib.pyplot.show")
def test_ellipses_background_raises_error_when_missing_coordinates(mock_show):
    """Test that ValueError is raised when environment lacks latitude or longitude.

    Parameters
    ----------
    mock_show : unittest.mock.MagicMock
        Mocks the matplotlib.pyplot.show() function to avoid displaying plots.
    """

    class EmptyEnvironment:
        """Empty environment object without latitude and longitude attributes."""

    mock_environment = EmptyEnvironment()
    mock_monte_carlo = MockMonteCarlo(environment=mock_environment)

    with pytest.raises(ValueError) as exc_info:
        mock_monte_carlo.plots.ellipses(background="satellite")

    error_message = str(exc_info.value).lower()
    assert "latitude" in error_message
    assert "longitude" in error_message
    assert "automatically fetching the background map" in error_message


@patch("matplotlib.pyplot.show")
def test_ellipses_background_raises_error_when_contextily_not_installed(mock_show):
    """Test that ImportError is raised when contextily is not installed.

    Parameters
    ----------
    mock_show : unittest.mock.MagicMock
        Mocks the matplotlib.pyplot.show() function to avoid displaying plots.
    """

    mock_monte_carlo = MockMonteCarlo(environment=SimpleEnvironment())

    def mock_import_optional_dependency(name):
        """Mock function that raises ImportError for contextily."""
        if name == "contextily":
            raise ImportError("No module named 'contextily'")
        return import_optional_dependency(name)

    with patch(
        "rocketpy.plots.monte_carlo_plots.import_optional_dependency",
        side_effect=mock_import_optional_dependency,
    ):
        with pytest.raises(ImportError) as exc_info:
            mock_monte_carlo.plots.ellipses(background="satellite")
        assert "contextily" in str(exc_info.value).lower()


@patch("matplotlib.pyplot.show")
def test_ellipses_background_works_with_custom_limits(mock_show):
    """Test that background maps work with custom axis limits.

    Parameters
    ----------
    mock_show : unittest.mock.MagicMock
        Mocks the matplotlib.pyplot.show() function to avoid displaying plots.
    """

    mock_monte_carlo = MockMonteCarlo(environment=SimpleEnvironment())

    result = mock_monte_carlo.plots.ellipses(
        background="satellite",
        xlim=(-5000, 5000),
        ylim=(-5000, 5000),
    )

    assert result is None


@patch("matplotlib.pyplot.show")
def test_ellipses_background_saves_file_successfully(mock_show):
    """Test that plots with background maps can be saved to file.

    Parameters
    ----------
    mock_show : unittest.mock.MagicMock
        Mocks the matplotlib.pyplot.show() function to avoid displaying plots.
    """

    filename = "monte_carlo_test.png"
    mock_monte_carlo = MockMonteCarlo(
        environment=SimpleEnvironment(), filename="monte_carlo_test"
    )

    try:
        result = mock_monte_carlo.plots.ellipses(background="satellite", save=True)
        assert result is None
        assert os.path.exists(filename)
    finally:
        if os.path.exists(filename):
            os.remove(filename)


@patch("matplotlib.pyplot.show")
def test_ellipses_background_raises_error_for_invalid_provider(mock_show):
    """Test that ValueError is raised for invalid map provider names.

    Parameters
    ----------
    mock_show : unittest.mock.MagicMock
        Mocks the matplotlib.pyplot.show() function to avoid displaying plots.
    """

    mock_monte_carlo = MockMonteCarlo(environment=SimpleEnvironment())
    invalid_provider = "Invalid.Provider.Name"

    with pytest.raises(ValueError) as exc_info:
        mock_monte_carlo.plots.ellipses(background=invalid_provider)

    error_message = str(exc_info.value)
    assert "Invalid map provider" in error_message
    assert invalid_provider in error_message
    # Check that error message includes built-in options
    assert any(option in error_message for option in ["satellite", "street", "terrain"])


@pytest.mark.parametrize(
    "exception_factory,expected_exception,expected_messages",
    [
        # ValueError case: invalid coordinates or zoom level
        (
            lambda: ValueError("Invalid coordinates"),
            ValueError,
            [
                "Input coordinates or zoom level are invalid",
                "Provided bounds",
                "Tip: Ensure West < East and South < North",
            ],
        ),
        # ConnectionError case: network errors (URLError)
        (
            lambda: urllib.error.URLError("Network error: Unable to fetch tiles"),
            ConnectionError,
            [
                "Network error while fetching tiles from provider",
                "Check your internet connection",
                "The tile server might be down or blocking requests",
            ],
        ),
        # ConnectionError case: network errors (HTTPError)
        (
            lambda: urllib.error.HTTPError(
                "http://example.com", 500, "Internal Server Error", None, None
            ),
            ConnectionError,
            [
                "Network error while fetching tiles from provider",
                "Check your internet connection",
                "The tile server might be down or blocking requests",
            ],
        ),
        # ConnectionError case: network errors (TimeoutError)
        (
            lambda: TimeoutError("Request timed out"),
            ConnectionError,
            [
                "Network error while fetching tiles from provider",
                "Check your internet connection",
                "The tile server might be down or blocking requests",
            ],
        ),
        # RuntimeError case: UnidentifiedImageError (invalid image data)
        (
            lambda: UnidentifiedImageError("Cannot identify image file"),
            RuntimeError,
            [
                "returned invalid image data",
                "API requires a key/token that is missing or invalid",
                "server likely returned an HTML error page instead of a PNG/JPG",
            ],
        ),
        # RuntimeError case: other unexpected exceptions
        (
            lambda: Exception("Unexpected error occurred"),
            RuntimeError,
            [
                "An unexpected error occurred while generating the map",
                "Bounds",
                "Provider",
                "Error Detail",
            ],
        ),
    ],
)
@patch("matplotlib.pyplot.show")
def test_ellipses_background_handles_bounds2img_failures(
    mock_show, exception_factory, expected_exception, expected_messages
):
    """Test that appropriate exceptions are raised when bounds2img fails.

    This parameterized test verifies error handling for all exception types
    that can occur during background map fetching:
    - ValueError: Invalid coordinates or zoom level
    - ConnectionError: Network errors (URLError, HTTPError, TimeoutError)
    - RuntimeError: Invalid image data (UnidentifiedImageError)
    - RuntimeError: Other unexpected exceptions

    Parameters
    ----------
    mock_show : unittest.mock.MagicMock
        Mocks the matplotlib.pyplot.show() function to avoid displaying plots.
    exception_factory : callable
        A function that returns the exception to raise in mock_bounds2img.
    expected_exception : type
        The expected exception type to be raised.
    expected_messages : list of str
        List of expected message substrings in the raised exception.
    """

    mock_monte_carlo = MockMonteCarlo(environment=SimpleEnvironment())
    contextily = pytest.importorskip("contextily")

    mock_contextily = MagicMock()
    mock_contextily.providers = contextily.providers

    def mock_bounds2img(*args, **kwargs):
        """Mock bounds2img that raises the specified exception."""
        raise exception_factory()

    mock_contextily.bounds2img = mock_bounds2img

    def mock_import_optional_dependency(name):
        """Mock import function that returns mock contextily."""
        if name == "contextily":
            return mock_contextily
        return import_optional_dependency(name)

    with patch(
        "rocketpy.plots.monte_carlo_plots.import_optional_dependency",
        side_effect=mock_import_optional_dependency,
    ):
        with pytest.raises(expected_exception) as exc_info:
            mock_monte_carlo.plots.ellipses(background="satellite")

        error_message = str(exc_info.value)
        for expected_msg in expected_messages:
            assert expected_msg in error_message, (
                f"Expected message '{expected_msg}' not found in error: {error_message}"
            )

        assert "Esri.WorldImagery" in error_message
