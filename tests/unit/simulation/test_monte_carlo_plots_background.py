# pylint: disable=unused-argument,assignment-from-no-return
import os
import urllib.error  # pylint: disable=unused-import
from unittest.mock import MagicMock, patch

import pytest

import matplotlib.pyplot as plt
from PIL import UnidentifiedImageError  # pylint: disable=unused-import

from rocketpy.plots.monte_carlo_plots import _MonteCarloPlots
from rocketpy.simulation import MonteCarlo

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


@patch("matplotlib.pyplot.show")
def test_ellipses_background_none(mock_show):
    """Test default behavior when background=None (no background map displayed).

    Parameters
    ----------
    mock_show :
        Mocks the matplotlib.pyplot.show() function to avoid showing the plots.
    """
    mock_monte_carlo = MockMonteCarlo(environment=SimpleEnvironment())
    # Test that background=None does not raise an error
    result = mock_monte_carlo.plots.ellipses(background=None)
    assert result is None


@patch("matplotlib.pyplot.show")
def test_ellipses_background_satellite(mock_show):
    """Test using satellite map when background="satellite".

    Parameters
    ----------
    mock_show :
        Mocks the matplotlib.pyplot.show() function to avoid showing the plots.
    """
    mock_monte_carlo = MockMonteCarlo(environment=SimpleEnvironment())
    # Test that background="satellite" does not raise an error
    result = mock_monte_carlo.plots.ellipses(background="satellite")
    assert result is None


@patch("matplotlib.pyplot.show")
def test_ellipses_background_street(mock_show):
    """Test using street map when background="street".

    Parameters
    ----------
    mock_show :
        Mocks the matplotlib.pyplot.show() function to avoid showing the plots.
    """
    mock_monte_carlo = MockMonteCarlo(environment=SimpleEnvironment())
    # Test that background="street" does not raise an error
    result = mock_monte_carlo.plots.ellipses(background="street")
    assert result is None


@patch("matplotlib.pyplot.show")
def test_ellipses_background_terrain(mock_show):
    """Test using terrain map when background="terrain".

    Parameters
    ----------
    mock_show :
        Mocks the matplotlib.pyplot.show() function to avoid showing the plots.
    """
    mock_monte_carlo = MockMonteCarlo(environment=SimpleEnvironment())
    # Test that background="terrain" does not raise an error
    result = mock_monte_carlo.plots.ellipses(background="terrain")
    assert result is None


@patch("matplotlib.pyplot.show")
def test_ellipses_background_custom_provider(mock_show):
    """Test using custom contextily provider for background.

    Parameters
    ----------
    mock_show :
        Mocks the matplotlib.pyplot.show() function to avoid showing the plots.
    """
    mock_monte_carlo = MockMonteCarlo(environment=SimpleEnvironment())
    # Test that custom provider does not raise an error
    result = mock_monte_carlo.plots.ellipses(background="CartoDB.Positron")
    assert result is None


@patch("matplotlib.pyplot.show")
def test_ellipses_image_takes_precedence_over_background(mock_show, tmp_path):
    """Test that image takes precedence when both image and background are provided.

    Parameters
    ----------
    mock_show :
        Mocks the matplotlib.pyplot.show() function to avoid showing the plots.
    tmp_path :
        pytest fixture providing a temporary directory.
    """
    mock_monte_carlo = MockMonteCarlo(environment=SimpleEnvironment())
    dummy_image_path = tmp_path / "dummy_image.png"
    dummy_image_path.write_bytes(b"dummy")

    # Test that when both image and background are provided, image takes precedence
    # This should not attempt to download background map
    import numpy as np  # pylint: disable=import-outside-toplevel

    mock_image = np.zeros((100, 100, 3), dtype=np.uint8)  # RGB image

    with patch("imageio.imread") as mock_imread:
        mock_imread.return_value = mock_image
        result = mock_monte_carlo.plots.ellipses(
            image=str(dummy_image_path), background="satellite"
        )
        assert result is None
        mock_imread.assert_called_once()


@patch("matplotlib.pyplot.show")
def test_ellipses_background_no_environment(mock_show):
    """Test that ValueError is raised when MonteCarlo object has no environment attribute.

    This test creates a MonteCarlo object without an environment attribute.
    The function should raise ValueError when trying to fetch background map.
    """
    mock_monte_carlo = MockMonteCarlo(environment=None)

    with pytest.raises(ValueError) as exc_info:
        mock_monte_carlo.plots.ellipses(background="satellite")
    assert "environment" in str(exc_info.value).lower()
    assert "automatically fetching the background map" in str(exc_info.value)


@patch("matplotlib.pyplot.show")
def test_ellipses_background_no_latitude_longitude(mock_show):
    """Test that ValueError is raised when environment has no latitude or longitude attributes.

    This test creates a mock environment without latitude and longitude attributes.
    The function should raise ValueError when trying to fetch background map.
    """

    # Create a simple environment object without latitude and longitude
    class EmptyEnvironment:
        """Empty environment object without latitude and longitude attributes."""

        def __init__(self):
            pass

    mock_environment = EmptyEnvironment()
    mock_monte_carlo = MockMonteCarlo(environment=mock_environment)

    with pytest.raises(ValueError) as exc_info:
        mock_monte_carlo.plots.ellipses(background="satellite")
    assert "latitude" in str(exc_info.value).lower()
    assert "longitude" in str(exc_info.value).lower()
    assert "automatically fetching the background map" in str(exc_info.value)


@patch("matplotlib.pyplot.show")
def test_ellipses_background_contextily_not_installed(mock_show):
    """Test that ImportError is raised when contextily is not installed.

    Parameters
    ----------
    mock_show :
        Mocks the matplotlib.pyplot.show() function to avoid showing the plots.
    """
    mock_monte_carlo = MockMonteCarlo(environment=SimpleEnvironment())
    from rocketpy.tools import import_optional_dependency as original_import  # pylint: disable=import-outside-toplevel

    # Create a mock function that only raises exception when importing contextily
    def mock_import_optional_dependency(name):
        if name == "contextily":
            raise ImportError("No module named 'contextily'")
        return original_import(name)

    with patch(
        "rocketpy.plots.monte_carlo_plots.import_optional_dependency",
        side_effect=mock_import_optional_dependency,
    ):
        with pytest.raises(ImportError) as exc_info:
            mock_monte_carlo.plots.ellipses(background="satellite")
        assert "contextily" in str(exc_info.value).lower()


@patch("matplotlib.pyplot.show")
def test_ellipses_background_with_custom_xlim_ylim(mock_show):
    """Test using background with custom xlim and ylim.

    Parameters
    ----------
    mock_show :
        Mocks the matplotlib.pyplot.show() function to avoid showing the plots.
    """
    mock_monte_carlo = MockMonteCarlo(environment=SimpleEnvironment())
    # Test using custom xlim and ylim
    result = mock_monte_carlo.plots.ellipses(
        background="satellite",
        xlim=(-5000, 5000),
        ylim=(-5000, 5000),
    )
    assert result is None


@patch("matplotlib.pyplot.show")
def test_ellipses_background_save(mock_show):
    """Test using background with save=True.

    Parameters
    ----------
    mock_show :
        Mocks the matplotlib.pyplot.show() function to avoid showing the plots.
    """
    filename = "monte_carlo_test.png"
    try:
        mock_monte_carlo = MockMonteCarlo(
            environment=SimpleEnvironment(), filename="monte_carlo_test"
        )
        # Test save functionality
        result = mock_monte_carlo.plots.ellipses(background="satellite", save=True)
        assert result is None
        # Verify file was created
        assert os.path.exists(filename)
    finally:
        if os.path.exists(filename):
            os.remove(filename)


@patch("matplotlib.pyplot.show")
def test_ellipses_background_invalid_provider(mock_show):
    """Test that ValueError is raised when an invalid map provider is specified.

    Parameters
    ----------
    mock_show :
        Mocks the matplotlib.pyplot.show() function to avoid showing the plots.
    """
    mock_monte_carlo = MockMonteCarlo(environment=SimpleEnvironment())
    with pytest.raises(ValueError) as exc_info:
        mock_monte_carlo.plots.ellipses(background="Invalid.Provider.Name")
    assert "Invalid map provider" in str(exc_info.value)
    assert "Invalid.Provider.Name" in str(exc_info.value)
    assert (
        "satellite" in str(exc_info.value)
        or "street" in str(exc_info.value)
        or "terrain" in str(exc_info.value)
    )


@patch("matplotlib.pyplot.show")
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
def test_ellipses_background_bounds2img_failure(
    mock_show, exception_factory, expected_exception, expected_messages
):
    """Test that appropriate exceptions are raised when bounds2img fails.

    This is a parameterized test that covers all exception types handled in
    the _fetch_background_map method:
    - ValueError: invalid coordinates or zoom level
    - ConnectionError: network errors (URLError, HTTPError, TimeoutError)
    - RuntimeError: UnidentifiedImageError (invalid image data)
    - RuntimeError: other unexpected exceptions

    Parameters
    ----------
    mock_show :
        Mocks the matplotlib.pyplot.show() function to avoid showing the plots.
    exception_factory : callable
        A function that returns the exception to raise in mock_bounds2img.
    expected_exception : type
        The expected exception type to be raised.
    expected_messages : list[str]
        List of expected message substrings in the raised exception.
    """
    mock_monte_carlo = MockMonteCarlo(environment=SimpleEnvironment())
    from rocketpy.tools import import_optional_dependency as original_import  # pylint: disable=import-outside-toplevel

    contextily = pytest.importorskip("contextily")

    mock_contextily = MagicMock()
    mock_contextily.providers = contextily.providers

    def mock_bounds2img(*args, **kwargs):
        raise exception_factory()

    mock_contextily.bounds2img = mock_bounds2img

    def mock_import_optional_dependency(name):
        if name == "contextily":
            return mock_contextily
        return original_import(name)

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
