# pylint: disable=unused-argument
import os
import warnings
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import pytest

plt.rcParams.update({"figure.max_open_warning": 0})


def _post_test_file_cleanup():
    """Clean monte carlo files after test session if they exist."""
    files_to_cleanup = [
        "monte_carlo_test.png",
    ]
    for filepath in files_to_cleanup:
        if os.path.exists(filepath):
            os.remove(filepath)


@patch("matplotlib.pyplot.show")
def test_ellipses_background_none(mock_show, monte_carlo_calisto_pre_loaded):
    """Test default behavior when background=None (no background map displayed).

    Parameters
    ----------
    mock_show :
        Mocks the matplotlib.pyplot.show() function to avoid showing the plots.
    monte_carlo_calisto_pre_loaded : MonteCarlo
        The MonteCarlo object, this is a pytest fixture.
    """
    try:
        # Test that background=None does not raise an error
        result = monte_carlo_calisto_pre_loaded.plots.ellipses(background=None)
        assert result is None
    finally:
        _post_test_file_cleanup()


@patch("matplotlib.pyplot.show")
def test_ellipses_background_satellite(mock_show, monte_carlo_calisto_pre_loaded):
    """Test using satellite map when background="satellite".

    Parameters
    ----------
    mock_show :
        Mocks the matplotlib.pyplot.show() function to avoid showing the plots.
    monte_carlo_calisto_pre_loaded : MonteCarlo
        The MonteCarlo object, this is a pytest fixture.
    """
    try:
        # Test that background="satellite" does not raise an error
        result = monte_carlo_calisto_pre_loaded.plots.ellipses(background="satellite")
        assert result is None
    finally:
        _post_test_file_cleanup()


@patch("matplotlib.pyplot.show")
def test_ellipses_background_street(mock_show, monte_carlo_calisto_pre_loaded):
    """Test using street map when background="street".

    Parameters
    ----------
    mock_show :
        Mocks the matplotlib.pyplot.show() function to avoid showing the plots.
    monte_carlo_calisto_pre_loaded : MonteCarlo
        The MonteCarlo object, this is a pytest fixture.
    """
    try:
        # Test that background="street" does not raise an error
        result = monte_carlo_calisto_pre_loaded.plots.ellipses(background="street")
        assert result is None
    finally:
        _post_test_file_cleanup()


@patch("matplotlib.pyplot.show")
def test_ellipses_background_terrain(mock_show, monte_carlo_calisto_pre_loaded):
    """Test using terrain map when background="terrain".

    Parameters
    ----------
    mock_show :
        Mocks the matplotlib.pyplot.show() function to avoid showing the plots.
    monte_carlo_calisto_pre_loaded : MonteCarlo
        The MonteCarlo object, this is a pytest fixture.
    """
    try:
        # Test that background="terrain" does not raise an error
        result = monte_carlo_calisto_pre_loaded.plots.ellipses(background="terrain")
        assert result is None
    finally:
        _post_test_file_cleanup()


@patch("matplotlib.pyplot.show")
def test_ellipses_background_custom_provider(mock_show, monte_carlo_calisto_pre_loaded):
    """Test using custom contextily provider for background.

    Parameters
    ----------
    mock_show :
        Mocks the matplotlib.pyplot.show() function to avoid showing the plots.
    monte_carlo_calisto_pre_loaded : MonteCarlo
        The MonteCarlo object, this is a pytest fixture.
    """
    try:
        # Test that custom provider does not raise an error
        result = monte_carlo_calisto_pre_loaded.plots.ellipses(
            background="CartoDB.Positron"
        )
        assert result is None
    finally:
        _post_test_file_cleanup()


@patch("matplotlib.pyplot.show")
def test_ellipses_image_takes_precedence_over_background(
    mock_show, monte_carlo_calisto_pre_loaded, tmp_path
):
    """Test that image takes precedence when both image and background are provided.

    Parameters
    ----------
    mock_show :
        Mocks the matplotlib.pyplot.show() function to avoid showing the plots.
    monte_carlo_calisto_pre_loaded : MonteCarlo
        The MonteCarlo object, this is a pytest fixture.
    tmp_path :
        pytest fixture providing a temporary directory.
    """
    try:
        dummy_image_path = tmp_path / "dummy_image.png"
        dummy_image_path.write_bytes(b"dummy")

        # Test that when both image and background are provided, image takes precedence
        # This should not attempt to download background map
        import numpy as np

        mock_image = np.zeros((100, 100, 3), dtype=np.uint8)  # RGB image

        with patch("imageio.imread") as mock_imread:
            mock_imread.return_value = mock_image
            result = monte_carlo_calisto_pre_loaded.plots.ellipses(
                image=str(dummy_image_path), background="satellite"
            )
            assert result is None
            mock_imread.assert_called_once()
    finally:
        _post_test_file_cleanup()


def test_ellipses_background_no_environment():
    """Test that ValueError is raised when MonteCarlo object has no environment attribute.

    This test creates a MonteCarlo object without an environment attribute.
    """
    from rocketpy.plots.monte_carlo_plots import _MonteCarloPlots

    class MockMonteCarlo:
        def __init__(self):
            self.results = {
                "apogee_x": [100, 200, 300],
                "apogee_y": [100, 200, 300],
                "x_impact": [1000, 2000, 3000],
                "y_impact": [1000, 2000, 3000],
            }
            self.filename = "test"

    mock_monte_carlo = MockMonteCarlo()
    plots = _MonteCarloPlots(mock_monte_carlo)

    with pytest.raises(
        ValueError, match="MonteCarlo object must have an 'environment' attribute"
    ):
        plots.ellipses(background="satellite")


def test_ellipses_background_no_latitude_longitude():
    """Test that ValueError is raised when environment has no latitude or longitude attributes.

    This test creates a mock environment without latitude and longitude attributes.
    """
    from rocketpy.plots.monte_carlo_plots import _MonteCarloPlots

    mock_environment = MagicMock()
    delattr(mock_environment, "latitude")
    delattr(mock_environment, "longitude")

    mock_monte_carlo = MagicMock()
    mock_monte_carlo.environment = mock_environment

    plots = _MonteCarloPlots(mock_monte_carlo)

    with pytest.raises(
        ValueError, match="Environment must have 'latitude' and 'longitude' attributes"
    ):
        plots.ellipses(background="satellite")


@patch("matplotlib.pyplot.show")
def test_ellipses_background_contextily_not_installed(
    mock_show, monte_carlo_calisto_pre_loaded
):
    """Test that a warning is issued when contextily is not installed.

    Parameters
    ----------
    mock_show :
        Mocks the matplotlib.pyplot.show() function to avoid showing the plots.
    monte_carlo_calisto_pre_loaded : MonteCarlo
        The MonteCarlo object, this is a pytest fixture.
    """
    try:
        from rocketpy.tools import import_optional_dependency as original_import

        # Create a mock function that only raises exception when importing contextily
        def mock_import_optional_dependency(name):
            if name == "contextily":
                raise ImportError("No module named 'contextily'")
            return original_import(name)

        with patch(
            "rocketpy.plots.monte_carlo_plots.import_optional_dependency",
            side_effect=mock_import_optional_dependency,
        ):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = monte_carlo_calisto_pre_loaded.plots.ellipses(
                    background="satellite"
                )
                assert result is None
                assert len(w) > 0
                assert any(
                    "contextily" in str(warning.message).lower() for warning in w
                )
    finally:
        _post_test_file_cleanup()


@patch("matplotlib.pyplot.show")
def test_ellipses_background_with_custom_xlim_ylim(
    mock_show, monte_carlo_calisto_pre_loaded
):
    """Test using background with custom xlim and ylim.

    Parameters
    ----------
    mock_show :
        Mocks the matplotlib.pyplot.show() function to avoid showing the plots.
    monte_carlo_calisto_pre_loaded : MonteCarlo
        The MonteCarlo object, this is a pytest fixture.
    """
    try:
        # Test using custom xlim and ylim
        result = monte_carlo_calisto_pre_loaded.plots.ellipses(
            background="satellite",
            xlim=(-5000, 5000),
            ylim=(-5000, 5000),
        )
        assert result is None
    finally:
        _post_test_file_cleanup()


@patch("matplotlib.pyplot.show")
def test_ellipses_background_save(mock_show, monte_carlo_calisto_pre_loaded):
    """Test using background with save=True.

    Parameters
    ----------
    mock_show :
        Mocks the matplotlib.pyplot.show() function to avoid showing the plots.
    monte_carlo_calisto_pre_loaded : MonteCarlo
        The MonteCarlo object, this is a pytest fixture.
    """
    try:
        # Test save functionality
        result = monte_carlo_calisto_pre_loaded.plots.ellipses(
            background="satellite", save=True
        )
        assert result is None
        # Verify file was created
        assert os.path.exists("monte_carlo_test.png")
    finally:
        _post_test_file_cleanup()
