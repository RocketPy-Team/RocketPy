import copy
import os
from unittest.mock import patch

import matplotlib as plt
import pytest

from rocketpy import Environment

plt.rcParams.update({"figure.max_open_warning": 0})


@pytest.mark.slow
@patch("matplotlib.pyplot.show")
def test_all_info(mock_show, env_analysis):  # pylint: disable=unused-argument
    """Test the EnvironmentAnalysis.all_info() method, which already invokes
    several other methods. It is a good way to test the whole class in a first view.
    However, if it fails, it is hard to know which method is failing.

    Parameters
    ----------
    env_analysis : rocketpy.EnvironmentAnalysis
        A simple object of the Environment Analysis class

    Returns
    -------
    None
    """
    assert env_analysis.info() is None
    assert env_analysis.all_info() is None
    assert env_analysis.plots.info() is None
    os.remove("wind_rose.gif")  # remove the files created by the method


@pytest.mark.slow
@patch("matplotlib.pyplot.show")
def test_exports(mock_show, env_analysis):  # pylint: disable=unused-argument
    """Check the export methods of the EnvironmentAnalysis class. It
    only checks if the method runs without errors. It does not check if the
    files are correct, as this would require a lot of work and would be
    difficult to maintain.

    Parameters
    ----------
    env_analysis : EnvironmentAnalysis
        A simple object of the EnvironmentAnalysis class.
    """

    assert env_analysis.export_mean_profiles() is None
    assert env_analysis.save("env_analysis_dict") is None

    env2 = copy.deepcopy(env_analysis)
    env2.load("env_analysis_dict")
    assert env2.all_info() is None

    # Delete file created by save method
    os.remove("env_analysis_dict")
    os.remove("wind_rose.gif")
    os.remove("export_env_analysis.json")


@pytest.mark.slow
@patch("matplotlib.pyplot.show")
def test_create_environment_object(mock_show, env_analysis):  # pylint: disable=unused-argument
    assert isinstance(env_analysis.create_environment_object(), Environment)
