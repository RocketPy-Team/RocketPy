import os
from unittest.mock import patch

import matplotlib as plt

plt.rcParams.update({"figure.max_open_warning": 0})


@patch("matplotlib.pyplot.show")
def test_dispersion(mock_show, dispersion):
    """Run a certain number of simulations in the dispersion loop and check if
    there are no errors.

    Parameters
    ----------
    dispersion : rocketpy.Dispersion
        Dispersion object to be tested.

    Returns
    -------
    None
    """

    dispersion.run_dispersion(
        number_of_simulations=10,
        append=False,
    )

    assert dispersion.allInfo() == None
    assert (
        dispersion.exportEllipsesToKML(
            filename="test_dispersion_class.kml", origin_lat=0, origin_lon=0
        )
        == None
    )
    assert dispersion.errors_log == []
    assert len(dispersion.outputs_log) == 10
    assert dispersion.plots.plot_ellipses(save=True) == None

    # Delete the test files
    os.remove("test_dispersion_class.disp_errors.txt")
    os.remove("test_dispersion_class.disp_inputs.txt")
    os.remove("test_dispersion_class.disp_outputs.txt")
    os.remove("test_dispersion_class.png")
    os.remove("test_dispersion_class.kml")
