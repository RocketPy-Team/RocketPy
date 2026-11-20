from abc import ABC

import matplotlib.pyplot as plt
import numpy as np

from .plot_helpers import show_or_save_plot


class _SensorPlots(ABC):
    """Base class that holds plot methods for a Sensor's measured data.

    The measured data layout is read from the sensor's ``channels`` class
    attribute -- a list of ``(label, unit)`` tuples describing each measured
    column (excluding the leading time column).
    """

    def __init__(self, sensor):
        self.sensor = sensor

    def _iter_runs(self):
        """Yield ``(run_index, run)`` for each list of measurements.

        A sensor added a single time to the rocket stores a flat list of
        tuples; a sensor added multiple times stores one such list per
        position (a nested list). The single-add case yields ``run_index``
        ``None``; the multi-add case yields the position index. Empty runs
        are skipped.
        """
        data = self.sensor.measured_data
        if not data:
            return
        if isinstance(data[0], list):  # sensor added multiple times -> nested
            for i, run in enumerate(data):
                if run:
                    yield i, run
        else:
            yield None, data

    def time_series(self, *, filename=None):
        """Plots each measured channel of the sensor against time, each
        measurement held until the next one.

        One subplot is created per channel. When the sensor was added to the
        rocket multiple times, each position is overlaid with a legend. To plot
        what a sensor read in a given flight, use ``flight.plots.sensors()``.

        Parameters
        ----------
        filename : str | None, optional
            The path the plot should be saved to. By default None, in which
            case the plot will be shown instead of saved.
        """
        runs = list(self._iter_runs())
        if not runs:
            print(f"No measured data recorded for sensor '{self.sensor.name}'.")
            return

        channels = self.sensor.channels
        multiple = runs[0][0] is not None
        n = len(channels)
        plt.figure(figsize=(9, 3 * n))
        for c, (label, unit) in enumerate(channels):
            ax = plt.subplot(n, 1, c + 1)
            for run_index, run in runs:
                arr = np.array(run, dtype=float)
                series_label = (
                    None if run_index is None else f"Instance {run_index + 1}"
                )
                ax.step(arr[:, 0], arr[:, c + 1], where="post", label=series_label)
            ax.set_title(f"{self.sensor.name} - {label}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(f"{label} ({unit})")
            ax.grid(True)
            if multiple:
                ax.legend()
        plt.subplots_adjust(hspace=0.5)
        show_or_save_plot(filename)

    def all(self):
        """Plots all the measured data of the sensor."""
        self.time_series()


class _AccelerometerPlots(_SensorPlots):
    """Class that holds plot methods for an Accelerometer's measured data."""


class _GyroscopePlots(_SensorPlots):
    """Class that holds plot methods for a Gyroscope's measured data."""


class _BarometerPlots(_SensorPlots):
    """Class that holds plot methods for a Barometer's measured data."""


class _GnssReceiverPlots(_SensorPlots):
    """Class that holds plot methods for a GnssReceiver's measured data."""
