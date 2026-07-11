import matplotlib.pyplot as plt

from .plot_helpers import show_or_save_plot


class _ControllerPlots:
    """Class that holds plot methods for Controller class.

    Attributes
    ----------
    _ControllerPlots.controller : Controller
        Controller object that will be used for the plots.
    """

    def __init__(self, controller):
        """Initializes _ControllerPlots class.

        Parameters
        ----------
        controller : Controller
            Instance of the Controller class.

        Returns
        -------
        None
        """
        self.controller = controller

    def control_history(self, *, filename=None):
        """Plots the recorded control state of every controlled object as a
        function of time, one axes per (object, variable) pair.

        Parameters
        ----------
        filename : str or None, optional
            Path to save the figure. If None, the figure is shown instead.

        Returns
        -------
        None
        """
        schedule = self.controller.recorded_schedule
        pairs = [
            (object_name, variable, function)
            for object_name, variables in schedule.items()
            for variable, function in variables.items()
        ]
        if not pairs:
            print(
                "No recorded control history - run a Flight with this controller first."
            )
            return

        fig, axes = plt.subplots(
            len(pairs), 1, figsize=(7, 3 * len(pairs)), sharex=True, squeeze=False
        )
        for ax, (object_name, variable, function) in zip(axes[:, 0], pairs):
            samples = self.controller.control_history[object_name][variable]
            times = [sample[0] for sample in samples]
            values = [sample[1] for sample in samples]
            ax.plot(times, values)
            ax.set_ylabel(variable)
            ax.set_title(f"{object_name}: {variable}")
            ax.grid(True)
        axes[-1, 0].set_xlabel("Time (s)")
        fig.suptitle(f"Control history - {self.controller.name}")
        fig.tight_layout()
        show_or_save_plot(filename)

    def all(self):
        """Plots all available controller plots.

        Returns
        -------
        None
        """
        self.control_history()


class _AirBrakesControllerPlots(_ControllerPlots):
    """Plots for AirBrakesController."""

    def deployment_level_curve(self, *, filename=None):
        """Plots the recorded deployment level as a function of time.

        Parameters
        ----------
        filename : str or None, optional
            Path to save the figure. If None, the figure is shown instead.

        Returns
        -------
        None
        """
        samples = self.controller.control_history.get("air_brakes", {}).get(
            "deployment_level", []
        )
        if not samples:
            print(
                "No recorded deployment level history - run a Flight with "
                "this controller first."
            )
            return
        times = [sample[0] for sample in samples]
        values = [sample[1] for sample in samples]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(times, values)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Deployment Level")
        ax.set_title(f"Deployment level - {self.controller.name}")
        ax.grid(True)
        fig.tight_layout()
        show_or_save_plot(filename)

    def all(self):
        """Plots all available air brakes controller plots.

        Returns
        -------
        None
        """
        self.deployment_level_curve()
