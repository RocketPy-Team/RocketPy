import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import offset_copy

from ..tools import generate_monte_carlo_ellipses, import_optional_dependency


class _MonteCarloPlots:
    """Class to plot the Monte Carlo analysis results."""

    def __init__(self, monte_carlo):
        self.monte_carlo = monte_carlo

    # pylint: disable=too-many-statements
    def ellipses(
        self,
        image=None,
        actual_landing_point=None,
        perimeter_size=3000,
        xlim=(-3000, 3000),
        ylim=(-3000, 3000),
        save=False,
    ):
        """
        Plot the error ellipses for the apogee and impact points of the rocket.

        Parameters
        ----------
        image : str, optional
            Path to the background image, usually a map of the launch site.
        actual_landing_point : tuple, optional
            Actual landing point of the rocket in (x, y) meters.
        perimeter_size : int, optional
            Size of the perimeter to be plotted. Default is 3000.
        xlim : tuple, optional
            Limits of the x-axis. Default is (-3000, 3000). Values in meters.
        ylim : tuple, optional
            Limits of the y-axis. Default is (-3000, 3000). Values in meters.
        save : bool, optional
            Whether to save the plot as a file. Default is False. If True, the
            plot is saved and not displayed. If False, the plot is displayed.

        Returns
        -------
        None
        """

        imageio = import_optional_dependency("imageio")

        # Import background map
        if image is not None:
            try:
                img = imageio.imread(image)
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    "The image file was not found. Please check the path."
                ) from e

        try:
            apogee_x = np.array(self.monte_carlo.results["apogee_x"])
            apogee_y = np.array(self.monte_carlo.results["apogee_y"])
        except KeyError:
            print("No apogee data found. Skipping apogee ellipses.")
            apogee_x = np.array([])
            apogee_y = np.array([])
        try:
            impact_x = np.array(self.monte_carlo.results["x_impact"])
            impact_y = np.array(self.monte_carlo.results["y_impact"])
        except KeyError:
            print("No impact data found. Skipping impact ellipses.")
            impact_x = np.array([])
            impact_y = np.array([])

        if len(apogee_x) == 0 and len(impact_x) == 0:
            raise ValueError("No apogee or impact data found. Cannot plot ellipses.")

        impact_ellipses, apogee_ellipses = generate_monte_carlo_ellipses(
            apogee_x,
            apogee_y,
            impact_x,
            impact_y,
        )

        # Create plot figure
        plt.figure(figsize=(8, 6), dpi=150)
        ax = plt.subplot(111)

        for ell in impact_ellipses:
            ax.add_artist(ell)
        for ell in apogee_ellipses:
            ax.add_artist(ell)

        # Draw points
        plt.scatter(0, 0, s=30, marker="*", color="black", label="Launch Point")
        plt.scatter(
            apogee_x, apogee_y, s=5, marker="^", color="green", label="Simulated Apogee"
        )
        plt.scatter(
            impact_x,
            impact_y,
            s=5,
            marker="v",
            color="blue",
            label="Simulated Landing Point",
        )

        if actual_landing_point:
            plt.scatter(
                actual_landing_point[0],
                actual_landing_point[1],
                s=20,
                marker="X",
                color="red",
                label="Measured Landing Point",
            )

        plt.legend()

        ax.set_title("1$\\sigma$, 2$\\sigma$ and 3$\\sigma$ Monte Carlo Ellipses")
        north_south_offset = offset_copy(
            ax.transAxes, fig=plt.gcf(), x=-72, y=0, units="points"
        )
        east_west_offset = offset_copy(
            ax.transAxes, fig=plt.gcf(), x=0, y=-30, units="points"
        )
        ax.text(0, 0, "West", va="bottom", ha="center", transform=east_west_offset)
        ax.text(1, 0, "East", va="bottom", ha="center", transform=east_west_offset)
        ax.text(0, 0, "South", va="bottom", ha="left", transform=north_south_offset)
        ax.text(0, 1, "North", va="top", ha="left", transform=north_south_offset)
        ax.set_ylabel("Y (m)")
        ax.set_xlabel("X (m)")

        # Add background image to plot
        # TODO: In the future, integrate with other libraries to plot the map (e.g. cartopy, ee, etc.)
        # You can translate the basemap by changing dx and dy (in meters)
        dx = 0
        dy = 0
        if image is not None:
            plt.imshow(
                img,
                zorder=0,
                extent=[
                    -perimeter_size - dx,
                    perimeter_size - dx,
                    -perimeter_size - dy,
                    perimeter_size - dy,
                ],
            )

        plt.axhline(0, color="black", linewidth=0.5)
        plt.axvline(0, color="black", linewidth=0.5)
        plt.xlim(*xlim)
        plt.ylim(*ylim)

        if save:
            plt.savefig(
                f"{self.monte_carlo.filename}.png", bbox_inches="tight", pad_inches=0
            )
        else:
            plt.show()

    def all(self, keys=None):
        """
        Plot the histograms of the Monte Carlo simulation results.

        Parameters
        ----------
        keys : str, list or tuple, optional
            The keys of the results to be plotted. If None, all results will be
            plotted. Default is None.

        Returns
        -------
        None
        """
        if keys is None:
            keys = self.monte_carlo.results.keys()
        elif isinstance(keys, str):
            keys = [keys]
        elif isinstance(keys, (list, tuple)):
            keys = list(set(keys).intersection(self.monte_carlo.results.keys()))
            if len(keys) == 0:
                raise ValueError(
                    "The specified 'keys' are not available in the results."
                )
        else:
            raise ValueError("The 'keys' argument must be a string, list, or tuple.")
        for key in keys:
            # Create figure with GridSpec
            fig = plt.figure(figsize=(8, 8))
            gs = fig.add_gridspec(2, 1, height_ratios=[1, 3])

            # Create subplots using gridspec
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])

            # Plot boxplot
            # TODO: changes vert to orientation="horizontal" when support for Py3.9 ends
            ax1.boxplot(self.monte_carlo.results[key], vert=False)
            ax1.set_title(f"Box Plot of {key}")
            ax1.set_yticks([])

            # Plot histogram
            ax2.hist(self.monte_carlo.results[key])
            ax2.set_title(f"Histogram of {key}")
            ax2.set_ylabel("Number of Occurrences")
            ax1.set_xticks([])

            plt.tight_layout()
            plt.show()

    def plot_comparison(self, other_monte_carlo):
        """
        Plot the histograms of the Monte Carlo simulation results.

        Parameters
        ----------
        other_monte_carlo : MonteCarlo
            MonteCarlo object which the current one will be compared to.

        Returns
        -------
        None
        """
        original_parameters_set = set(self.monte_carlo.processed_results.keys())
        other_parameters_set = set(other_monte_carlo.processed_results.keys())
        intersection_set = original_parameters_set.intersection(other_parameters_set)

        fill_colors = ["red", "blue"]
        for key in intersection_set:
            # Create figure with GridSpec
            fig = plt.figure(figsize=(8, 8))
            gs = fig.add_gridspec(2, 1, height_ratios=[1, 3])

            # Create subplots using gridspec
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])

            # Plot boxplot
            bp = ax1.boxplot(
                [other_monte_carlo.results[key], self.monte_carlo.results[key]],
                vert=False,
                tick_labels=["Other", "Original"],
                patch_artist=True,
            )
            for patch, color in zip(bp["boxes"], fill_colors):
                patch.set_facecolor(color)
            ax1.set_title(f"Box Plot of {key}")

            # Plot histogram
            ax2.hist(
                self.monte_carlo.results[key],
                alpha=0.5,
                color="blue",
                label="Original",
                density=True,
            )
            ax2.hist(
                other_monte_carlo.results[key],
                alpha=0.5,
                color="red",
                label="Other",
                density=True,
            )
            ax2.set_title(f"Histogram of {key}")
            ax2.set_ylabel("Density")

            plt.tight_layout()
            plt.legend()
            plt.show()

    # pylint: disable=too-many-statements
    def ellipses_comparison(
        self,
        other_monte_carlo,
        image=None,
        perimeter_size=3000,
        xlim=(-3000, 3000),
        ylim=(-3000, 3000),
        save=False,
    ):
        """
        Plot the error ellipses for the apogee and impact points of the rocket.

        Parameters
        ----------
        other_monte_carlo : MonteCarlo
            MonteCarlo object which the current one will be compared to.
        image : str, optional
            Path to the background image, usually a map of the launch site.
        perimeter_size : int, optional
            Size of the perimeter to be plotted. Default is 3000.
        xlim : tuple, optional
            Limits of the x-axis. Default is (-3000, 3000). Values in meters.
        ylim : tuple, optional
            Limits of the y-axis. Default is (-3000, 3000). Values in meters.
        save : bool, optional
            Whether to save the plot as a file. Default is False. If True, the
            plot is saved and not displayed. If False, the plot is displayed.

        Returns
        -------
        None
        """
        imageio = import_optional_dependency("imageio")

        # Import background map
        if image is not None:
            try:
                img = imageio.imread(image)
            except FileNotFoundError as e:  # pragma no cover
                raise FileNotFoundError(
                    "The image file was not found. Please check the path."
                ) from e

        try:
            original_apogee_x = np.array(self.monte_carlo.results["apogee_x"])
            original_apogee_y = np.array(self.monte_carlo.results["apogee_y"])
            other_apogee_x = np.array(other_monte_carlo.results["apogee_x"])
            other_apogee_y = np.array(other_monte_carlo.results["apogee_y"])
        except KeyError:
            print("No apogee data found. Skipping apogee ellipses.")
            original_apogee_x = np.array([])
            original_apogee_y = np.array([])
            other_apogee_x = np.array([])
            other_apogee_y = np.array([])
        try:
            original_impact_x = np.array(self.monte_carlo.results["x_impact"])
            original_impact_y = np.array(self.monte_carlo.results["y_impact"])
            other_impact_x = np.array(other_monte_carlo.results["x_impact"])
            other_impact_y = np.array(other_monte_carlo.results["y_impact"])
        except KeyError:
            print("No impact data found. Skipping impact ellipses.")
            original_impact_x = np.array([])
            original_impact_y = np.array([])
            other_impact_x = np.array([])
            other_impact_y = np.array([])

        if (
            len(original_apogee_x) == 0 and len(original_impact_x) == 0
        ):  # pragma no cover
            raise ValueError("No apogee or impact data found. Cannot plot ellipses.")

        original_impact_ellipses, original_apogee_ellipses = (
            generate_monte_carlo_ellipses(
                original_apogee_x,
                original_apogee_y,
                original_impact_x,
                original_impact_y,
                apogee_rgb=(0.0117647, 0.1490196, 0.9882352),
                impact_rgb=(0.9882352, 0.0117647, 0.6392156),
            )
        )

        other_impact_ellipses, other_apogee_ellipses = generate_monte_carlo_ellipses(
            other_apogee_x,
            other_apogee_y,
            other_impact_x,
            other_impact_y,
            apogee_rgb=(0.9882352, 0.8509803, 0.0117647),
            impact_rgb=(0.0117647, 0.9882352, 0.3607843),
        )

        # Create plot figure
        plt.figure(figsize=(8, 6), dpi=150)
        ax = plt.subplot(111)

        # Draw ellipses and points for original monte carlo
        for ell in original_impact_ellipses:
            ax.add_artist(ell)
        for ell in original_apogee_ellipses:
            ax.add_artist(ell)

        plt.scatter(0, 0, s=30, marker="*", color="black", label="Launch Point")
        plt.scatter(
            original_apogee_x,
            original_apogee_y,
            s=5,
            marker="^",
            color="#0326FC",
            label="Original Apogee",
        )
        plt.scatter(
            original_impact_x,
            original_impact_y,
            s=5,
            marker="v",
            color="#FC03A3",
            label="Original Landing Point",
        )

        # Draw ellipses and points for other monte carlo
        for ell in other_impact_ellipses:
            ax.add_artist(ell)
        for ell in other_apogee_ellipses:
            ax.add_artist(ell)

        plt.scatter(
            other_apogee_x,
            other_apogee_y,
            s=5,
            marker="^",
            color="#FCD903",
            label="Other Apogee",
        )
        plt.scatter(
            other_impact_x,
            other_impact_y,
            s=5,
            marker="v",
            color="#03FC5C",
            label="Other Landing Point",
        )

        plt.legend()
        ax.set_title("1$\\sigma$, 2$\\sigma$ and 3$\\sigma$ Monte Carlo Ellipses")
        ax.set_ylabel("North (m)")
        ax.set_xlabel("East (m)")

        # Add background image to plot
        # TODO: In the future, integrate with other libraries to plot the map (e.g. cartopy, ee, etc.)
        # You can translate the basemap by changing dx and dy (in meters)
        dx = 0
        dy = 0
        if image is not None:
            plt.imshow(
                img,
                zorder=0,
                extent=[
                    -perimeter_size - dx,
                    perimeter_size - dx,
                    -perimeter_size - dy,
                    perimeter_size - dy,
                ],
            )

        plt.axhline(0, color="black", linewidth=0.5)
        plt.axvline(0, color="black", linewidth=0.5)
        plt.xlim(*xlim)
        plt.ylim(*ylim)

        if save:
            plt.savefig(
                f"{self.monte_carlo.filename}.png", bbox_inches="tight", pad_inches=0
            )
        else:
            plt.show()
