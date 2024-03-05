import matplotlib.pyplot as plt

from ..tools import generate_monte_carlo_ellipses


class _MonteCarloPlots:
    """Class to plot the monte carlo analysis results."""

    def __init__(self, monte_carlo):
        self.monte_carlo = monte_carlo

    def ellipses(
        self,
        image=None,
        actual_landing_point=None,
        perimeterSize=3000,
        xlim=(-3000, 3000),
        ylim=(-3000, 3000),
        save=False,
    ):
        """A function to plot the error ellipses for the apogee and impact
        points of the rocket. The function also plots the real landing point, if
        given

        Parameters
        ----------
        image : str, optional
            The path to the image to be used as the background
        actual_landing_point : tuple, optional
            A tuple containing the actual landing point of the rocket, if known.
            Useful when comparing the Monte Carlo results with the actual landing.
            Must be given in tuple format, such as (x, y) in meters.
            By default None.
        perimeterSize : int, optional
            The size of the perimeter to be plotted. The default is 3000.
        xlim : tuple, optional
            The limits of the x axis. The default is (-3000, 3000).
        ylim : tuple, optional
            The limits of the y axis. The default is (-3000, 3000).
        save : bool
            Whether save the output into a file or not. The default is False.
            If True, the .show() method won't be called, and the image will be
            saved with the same name as filename attribute, using a .png format.

        Returns
        -------
        None
        """
        # Import background map
        if image is not None:
            # TODO: use the optional import function
            try:
                from imageio import imread

                img = imread(image)
            except ImportError:
                raise ImportError(
                    "The 'imageio' package could not be. Please install it to add background images."
                )
            except FileNotFoundError:
                raise FileNotFoundError(
                    "The image file was not found. Please check the path."
                )

        (
            impact_ellipses,
            apogee_ellipses,
            apogee_x,
            apogee_y,
            impact_x,
            impact_y,
        ) = generate_monte_carlo_ellipses(self.monte_carlo.results)

        # Create plot figure
        plt.figure(num=None, figsize=(8, 6), dpi=150, facecolor="w", edgecolor="k")
        ax = plt.subplot(111)

        for ell in impact_ellipses:
            ax.add_artist(ell)
        for ell in apogee_ellipses:
            ax.add_artist(ell)

        # Draw launch point
        plt.scatter(0, 0, s=30, marker="*", color="black", label="Launch Point")
        # Draw apogee points
        plt.scatter(
            apogee_x, apogee_y, s=5, marker="^", color="green", label="Simulated Apogee"
        )
        # Draw impact points
        plt.scatter(
            impact_x,
            impact_y,
            s=5,
            marker="v",
            color="blue",
            label="Simulated Landing Point",
        )
        # Draw real landing point
        if actual_landing_point != None:
            plt.scatter(
                actual_landing_point[0],
                actual_landing_point[1],
                s=20,
                marker="X",
                color="red",
                label="Measured Landing Point",
            )

        plt.legend()

        # Add title and labels to plot
        ax.set_title(
            "1$\\sigma$, 2$\\sigma$ and 3$\\sigma$ "
            + "Monte Carlo Ellipses: Apogee and Landing Points"
        )
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
                    -perimeterSize - dx,
                    perimeterSize - dx,
                    -perimeterSize - dy,
                    perimeterSize - dy,
                ],
            )
        plt.axhline(0, color="black", linewidth=0.5)
        plt.axvline(0, color="black", linewidth=0.5)
        plt.xlim(*xlim)
        plt.ylim(*ylim)

        # Save plot and show result
        if save:
            plt.savefig(
                str(self.monte_carlo.filename) + ".png",
                bbox_inches="tight",
                pad_inches=0,
            )
        else:
            plt.show()

    def all(self, keys=None):
        """Plot the results of the Monte Carlo analysis.

        Parameters
        ----------
        keys : str, list or tuple, optional
            The keys of the results to be plotted. If None, all results will be
            plotted. The default is None.

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
                    "The selected 'keys' are not available in the results. "
                    "Please check the documentation."
                )
        else:
            raise ValueError(
                "The 'keys' argument must be a string, list or tuple. "
                "Please check the documentation."
            )
        for key in keys:
            plt.figure()
            plt.hist(
                self.monte_carlo.results[key],
            )
            plt.title(f"Histogram of {key}")
            plt.ylabel("Number of Occurrences")
            plt.show()
